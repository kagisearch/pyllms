from __future__ import annotations

import asyncio
import os
import statistics
import typing as t
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from logging import getLogger

from prettytable import PrettyTable

from ._providers import PROVIDER_MAP, Provider
from .providers.base import AsyncProvider, AsyncStreamResult, Result, StreamProvider, StreamResult, SyncProvider

LOGGER = getLogger(__name__)


SyncAPIResult = list[Result]
AsyncAPIResult = t.Awaitable[SyncAPIResult]
APIResult = t.Union[SyncAPIResult, AsyncAPIResult]


@dataclass
class LLMS:
    _models: dict[str, Provider] = field(default_factory=dict)

    def __post_init__(self):
        default_model = os.getenv("LLMS_DEFAULT_MODEL") or "gpt-3.5-turbo"
        try:
            self.add_provider(default_model)
        except ValueError:
            warnings.warn(f"Default model {default_model} not found in any provider", stacklevel=2)

    @classmethod
    def single_provider(cls, model: str, api_key: str | None = None, **kwargs):
        return cls().add_provider(model, api_key, **kwargs)

    def add_provider(self, provider_name: str, model: str | None = None, api_key: str | None = None, **kwargs):
        provider = PROVIDER_MAP[provider_name]
        if provider.api_key_name:
            api_key = api_key or os.getenv(provider.api_key_name)
            if not api_key:
                msg = f"{provider.api_key_name} environment variable is required"
                raise Exception(msg)

        self._models[provider_name] = provider.kind(api_key=api_key or "", model=model, **kwargs)
        return self

    def stream_models(self) -> list[StreamProvider]:
        return [m for m in self._models.values() if isinstance(m, StreamProvider)]

    def async_models(self) -> list[AsyncProvider]:
        return [m for m in self._models.values() if isinstance(m, AsyncProvider)]

    def sync_models(self) -> list[SyncProvider]:
        return list(self._models.values())

    def to_list(self, query: str | None = None) -> list[dict[str, t.Any]]:
        return [
            {
                "provider": provider.__name__,
                "name": model,
                "cost": cost,
            }
            for provider in self._models.values()
            for model, cost in provider.MODEL_INFO.items()
            if not query or (query.lower() in model.lower() or query.lower() in provider.__name__.lower())
        ]

    def count_tokens(self, content: str | list[dict[str, t.Any]]) -> int | list[int]:
        results = [provider.count_tokens(content) for provider in self._models.values()]
        return results if len(self._models) > 1 else results[0]

    def complete(self, prompt: str, **kwargs: t.Any) -> SyncAPIResult:
        def sync_run(p):
            return p.complete(prompt, **kwargs)

        with ThreadPoolExecutor() as executor:
            return list(executor.map(sync_run, self.sync_models()))

    def acomplete(self, prompt: str, **kwargs: t.Any) -> AsyncAPIResult:
        async def gather():
            return await asyncio.gather(*[p.acomplete(prompt, **kwargs) for p in self.async_models()])

        return gather()

    def complete_stream(self, prompt: str, **kwargs: t.Any) -> StreamResult:
        sm = self.stream_models()
        if len(sm) > 1:
            msg = "Streaming is possible only with a single model"
            raise ValueError(msg)
        return sm[0].complete_stream(prompt, **kwargs)

    def acomplete_stream(self, prompt: str, **kwargs: t.Any) -> AsyncStreamResult:
        sm = self.stream_models()
        if len(sm) > 1:
            msg = "Streaming is possible only with a single model"
            raise ValueError(msg)
        return sm[0].acomplete_stream(prompt, **kwargs)

    def benchmark(
        self,
        problems: list[tuple[str, str]] | None = None,
        evaluator: SyncProvider | None = None,
        show_outputs: bool = False,
        delay: float = 0,
        **kwargs: t.Any,
    ) -> tuple[PrettyTable, PrettyTable]:
        from . import _bench as bench

        problems = problems or bench.PROBLEMS

        model_results = {}

        # Run completion tasks in parallel for each model, but sequentially for each prompt within a model
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(bench.process_prompts_sequentially, model, problems, evaluator, delay, **kwargs)
                for model in self._models.values()
            ]

            for future in as_completed(futures):
                try:
                    (
                        model,
                        outputs,
                        evaluation_queue,
                        evaluation_threads,
                    ) = future.result()
                    if not outputs:
                        continue

                    model_results[model] = {
                        "outputs": outputs,
                        "total_latency": 0,
                        "total_cost": 0,
                        "evaluation": [None] * len(outputs),
                    }

                    for output_data in outputs:
                        if output_data:  # Check if output_data is not None
                            model_results[model]["total_latency"] += output_data["latency"]
                            model_results[model]["total_cost"] += output_data["cost"]

                    if evaluator and evaluation_threads:
                        # Wait for all evaluation threads to complete
                        for thread in evaluation_threads:
                            if thread:  # Check if thread exists
                                thread.join()

                        # Process all evaluation results
                        while not evaluation_queue.empty():
                            index, evaluation = evaluation_queue.get()
                            model_results[model]["evaluation"][index] = evaluation
                except Exception as e:
                    print(f"Error processing results: {str(e)}")
                    # Don't add failed models to results
                    continue

        for model in model_results:
            outputs = model_results[model]["outputs"]
            model_results[model]["median_latency"] = statistics.median([output["latency"] for output in outputs])

            total_tokens = sum([output["tokens"] for output in outputs])
            total_latency = model_results[model]["total_latency"]
            model_results[model]["aggregated_speed"] = total_tokens / total_latency

        if evaluator:
            sorted_models = sorted(
                model_results,
                key=lambda x: model_results[x]["aggregated_speed"] * sum(filter(None, model_results[x]["evaluation"])),
                reverse=True,
            )
        else:
            sorted_models = sorted(
                model_results,
                key=lambda x: model_results[x]["aggregated_speed"],
                reverse=True,
            )

        headers = [
            "Model",
            "Output",
            "Tokens",
            "Cost ($)",
            "Latency (s)",
            "Speed (tokens/sec)",
            "Evaluation",
        ]

        if not show_outputs:
            headers.remove("Output")

        if not evaluator:
            headers.remove("Evaluation")

        table = PrettyTable(headers)

        for model in sorted_models:
            model_data = model_results[model]

            total_tokens = 0
            total_score = 0
            valid_evaluations = 0
            for index, output_data in enumerate(model_data["outputs"]):
                total_tokens += output_data["tokens"]
                if evaluator and model_results[model]["evaluation"][index] is not None:
                    total_score += model_results[model]["evaluation"][index]
                    valid_evaluations += 1
                row_data = [
                    str(model),
                    output_data["text"],
                    output_data["tokens"],
                    f"{output_data['cost']:.5f}",
                    f"{output_data['latency']:.2f}",
                    f"{output_data['tokens'] / output_data['latency']:.2f}",
                ]
                if not show_outputs:
                    row_data.remove(output_data["text"])
                if evaluator:
                    row_data.append(model_results[model]["evaluation"][index])
                table.add_row(row_data)

            if show_outputs:
                row_data = [
                    str(model),
                    "",
                    f"{total_tokens}",
                    f"{model_data['total_cost']:.5f}",
                    f"{model_data['median_latency']:.2f}",
                    f"{total_tokens / model_data['total_latency']:.2f}",
                ]

            else:
                row_data = [
                    str(model),
                    f"{total_tokens}",
                    f"{model_data['total_cost']:.5f}",
                    f"{model_data['median_latency']:.2f}",
                    f"{total_tokens / model_data['total_latency']:.2f}",
                ]
            if evaluator:
                if valid_evaluations > 0:
                    acc = 100 * total_score / valid_evaluations
                    row_data.append(f"{acc:.2f}%")
                else:
                    row_data.append("N/A")

            table.add_row(row_data)

        # Track easiest and hardest questions
        easiest_questions = []
        hardest_questions = []
        for i, problem in enumerate(problems):
            all_correct = all(model_results[model]["evaluation"][i] == 1 for model in model_results)
            all_incorrect = all(model_results[model]["evaluation"][i] == 0 for model in model_results)

            if all_correct:
                easiest_questions.append((i, problem[0]))
            elif all_incorrect:
                hardest_questions.append((i, problem[0]))

        # Create a new table for easiest and hardest questions
        questions_table = PrettyTable(["Category", "Index", "Question"])
        questions_table.align["Question"] = "l"  # Left-align the Question column

        for index, question in easiest_questions:
            questions_table.add_row(
                [
                    "Easiest",
                    index,
                    question[:100] + ("..." if len(question) > 100 else ""),
                ]
            )

        for index, question in hardest_questions:
            questions_table.add_row(
                [
                    "Hardest",
                    index,
                    question[:100] + ("..." if len(question) > 100 else ""),
                ]
            )

        return table, questions_table

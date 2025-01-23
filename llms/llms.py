import asyncio
import builtins
import concurrent.futures
import os
import queue
import re
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging import getLogger
from typing import Any, Optional, Union

from prettytable import PrettyTable

from ._providers import PROVIDER_MAP, Provider
from .providers.base_provider import BaseProvider
from .results.result import AsyncStreamResult, Result, Results, StreamResult

LOGGER = getLogger(__name__)


class LLMS:
    _providers: list[BaseProvider] = []
    _models: list[str] = []

    def __init__(self, model: Union[str, list[str], None] = None, **kwargs: Any) -> None:
        """Programmatically load api keys and instantiate providers."""
        self._load_api_keys(kwargs)
        self._set_models(model)
        self._initialize_providers(kwargs)

    def __repr__(self) -> str:
        return f"LLMS({','.join(self._models)})"

    @property
    def n_provider(self) -> int:
        return len(self._providers)

    def list(self, query: Optional[str] = None) -> list[dict[str, Any]]:
        return [
            {
                "provider": provider.__name__,
                "name": model,
                "cost": cost,
            }
            for provider in [p.provider for p in self._provider_map.values()]
            for model, cost in provider.MODEL_INFO.items()
            if not query or (query.lower() in model.lower() or query.lower() in provider.__name__.lower())
        ]

    def count_tokens(self, content: Union[str, builtins.list[dict[str, Any]]]) -> Union[int, builtins.list[int]]:
        results = [provider.count_tokens(content) for provider in self._providers]
        return results if self.n_provider > 1 else results[0]

    def _process_completion(self, prompt: str, is_async: bool, **kwargs: Any) -> Union[Result, Results]:
        async def _async_generate(provider):
            return await provider.acomplete(prompt, **kwargs)

        def _sync_generate(provider):
            return provider.complete(prompt, **kwargs)

        if self.n_provider > 1:
            if is_async:

                async def gather_results():
                    return await asyncio.gather(*[_async_generate(provider) for provider in self._providers])

                results = asyncio.run(gather_results())
            else:
                with ThreadPoolExecutor() as executor:
                    results = list(executor.map(_sync_generate, self._providers))
            return Results(results)
        provider = self._providers[0]
        return _async_generate(provider) if is_async else _sync_generate(provider)

    def complete(self, prompt: str, **kwargs: Any) -> Union[Result, Results]:
        return self._process_completion(prompt, is_async=False, **kwargs)

    async def acomplete(self, prompt: str, **kwargs: Any) -> Union[Result, Results]:
        return await self._process_completion(prompt, is_async=True, **kwargs)

    def complete_stream(self, prompt: str, **kwargs: Any) -> StreamResult:
        if self.n_provider > 1:
            msg = "Streaming is possible only with a single model"
            raise ValueError(msg)
        return self._providers[0].complete_stream(prompt, **kwargs)

    async def acomplete_stream(self, prompt: str, **kwargs: Any) -> AsyncStreamResult:
        if self.n_provider > 1:
            msg = "Streaming is possible only with a single model"
            raise ValueError(msg)
        return await self._providers[0].acomplete_stream(prompt, **kwargs)

    def benchmark(
        self,
        problems: Optional[builtins.list[tuple[str, str]]] = None,
        evaluator: Optional[BaseProvider] = None,
        show_outputs: bool = False,
        html: bool = False,
        delay: float = 0,
        **kwargs: Any,
    ) -> Union[PrettyTable, str]:
        if not problems:
            from ._bench import PROBLEMS

            problems = PROBLEMS

        def evaluate_answers(evaluator, query_answer_pairs: list[tuple[str, str, str]]) -> list[int]:
            system = """You are an evaluator for an AI system. Your task is to determine whether the AI's answer matches the correct answer. You will be given two inputs: the AI's answer and the correct answer. Your job is to compare these and output a binary score: 1 if the AI's answer is correct, and 0 if it is not.

        To evaluate the AI's performance:
        1. Carefully compare the AI's answer to the correct answer.
        2. Consider the following:
           - Does the AI's answer convey the same meaning as the correct answer?
           - Are there any significant discrepancies or omissions in the AI's answer?
           - If there are minor differences in wording but the core information is the same, consider it correct.

        After your evaluation, provide your assessment in the following format:
        <evaluation>
        [Your reasoning for the score]
        </evaluation>
        <score>[0 or 1]</score>

        Remember, output only 0 (not correct) or 1 (correct) as the final score. Do not include any additional explanation or text outside of the specified tags."""

            scores = []
            for i, (_query, correct_answer, ai_answer) in enumerate(query_answer_pairs, start=1):
                prompt = f"""Here is the AI's answer:
        <ai_answer>
        {ai_answer}
        </ai_answer>
        Here is the correct answer:
        <correct_answer>
        {correct_answer}
        </correct_answer>"""

                evaluator_result = evaluator.complete(prompt, system_message=system).text
                # print(correct_answer, ai_answer, evaluator_result)

                # Extract the score from the evaluator's response
                score_match = re.search(r"<score>(\d)</score>", evaluator_result)
                if score_match:
                    score = int(score_match.group(1))
                    scores.append(score)
                else:
                    msg = f"Could not extract score from evaluator's response for query {i}"
                    raise ValueError(msg)

            return scores

        model_results = {}

        def process_prompt(model, prompt, index, evaluator, evaluation_queue, **kwargs):
            try:
                print(model, index)  # , prompt[0])
                result = model.complete(prompt[0], max_tokens=1000, temperature=0, **kwargs)
                if delay > 0:
                    time.sleep(delay)
                output_data = {
                    "text": result.text,
                    "tokens": result.meta["tokens_completion"],
                    "latency": result.meta["latency"],
                    "cost": result.meta["cost"],
                    "prompt_index": index,
                }
            except Exception as e:
                print(f"Error with {model}: {str(e)}")
                return None

            if evaluator:
                evaluation_thread = threading.Thread(
                    target=lambda: evaluation_queue.put(
                        (
                            index,
                            evaluate_answers(evaluator, [(prompt[0], prompt[1], result.text)])[0],
                        )
                    )
                )
                evaluation_thread.start()
                output_data["evaluation_thread"] = evaluation_thread

            return output_data

        def process_prompts_sequentially(model, prompts, evaluator, **kwargs):
            results = []
            evaluation_queue = queue.Queue()
            evaluation_threads = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                futures = [
                    executor.submit(
                        process_prompt,
                        model,
                        prompt,
                        index,
                        evaluator,
                        evaluation_queue,
                        **kwargs,
                    )
                    for index, prompt in enumerate(prompts)
                ]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result is not None:
                        results.append(result)
                        if evaluator and "evaluation_thread" in result:
                            evaluation_threads.append(result.get("evaluation_thread"))
            return model, results, evaluation_queue, evaluation_threads

        # Run completion tasks in parallel for each model, but sequentially for each prompt within a model
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_prompts_sequentially, model, problems, evaluator, **kwargs)
                for model in self._providers
            ]

            for future in as_completed(futures):
                try:
                    (
                        model,
                        outputs,
                        evaluation_queue,
                        evaluation_threads,
                    ) = future.result()
                    if not outputs:  # Skip if no successful outputs
                        continue

                    if outputs:  # Only process if we have valid outputs
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

        # Return both tables
        return table, questions_table

        if not html:
            return table, questions_table
        return table.get_html_string(), questions_table.get_html_string()

    def _load_api_keys(self, kwargs: dict[str, Any]) -> None:
        self._provider_map = {
            name: Provider(
                provider=provider.provider,
                api_key_name=provider.api_key_name,
                api_key=kwargs.pop(provider.api_key_name.lower(), None) or os.getenv(provider.api_key_name)
                if provider.api_key_name
                else None,
                needs_api_key=provider.needs_api_key,
            )
            for name, provider in PROVIDER_MAP.items()
        }

    def _set_models(self, model: Optional[Union[str, builtins.list[str]]]) -> None:
        default_model = os.getenv("LLMS_DEFAULT_MODEL") or "gpt-3.5-turbo"
        self._models = [default_model] if model is None else ([model] if isinstance(model, str) else model)

    def _validate_model(self, single_model: str, provider: Provider) -> bool:
        return single_model in provider.provider.MODEL_INFO and (provider.api_key or not provider.needs_api_key)

    def _initialize_providers(self, kwargs: dict[str, Any]) -> None:
        self._providers = [
            provider.provider(
                model=single_model,
                **({**kwargs, "api_key": provider.api_key} if provider.needs_api_key else kwargs),
            )
            for single_model in self._models
            for provider in self._provider_map.values()
            if self._validate_model(single_model, provider)
        ]

        if not self._providers:
            msg = "No valid providers found for the specified models"
            raise ValueError(msg)

        for provider in self._providers:
            LOGGER.info(f"Initialized {provider.model} with {provider.__class__.__name__}")

import os
import markdown2
import statistics
from prettytable import PrettyTable
from .providers import OpenAIProvider
from .providers import AnthropicProvider
from .providers import AI21Provider
from concurrent.futures import ThreadPoolExecutor, as_completed


class Result:
    def __init__(self, results):
        self._results = results

    @property
    def text(self):
        if len(self._results) == 1:
            return self._results[0]["text"]
        return [result["text"] for result in self._results]

    @property
    def html(self):
        return [markdown2.markdown(result["text"]) for result in self._results]

    @property
    def meta(self):
        if len(self._results) == 1:
            return self._results[0]["meta"]
        return [result["meta"] for result in self._results]


class LLMS:
    def __init__(
        self, model=None, openai_api_key=None, anthropic_api_key=None, ai21_api_key=None
    ):
        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")

        if anthropic_api_key is None:
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

        if ai21_api_key is None:
            ai21_api_key = os.getenv("AI21_API_KEY")

        if model is None:
            model = ["gpt-3.5-turbo"]
        elif isinstance(model, str):
            model = [model]

        self._providers = []
        for single_model in model:
            if (
                openai_api_key is not None
                and single_model in OpenAIProvider.MODEL_COSTS
            ):
                self._providers.append(
                    OpenAIProvider(api_key=openai_api_key, model=single_model)
                )
            elif (
                anthropic_api_key is not None
                and single_model in AnthropicProvider.MODEL_COSTS
            ):
                self._providers.append(
                    AnthropicProvider(api_key=anthropic_api_key, model=single_model)
                )
            elif ai21_api_key is not None and single_model in AI21Provider.MODEL_COSTS:
                self._providers.append(
                    AI21Provider(api_key=ai21_api_key, model=single_model)
                )
            else:
                raise ValueError("Invalid API key and model combination", single_model)

    def list(self, query=None):
        model_info_list = []

        all_providers = [OpenAIProvider, AI21Provider, AnthropicProvider]

        for provider in all_providers:
            for model, cost in provider.MODEL_COSTS.items():
                if query and ((query.lower() not in model.lower()) and (query.lower() not in provider.__name__.lower())):
                    continue
                model_info = {
                    "provider": provider.__name__,
                    "name": model,
                    "cost": cost,
                }
                model_info_list.append(model_info)

        sorted_list = sorted(
            model_info_list, key=lambda x: x["cost"]["prompt"] + x["cost"]["completion"]
        )
        return sorted_list

    def complete(self, prompt, **kwargs):
        def _generate(provider):
            response = provider.complete(prompt, **kwargs)
            return {
                "text": response["text"],
                "meta": response["meta"],
                "provider": provider,
            }

        results = []
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(_generate, provider): provider
                for provider in self._providers
            }
            for future in as_completed(futures):
                results.append(future.result())

        return Result(results)

    def benchmark(self, show_outputs=False, html=False):
        prompts = [
            "What is the capital of France?",
            "Explain the process of photosynthesis.",
            "How does a combustion engine work?",
            "What are the primary colors?",
            "What is the Pythagorean theorem?",
            "Give me five suitable names for a baby born on startship Enterprise",
            "Summarize social cognitive theory in two sentences",
            "what is the recipe for world's best omlette?",
            "How to make money online?",
        ]

        model_results = {}

        # Helper function to run a completion task and store the result
        def process_prompt(model, prompt):
            result = model.complete(prompt)
            output_data = {
                "text": result["text"],
                "tokens": result["meta"]["tokens"],
                "latency": result["meta"]["latency"],
                "cost": result["meta"]["cost"],
            }
            return model, output_data

        with ThreadPoolExecutor() as executor:
            model_results = {}
            futures = []
            for model in self._providers:
                model_results[model] = {
                    "outputs": [],
                    "total_latency": 0,
                    "total_cost": 0,
                }
                for prompt in prompts:
                    future = executor.submit(process_prompt, model, prompt)
                    futures.append((model, future))

            for future in as_completed([f[1] for f in futures]):
                model, output_data = next((f for f in futures if f[1] == future))
                result = future.result()
                model_results[model]["outputs"].append(result[1])
                model_results[model]["total_latency"] += result[1]["latency"]
                model_results[model]["total_cost"] += result[1]["cost"]

        for model in model_results:
            outputs = model_results[model]["outputs"]
            model_results[model]["median_latency"] = statistics.median(
                [output["latency"] for output in outputs]
            )

        sorted_models = sorted(
            model_results, key=lambda x: model_results[x]["median_latency"]
        )

        headers = [
            "Model",
            "Output",
            "Tokens",
            "Cost ($)",
            "Latency (s)",
            "Speed (tokens/sec)",
        ]
        if not show_outputs:
            headers.remove("Output")

        table = PrettyTable(headers)

        for model in sorted_models:
            model_data = model_results[model]
            total_tokens = 0
            for output_data in model_data["outputs"]:
                total_tokens += output_data["tokens"]
                row_data = [
                    model,
                    output_data["text"],
                    output_data["tokens"],
                    f'{output_data["cost"]:.5f}',
                    f'{output_data["latency"]:.2f}',
                    f'{output_data["tokens"]/output_data["latency"]:.2f}',
                ]
                if not show_outputs:
                    row_data.remove(output_data["text"])
                table.add_row(row_data)
            if show_outputs:
                table.add_row(
                    [
                        model,
                        "",
                        f"Total Tokens: {total_tokens}",
                        f"Total Cost: {model_data['total_cost']:.5f}",
                        f"Median Latency: {model_data['median_latency']:.2f}",
                        f"Aggregrated speed:: {total_tokens/model_data['total_latency']:.2f}",
                    ]
                )
            else:
                table.add_row(
                    [
                        model,
                        f"Total Tokens: {total_tokens}",
                        f"Total Cost: {model_data['total_cost']:.5f}",
                        f"Median Latency: {model_data['median_latency']:.2f}",
                        f"Aggregrated speed: {total_tokens/model_data['total_latency']:.2f}",
                    ]
                )
        if not html:
            return table
        else:
            return table.get_html_string()

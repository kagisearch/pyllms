import os
import markdown2
import statistics
from prettytable import PrettyTable
from .providers import OpenAIProvider
from .providers import AnthropicProvider
from .providers import AI21Provider
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple


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
            model = os.getenv("LLMS_DEFAULT_MODEL")
            if model is None:
                model = ["gpt-3.5-turbo"]
            else:
                model = [model]
        elif isinstance(model, str):
            model = [model]

        self._providers = []
        for single_model in model:
            if (
                openai_api_key is not None
                and single_model in OpenAIProvider.MODEL_INFO
            ):
                self._providers.append(
                    OpenAIProvider(api_key=openai_api_key, model=single_model)
                )
            elif (
                anthropic_api_key is not None
                and single_model in AnthropicProvider.MODEL_INFO
            ):
                self._providers.append(
                    AnthropicProvider(api_key=anthropic_api_key, model=single_model)
                )
            elif ai21_api_key is not None and single_model in AI21Provider.MODEL_INFO:
                self._providers.append(
                    AI21Provider(api_key=ai21_api_key, model=single_model)
                )
            else:
                raise ValueError("Invalid API key and model combination", single_model)

    def list(self, query=None):
        model_info_list = []

        all_providers = [OpenAIProvider, AI21Provider, AnthropicProvider]

        for provider in all_providers:
            for model, cost in provider.MODEL_INFO.items():
                if query and (
                    (query.lower() not in model.lower())
                    and (query.lower() not in provider.__name__.lower())
                ):
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

    def benchmark(self, prompts=None, evaluator=None, show_outputs=False, html=False):
        
        if not prompts:
            prompts = [
                "What is the capital of the country where Christopher Columbus was born?",
                "A glass door has ‘push’ written on it in mirror writing. Should you push or pull it and why?",
                "Explain the process of photosynthesis in plants, including the role of chlorophyll and the importance of light. Discuss the main outcomes of photosynthesis and why it is essential for life on Earth.",
                "Solve The Two Door Riddle: You are in a room with two doors. One door leads to certain death, and the other leads to freedom. There are two guards, one in front of each door. One guard always tells the truth, and the other guard always lies. You can only ask one question to one guard to determine which door leads to freedom. What question should you ask?",
                "Solve the quadratic equation: x^2 - 5x + 6 = 0",
                "How much is 7! + 7? Describe steps you took.",
                "Describe the differences between depth-first search (DFS) and breadth-first search (BFS) algorithms in graph traversal, and provide an example use case for each of them.",
                "Write a Python function that takes a string of text as input and returns a dictionary containing the frequency of each word in the text. Discuss the time complexity of your solution and any possible improvements to optimize its performance.",
                "Write a Python function that takes a list of integers as input, finds the two numbers with the largest product, and returns their product. Also, provide a brief explanation of the logic behind your solution.",
                "You are given a string containing a sequence of ASCII characters. Your task is to write a JavaScript function that compresses the string by replacing consecutive occurrences of the same character with the character followed by the number of times it appears consecutively. Then, write a JavaScript function to decompress the compressed string back to its original form. The input string only contains printable ASCII characters. For example, if the input string is 'aaabccddd', the compressed string should be 'a3b1c2d3'. The decompressed string should be the same as the input string.",
                "Given the following messy and unstructured data, extract the names, email addresses, and phone numbers of the individuals listed:\
    \
    John Doe - johndoe@email.com (555) 123-4567\
    random text 1\
    Jane Smith\
    random text 2, 555-\
    987-6543\
    janesmith@email.com\
    random text 3\
    Bob Johnson - bob.johnson@email.com\
    random text 4 area code 555 phone: 111-2222\
    ",
            ]

        def evaluate_answers(
            evaluator, query_answer_pairs: List[Tuple[str, str]]
        ) -> List[int]:
            system='''
            You are a truthful evaluator of the capabilties of other AI models.

You are given a list of queries and answers by an AI model. For each query first think about  the solution yourself, then score the reply of the other AI, compared to yours on a scale 1 to 10 (10 being great).

For example:

Query: What is the capital of the country where Christopher Columbus was born?
Answer: Christopher Columbus was born in Genoa, Italy.

Query : A glass door has ‘push’ written on it in mirror writing. Should you push or pull it and why?
Answer: You should push the door. The reason for this is that the mirror writing is intended for people on the other side of the door to read, not for you. So, if you push the door, you will be pushing it in the direction that the people on the other side are expecting it to move.


Christopher Columbus was born in the Republic of Genoa, which is now part of Italy. The capital of Italy is Rome. So you would score it 3 (it is wrong answer, but city was correct)
Since the word "push" is written in mirror writing, it suggests that the instruction is intended for people on the other side of the door. Therefore, you should pull the door to open it.
You would score this 1 (it is wrong)


Your only output should be a list of comma seperated integers representing your evaluation score for each answer. No other output is allowed. For example above your output will be:
1, 3

'''
            #prompt = "Please evaluate the following answers on a scale of 1 to 10 (10 being the best):\n\n"
            prompt=""
            for i, (query, answer) in enumerate(query_answer_pairs):
                prompt += f"Query #{i + 1}: {query}\nAnswer #{i + 1}: {answer}\n\n"
#            prompt += "Please provide a score for each answer as a list of integers separated by commas, with no additional text or explanation. For example: 6, 10, 10"
            print(prompt)
            evaluator_result = evaluator.complete(prompt, system=system).text
            print(evaluator_result)
            scores = evaluator_result.split(",")
            return [int(score.strip()) for score in scores]

        model_results = {}

        def process_prompt(model, prompt, index):
            print(model, index)
            result = model.complete(prompt)
            output_data = {
                "text": result["text"],
                "tokens": result["meta"]["tokens"],
                "latency": result["meta"]["latency"],
                "cost": result["meta"]["cost"],
                "prompt_index": index,
            }
            return output_data

        def process_prompts_sequentially(model, prompts):
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                futures = [
                    executor.submit(process_prompt, model, prompt, index)
                    for index, prompt in enumerate(prompts)
                ]
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
            return model, results

        # Run completion tasks in parallel for each model, but sequentially for each prompt within a model
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_prompts_sequentially, model, prompts)
                for model in self._providers
            ]

            for future in as_completed(futures):
                model, outputs = future.result()
                model_results[model] = {
                    "outputs": outputs,
                    "total_latency": 0,
                    "total_cost": 0,
                }

                for output_data in outputs:
                    model_results[model]["total_latency"] += output_data["latency"]
                    model_results[model]["total_cost"] += output_data["cost"]

        for model in model_results:
            outputs = model_results[model]["outputs"]
            model_results[model]["median_latency"] = statistics.median(
                [output["latency"] for output in outputs]
            )

            total_tokens = sum([output["tokens"] for output in outputs])
            total_latency = model_results[model]["total_latency"]
            model_results[model]["aggregated_speed"] = total_tokens / total_latency

        if evaluator:
            for model in model_results:
                all_query_answer_pairs = []
                model_data = model_results[model]
                for output_data in model_data["outputs"]:
                    prompt_index = output_data["prompt_index"]
                    all_query_answer_pairs.append(
                        (prompts[prompt_index], output_data["text"])
                    )

                evaluation = evaluate_answers(evaluator, all_query_answer_pairs)
                # Add evaluation to results
                model_results[model]["evaluation"] = []
                for i in range(len(model_results[model]["outputs"])):
                    model_results[model]["evaluation"].append(evaluation[i])

            sorted_models = sorted(
                model_results,
                key=lambda x: model_results[x]["aggregated_speed"]
                * sum(model_results[x]["evaluation"]),
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
            for index, output_data in enumerate(model_data["outputs"]):
                total_tokens += output_data["tokens"]
                if evaluator:
                    total_score += model_results[model]["evaluation"][index]
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
                if evaluator:
                    row_data.append(model_results[model]["evaluation"][index])
                table.add_row(row_data)

            if show_outputs:
                row_data = [
                    model,
                    "",
                    f"Total Tokens: {total_tokens}",
                    f"Total Cost: {model_data['total_cost']:.5f}",
                    f"Median Latency: {model_data['median_latency']:.2f}",
                    f"Aggregated speed: {total_tokens/model_data['total_latency']:.2f}",
                ]

            else:
                row_data = [
                    model,
                    f"Total Tokens: {total_tokens}",
                    f"Total Cost: {model_data['total_cost']:.5f}",
                    f"Median Latency: {model_data['median_latency']:.2f}",
                    f"Aggregated speed: {total_tokens/model_data['total_latency']:.2f}",
                ]
            if evaluator:
                row_data.append(f"Total Score: {total_score}")

            table.add_row(row_data)

        if not html:
            return table
        else:
            return table.get_html_string()

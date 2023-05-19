import asyncio
import os
import re
import statistics
from prettytable import PrettyTable
from .providers import OpenAIProvider
from .providers import AnthropicProvider
from .providers import AI21Provider
from .providers import CohereProvider
from .providers import AlephAlphaProvider
from .providers import HuggingfaceHubProvider
from .providers import GoogleProvider
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple


class Result:
    def __init__(self, results):
        self._results = results

    @property
    def text(self):
        if isinstance(self._results, list):
            return [result["text"] for result in self._results]
        else:
            return self._results["text"]

    @property
    def meta(self):
        if isinstance(self._results, list):
            return [result["meta"] for result in self._results]
        else:
            return self._results["meta"]


class LLMS:
    def __init__(
        self,
        model=None,
        openai_api_key=None,
        anthropic_api_key=None,
        ai21_api_key=None,
        cohere_api_key=None,
        alephalpha_api_key=None,
        huggingfacehub_api_key=None,
    ):
        if openai_api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")

        if anthropic_api_key is None:
            anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

        if ai21_api_key is None:
            ai21_api_key = os.getenv("AI21_API_KEY")

        if cohere_api_key is None:
            cohere_api_key = os.getenv("COHERE_API_KEY")

        if alephalpha_api_key is None:
            alephalpha_api_key = os.getenv("ALEPHALPHA_API_KEY")

        if huggingfacehub_api_key is None:
            huggingfacehub_api_key = os.getenv("HUGGINFACEHUB_API_KEY")

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
            if openai_api_key is not None and single_model in OpenAIProvider.MODEL_INFO:
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
            elif (
                cohere_api_key is not None and single_model in CohereProvider.MODEL_INFO
            ):
                self._providers.append(
                    CohereProvider(api_key=cohere_api_key, model=single_model)
                )
            elif (
                alephalpha_api_key is not None
                and single_model in AlephAlphaProvider.MODEL_INFO
            ):
                self._providers.append(
                    AlephAlphaProvider(api_key=alephalpha_api_key, model=single_model)
                )
            elif (
                huggingfacehub_api_key is not None
                and single_model in HuggingfaceHubProvider.MODEL_INFO
            ):
                self._providers.append(
                    HuggingfaceHubProvider(
                        api_key=huggingfacehub_api_key, model=single_model
                    )
                )
            elif single_model in GoogleProvider.MODEL_INFO:
                self._providers.append(GoogleProvider(model=single_model))
            else:
                raise ValueError("Invalid API key and model combination", single_model)

    def __repr__(self) -> str:
        return f"LLMS({self.model})"

    @property
    def n_provider(self):
        return len(self._providers)

    def list(self, query=None):
        model_info_list = []

        all_providers = [
            OpenAIProvider,
            AI21Provider,
            AnthropicProvider,
            CohereProvider,
            AlephAlphaProvider,
            HuggingfaceHubProvider,
            GoogleProvider,
        ]

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

    def count_tokens(self, content):
        results = []
        for provider in self._providers:
            results.append(provider.count_tokens(content))
        if self.n_provider > 1:
            return results
        else:
            return results[0]

    def complete(self, prompt, **kwargs):
        def _generate(provider):
            result = provider.complete(prompt, **kwargs)
            return result

        if self.n_provider > 1:
            results = []
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(_generate, provider): provider
                    for provider in self._providers
                }
                for future in as_completed(futures):
                    results.append(future.result())

            return Result(results)
        else:
            result = self._providers[0].complete(prompt, **kwargs)
            return Result(result)

    async def acomplete(
        self,
        prompt: str,
        **kwargs,
    ):
        if self.n_provider > 1:
            tasks = [
                provider.acomplete(prompt, **kwargs) for provider in self._providers
            ]
            results = await asyncio.gather(*tasks, return_exceptions=False)
            return Result(results)

        else:
            provider = self._providers[0]
            result = await provider.acomplete(prompt, **kwargs)
            return Result(result)

    def complete_stream(self, prompt, **kwargs):
        if len(self._providers) > 1:
            raise ValueError("Streaming is possible only with a single model")

        yield from self._providers[0].complete_stream(prompt, **kwargs)

    def benchmark(self, problems=None, evaluator=None, show_outputs=False, html=False):
        if not problems:
          
            problems = [
               
                (
                    "A glass door has ‘push’ written on it in mirror writing. Should you push or pull it and why?",
                    "pull",
                ),
                ("Solve the quadratic equation: x^2 - 5x + 6 = 0", "x=2, x=3"),
                ("How much is 7! * 3! -1234.5 ?", "29005.5"),
                (
                    'translate this sentence by alternating words in gemran and french "it was a beautiful day that thursday and I want skiing outside. it started raining soon although they said it won\'t be until friday, so I went to the pool instead"',
                    "",
                ),
                ("Convert December 21 1:50pm pacific to taipei time", "5:50 am"),
                (
                    "In my kitchen there's a table with a cup with a ball inside. I moved the cup to my bed in my bedroom and turned the cup upside down. I grabbed the cup again and moved to the main room. Where's the ball now?",
                    "on the bed in the bedroom",
                ),
                (
                    'Capture the essence of this in exactly 7 words: "There’s much that divides us in Northern Ireland though one thing is guaranteed to bring us together: local phrases. Call it slang, call it colloquialisms, we all know only too well how important words are to where we’re from . . . and when it comes to the phrases that make us ‘us,’ we’ve got a lot to say. While you don’t need advance knowledge of the words to fit in, well, it helps. How else will you know where ‘foundered’ sits on the scale of warm to freezing? Or deciding whether that new car purchase is more ‘clinker’ than ‘beezer’? Or appreciating that ‘grand’ can mean exactly that or anything but? If the best way to get to know a nation is to understand their language, then surely tourists must be at times confused about what comes out of our mouths. Throughout the island of Ireland, we have utterly brilliant ways to verbally express ourselves.“I think it’s really important,” says Dr Frank Ferguson, research director for English Language and Literature at Ulster University, about the vitality of slang as part of language."',
                    "Make sure the answer has exactly 7 words",
                ),
                (
                    "Write a Python function that takes a list of integers as input and returns the length of the longest increasing subsequence. An increasing subsequence is a subsequence of the given list where the elements are in strictly increasing order. Your function should have an efficient solution with a time complexity better than O(n^2), where n is the length of the input list. Output only code with no explainations and provide example usage.",
                    "",
                ),
                (
                    "Write a Python function that takes a list of integers as input and returns the maximum sum of non-adjacent elements in the list. The function should return 0 if the input list is empty. Your function should have an efficient solution with a time complexity of O(n), where n is the length of the input list. Output only code with no explainations and provide example usage.",
                    "",
                ),
                (
                    "You are given a 2D binary matrix filled with 0's and 1's. Your task is to write a JavaScript function that finds the largest rectangle containing only 1's and returns its area. Your function should have an efficient solution with a time complexity better than O(n^3), where n is the total number of elements in the input matrix. Output only code with no explainations and provide example usage.",
                    "",
                ),
                (
                    "Given the following messy and unstructured data, extract the clean names, email addresses, and phone numbers (as digits) of the individuals listed:\
John Doe - johndoe (at) email.com (five-five-five) one-two-three-four-five-six-seven\
random text not a phone 123 4468888\
Jane Smith\
random text 2, cinque-cinque-cinque-\
nove-otto-sette-sei-quattro-tre\
janesmith en email punto com\
texto aleatorio 3\
Bob Johnson - first name dot last name dot wild🐻@email.com\
texto aleatorio 4 código de área five-five-five teléfono: eins-eins-eins-zwei-zwei-zwei-zwei",
                    "Name: John Doe\
Email: johndoe@email.com \
Phone: 5551234567\
\
Name: Jane Smith \
Email: janesmith@email.com\
Phone: 5559876432\
\
Name: Bob Johnson\
Email: first.name.wild🐻@email.com\
Phone: 5551112222\
",
                ),
                ("write three sentences each ending with the word apple", ""),
                ("Please count the number of t in eeooeotetto", "3"),
                (
                    "Use m to substitute p, a to substitute e, n to substitute a, g to substitute c, o to substitute h,\
how to spell peach under this rule?",
                    "mango",
                ),
                (
                    "two workers paint the fence in 8 hours. how long will it take one worker paint the same fence if they are injured and need to take a 30 min break after every hour of work?",
                    "5.5 hours",
                ),
                ("5+55+555+5555+55555-1725=", "60000"),
                ("-2-2-2-2-2-2*-2*-2=", "-18"),
                ('what is the 13th letter of the word "supralapsarian"', 'a'),
                ('Vlad\'s uncle can still beat him in sprinting although he is 30 years younger. who is "he" referring to?', 'Vlad'),
                ("Belgium uses 160 million litres of petrol each day. Three is enough petrol stored to last 60 days. how much more petrol does Belgium need to buy to have enough stored for 90 days. A) 4 million litres, B) 4,8 million litres, C) 480 million litres D) 160 million litres E) 4800 million litres", "E) 4800 million litres"),
                ("The sum of three numbers is 96. The first number is 6 times the third number, and the third number is 40 less than the second number. What is the absolute value of the difference between the first and second numbers?", "5"),
                ("The least common multiple of a positive integer n and 18 is 180, and the greatest common divisor of n and 45 is 15. What is the sum of the digits of n?", "n = 60 thus the answer is 6"),
                ("what square is the black king on in this chess position: 1Bb3BN/R2Pk2r/1Q5B/4q2R/2bN4/4Q1BK/1p6/1bq1R1rb w - - 0 1", "e7")

            ]

        def evaluate_answers(
            evaluator, query_answer_pairs: List[Tuple[str, str]]
        ) -> List[int]:
            system = """
You are given a problem and answer by a student. Sometimes the correct solution will be provided with the problem as a hint, but if it is not, then first think about the solution yourself, then score the solution of the student's solution with one of these scores:
0 - Incorrect solution
3 - Correct solution

Your output should be using this template:
Score: #
"""
            scores = []
            for i, (query, hint, answer) in enumerate(query_answer_pairs, start=1):
                if not len(hint):
                    prompt = f"Problem: {query}\nStudent solution: {answer}"
                else:
                    prompt = f"Problem: {query}\nHint (correct answer): {hint}\nStudent solution: {answer}"
#                print(prompt)
                evaluator_result = evaluator.complete(
                    prompt, system_message=system
                ).text
#                print(evaluator_result)
                found = re.search(r"Score: (\d+)", evaluator_result)
                if found:
                    scores.append(int(found.group(1)))
                else:
                    print("No score found!", evaluator_result)

#            print(scores)
            return scores

        model_results = {}

        def process_prompt(model, prompt, index):
            print(model, index)
            result = model.complete(prompt, max_tokens=1000, temperature=0)
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
                    executor.submit(process_prompt, model, prompt[0], index)
                    for index, prompt in enumerate(prompts)
                ]
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
            return model, results

        # Run completion tasks in parallel for each model, but sequentially for each prompt within a model
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_prompts_sequentially, model, problems)
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
                        (
                            problems[prompt_index][0],
                            problems[prompt_index][1],
                            output_data["text"],
                        )
                    )

                evaluation = evaluate_answers(evaluator, all_query_answer_pairs)
                # Add evaluation to results
                model_results[model]["evaluation"] = []
                for i in range(len(model_results[model]["outputs"])):
                    model_results[model]["evaluation"].append(evaluation[i])
            print(model_results)
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
                    str(model),
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
                    str(model),
                    "",
                    f"Total Tokens: {total_tokens}",
                    f"Total Cost: {model_data['total_cost']:.5f}",
                    f"Median Latency: {model_data['median_latency']:.2f}",
                    f"Aggregated speed: {total_tokens/model_data['total_latency']:.2f}",
                ]

            else:
                row_data = [
                    str(model),
                    f"Total Tokens: {total_tokens}",
                    f"Total Cost: {model_data['total_cost']:.5f}",
                    f"Median Latency: {model_data['median_latency']:.2f}",
                    f"Aggregated speed: {total_tokens/model_data['total_latency']:.2f}",
                ]
            if evaluator:
                acc=100*total_score/(3*len(model_data["evaluation"]))
                row_data.append(f"Accuracy: {acc:.2f}%")

            table.add_row(row_data)

        if not html:
            return table
        else:
            return table.get_html_string()

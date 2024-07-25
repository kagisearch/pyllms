import asyncio
import os
import re
import statistics
import threading
import queue
from dataclasses import dataclass
from prettytable import PrettyTable
from .providers import OpenAIProvider
from .providers import AnthropicProvider
from .providers import BedrockAnthropicProvider
from .providers import AI21Provider
from .providers import CohereProvider
from .providers import AlephAlphaProvider
from .providers import HuggingfaceHubProvider
from .providers import GoogleProvider
from .providers import GoogleGenAIProvider
from .providers import MistralProvider
from .providers import OllamaProvider
from .providers.base_provider import BaseProvider
from .results.result import AsyncStreamResult, Result, Results, StreamResult
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Tuple, Type, Union
from logging import getLogger


LOGGER = getLogger(__name__)


@dataclass
class Provider:
    provider: Type[BaseProvider]
    api_key_name: Optional[str] = None
    api_key: Optional[str] = None
    needs_api_key: bool = True


class LLMS:
    _possible_providers: List[Provider] = [
        Provider(OpenAIProvider, api_key_name="OPENAI_API_KEY"),
        Provider(AnthropicProvider, api_key_name="ANTHROPIC_API_KEY"),
        Provider(BedrockAnthropicProvider, needs_api_key=False),
        Provider(AI21Provider, api_key_name="AI21_API_KEY"),
        Provider(CohereProvider, api_key_name="COHERE_API_KEY"),
        Provider(AlephAlphaProvider, api_key_name="ALEPHALPHA_API_KEY"),
        Provider(HuggingfaceHubProvider, api_key_name="HUGGINFACEHUB_API_KEY"),
        Provider(GoogleGenAIProvider, api_key_name="GOOGLE_API_KEY"),
        Provider(MistralProvider, api_key_name="MISTRAL_API_KEY"),
        Provider(GoogleProvider, needs_api_key=False),
        Provider(OllamaProvider, needs_api_key=False)
    ]
    _providers: List[BaseProvider] = []
    _models: List[str] = []

    def __init__(self,
                 model: Union[str, List[str], None] = None,
                 **kwargs
                 ):
        """Programmatically load api keys and instantiate providers."""

        for provider in [p for p in self._possible_providers if p.api_key_name]:
            assert provider.api_key_name  # for static type checking only
            api_key = None
            if provider.api_key_name.lower() in kwargs:  # get api key from kwargs
                api_key = kwargs.pop(provider.api_key_name.lower())
            elif provider.api_key_name in os.environ:  # otherwise, get it from environment variable
                api_key = os.getenv(provider.api_key_name)
            provider.api_key = api_key

        if model is None:  # if no model is specified, use default: from environment variable or gpt-3.5-turbo
            default_model = os.getenv("LLMS_DEFAULT_MODEL") or "gpt-3.5-turbo"
            self._models = [default_model]
        else:
            self._models = [model] if isinstance(model, str) else model

        self._providers = []
        for single_model in self._models:
            for provider in self._possible_providers:
                if single_model in provider.provider.MODEL_INFO:
                    LOGGER.info(f"Found {single_model} in {provider.provider.__name__}")
                    if provider.api_key:
                        self._providers.append(provider.provider(api_key=provider.api_key, model=single_model))
                    elif not provider.needs_api_key:
                        self._providers.append(provider.provider(model=single_model, **kwargs))
                    else:
                        raise ValueError("Invalid API key and model combination", single_model)

    def __repr__(self) -> str:
        return f"LLMS({','.join(self._models)})"

    @property
    def n_provider(self):
        return len(self._providers)

    def list(self, query=None):
        model_info_list = []

        for provider in [p.provider for p in self._possible_providers]:
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

    def complete(self, prompt: str, **kwargs) -> Union[Result, Results]:
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

            return Results(results)
        else:
            return self._providers[0].complete(prompt, **kwargs)

    async def acomplete(
        self,
        prompt: str,
        **kwargs,
    ) -> Union[Result, Results]:
        if self.n_provider > 1:
            tasks = [
                provider.acomplete(prompt, **kwargs) for provider in self._providers
            ]
            results = await asyncio.gather(*tasks, return_exceptions=False)
            return Results(results)

        else:
            provider = self._providers[0]
            return await provider.acomplete(prompt, **kwargs)

    def complete_stream(self, prompt, **kwargs) -> StreamResult:
        if self.n_provider > 1:
            raise ValueError("Streaming is possible only with a single model")
        return self._providers[0].complete_stream(prompt, **kwargs)

    async def acomplete_stream(self, prompt, **kwargs) -> AsyncStreamResult:
        if self.n_provider > 1:
            raise ValueError("Streaming is possible only with a single model")
        return await self._providers[0].acomplete_stream(prompt, **kwargs)

    def benchmark(self, problems=None, evaluator=None, show_outputs=False, html=False, **kwargs):
        if not problems:
            problems = [
                (
                    "A glass door has ‘push’ written on it in mirror writing. Should you push or pull it and why?",
                    "pull",
                ),
                ('Given the string: "A# B# #B A# A# #B #B A# A# #B A# A#" Could you check for any instances of "A# #B" and replace them with "B# #A"? print only the answer', "B# B# #A B# B# #A #A B# B# #A B# B#"),
                ("Kevin currently has 8 apples. He ate 3 apples yesterday. How many apples does Kevin have now?", "8"),
                (
                    'What is the largest land animal? If that animal has wings, answer "The African Elephant". Otherwise, answer "The Mouse". Do not provide any explanation for your choice.',
                    "The Mouse",
                ),
                ("Convert December 21 1:50pm pacific to taipei time", "5:50 am"),
                (
                    "In my kitchen there's a table with a cup with a ball inside. I moved the cup to my bed in my bedroom and turned the cup upside down. I grabbed the cup again and moved to the main room. Where's the ball now?",
                    "on the bed in the bedroom",
                ),
                (
                    """using System;struct a{static int Main(){object[]c={"\u0048e\x6c\x6co "+(C\u0068ar)(86+1)+"or\x6c\x64"};typeof(Conso\u006ce).GetMet\u0068o\u0064s()[101].Invoke(c,c);return 0;}}
                    
                    What does this code do in one sentence?""",
                    'prints "Hello World" to the console',
                ),
                (
                    """#include <stdio.h> 

#define N(a)       "%"#a"$hhn"
#define O(a,b)     "%10$"#a"d"N(b)
#define U          "%10$.*37$d"
#define G(a)       "%"#a"$s"
#define H(a,b)     G(a)G(b)
#define T(a)       a a 
#define s(a)       T(a)T(a)
#define A(a)       s(a)T(a)a
#define n(a)       A(a)a
#define D(a)       n(a)A(a)
#define C(a)       D(a)a
#define R          C(C(N(12)G(12)))
#define o(a,b,c)   C(H(a,a))D(G(a))C(H(b,b)G(b))n(G(b))O(32,c)R
#define SS         O(78,55)R "\n\033[2J\n%26$s";
#define E(a,b,c,d) H(a,b)G(c)O(253,11)R G(11)O(255,11)R H(11,d)N(d)O(253,35)R
#define S(a,b)     O(254,11)H(a,b)N(68)R G(68)O(255,68)N(12)H(12,68)G(67)N(67)

char* fmt = O(10,39)N(40)N(41)N(42)N(43)N(66)N(69)N(24)O(22,65)O(5,70)O(8,44)N(
            45)N(46)N    (47)N(48)N(    49)N( 50)N(     51)N(52)N(53    )O( 28,
            54)O(5,        55) O(2,    56)O(3,57)O(      4,58 )O(13,    73)O(4,
            71 )N(   72)O   (20,59    )N(60)N(61)N(       62)N (63)N    (64)R R
            E(1,2,   3,13   )E(4,    5,6,13)E(7,8,9        ,13)E(1,4    ,7,13)E
            (2,5,8,        13)E(    3,6,9,13)E(1,5,         9,13)E(3    ,5,7,13
            )E(14,15,    16,23)    E(17,18,19,23)E(          20, 21,    22,23)E
            (14,17,20,23)E(15,    18,21,23)E(16,19,    22     ,23)E(    14, 18,
            22,23)E(16,18,20,    23)R U O(255 ,38)R    G (     38)O(    255,36)
            R H(13,23)O(255,    11)R H(11,36) O(254    ,36)     R G(    36 ) O(
            255,36)R S(1,14    )S(2,15)S(3, 16)S(4,    17 )S     (5,    18)S(6,
            19)S(7,20)S(8,    21)S(9    ,22)H(13,23    )H(36,     67    )N(11)R
            G(11)""O(255,    25 )R        s(C(G(11)    ))n (G(          11) )G(
            11)N(54)R C(    "aa")   s(A(   G(25)))T    (G(25))N         (69)R o
            (14,1,26)o(    15, 2,   27)o   (16,3,28    )o( 17,4,        29)o(18
            ,5,30)o(19    ,6,31)o(        20,7,32)o    (21,8,33)o       (22 ,9,
            34)n(C(U)    )N( 68)R H(    36,13)G(23)    N(11)R C(D(      G(11)))
            D(G(11))G(68)N(68)R G(68)O(49,35)R H(13,23)G(67)N(11)R C(H(11,11)G(
            11))A(G(11))C(H(36,36)G(36))s(G(36))O(32,58)R C(D(G(36)))A(G(36))SS

#define arg d+6,d+8,d+10,d+12,d+14,d+16,d+18,d+20,d+22,0,d+46,d+52,d+48,d+24,d\
            +26,d+28,d+30,d+32,d+34,d+36,d+38,d+40,d+50,(scanf(d+126,d+4),d+(6\
            -2)+18*(1-d[2]%2)+d[4]*2),d,d+66,d+68,d+70, d+78,d+80,d+82,d+90,d+\
            92,d+94,d+97,d+54,d[2],d+2,d+71,d+77,d+83,d+89,d+95,d+72,d+73,d+74\
            ,d+75,d+76,d+84,d+85,d+86,d+87,d+88,d+100,d+101,d+96,d+102,d+99,d+\
            67,d+69,d+79,d+81,d+91,d+93,d+98,d+103,d+58,d+60,d+98,d+126,d+127,\
            d+128,d+129

char d[538] = {1,0,10,0,10};

int main() {
    while(*d) printf(fmt, arg);
}

what does this program do in one sentence?""",
                    "it is an obfuscated implementation of a Tic Tac Toe game",
                ),
                ("How many r's are in strawberry?", "3"),
                (
                    "Use  g to substitute c, m to substitute p, a to substitute e, o to substitute h and n to substitute a\
how to spell cheap under this rule?",
                    "goanm",
                ),
                (
                    "two workers paint the fence in 8 hours. how long will it take one worker paint the same fence if they are injured and need to take a 30 min break after every hour of work?",
                    "23.5 hours",
                ),
                ("Alan, Bob, Colin, Dave and Emily are standing in a circle. Alan is on Bob’s immediate left. Bob is on Colin’s immediate left. Colin is on Dave’s immediate left. Dave is on Emily’s immediate left. Who is on Alan’s immediate right?", "Bob"),
                ("-2-2-2-2-2-2*-2*-2-2/-2=", "-17"),
                ('what is the 13th letter of the word "supralapsarian"', "a"),
                (
                    'Vlad\'s uncle can still beat him in sprinting although he is 30 years younger. who is "he" referring to?',
                    "Vlad",
                ),
                (
                    "A farmer and a sheep are standing on one side of a river. There is a boat with enough room for one human and one animal. what is the fewest number of trips for farmer to get across the river with the sheep?",
                    "one"
                ),
                (
                    "Words: cmomittee, te, unnimously, agred, t, implment, te, nw, plocy, aftr, throgh, discusion, an, consdration\
                    Rearrange the words and fix typos to form a meaningful sentence","Answer should match this exactly: The committee unanimously agreed to implement the new policy after thorough discussion and consideration."
                (
                    "The least common multiple of a positive integer n and 18 is 180, and the greatest common divisor of n and 45 is 15. What is the sum of the digits of n?",
                    "n = 60 thus the answer is 6",
                ),
                ("""The sky was a brilliant shade of blue, dotted with fluffy white clouds.
                In the above sentence add the word 'green' after 'blue' and 'black' after 'clouds'. then remove words 'shade' and 'clouds'. then move the word 'sky' forward three words. Substitute "brilliant" with "vibrant". Replace "dotted" with "adorned". Move "was a" to the end of the sentence. what is left?""",
                "Answer should match this exactly: The vibrant sky of blue green, adorned with fluffy white black was a."
                ),
                ("""Begin with these sentence:
"The quick brown fox jumps over the lazy dog. A wizard's job is to vex chumps quickly in fog."
Now, follow these instructions:

Remove all words containing the letter 'i'.""",
                    "Answer should match this exactly: The brown fox jumps over the lazy dog. A job to vex chumps fog."
                ),
                ("""Begin with these sentence:
"The quick brown fox jumps over the lazy dog. A wizard's job is to vex chumps quickly in fog."
Now, follow these instructions:
- Insert the word "cybernetic" after every word ending in 's'."""
                "Answer should match this exactly: A wizard's cybernetic job is cybernetic to vex chumps cybernetic quickly in fog."
                ),
                (
                    "what square is the black king on in this chess position: 1Bb3BN/R2Pk2r/1Q5B/4q2R/2bN4/4Q1BK/1p6/1bq1R1rb w - - 0 1",
                    "e7",
                ),
                (
                    "An arrow points up. We rotate it 90 degrees to the clockwise, mirror it along its flat end, and rotate it another 90 degrees clockwise. Which direction is it pointing?",
                    "up",
                ),
                ("Current flight information (the following flights are one-way only, and all the flights available are included below):\n\
There is a flight from city G to city B\n\
There is a flight from city H to city K\n\
There is a flight from city L to city M\n\
There is a flight from city F to city H\n\
There is a flight from city G to city J\n\
There is a flight from city B to city I\n\
There is a flight from city L to city A\n\
There is a flight from city H to city N\n\
There is a flight from city B to city D\n\
There is a flight from city J to city C\n\
Question: Is there a series of flights that goes from city F to city I?", "No"),
            ('Bob (a boy) has 3 sisters. Each sister has 2 brothers. How many brothers does Bob have?', '1'),
            ('Imagine there is a circular pond in an oasis, with two trees at the edge of the pond, on opposite sides. Bob sets up a hammock by hanging it between the two trees. He gets into the hammock and falls asleep. If he were to roll over in his sleep and fall out of the hammock, where would he fall?',
                'water, in the center of the pond'),
            ]


        def evaluate_answers(evaluator, query_answer_pairs: List[Tuple[str, str, str]]) -> List[int]:
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
            for i, (query, correct_answer, ai_answer) in enumerate(query_answer_pairs, start=1):
                prompt = f"""Here is the AI's answer:
        <ai_answer>
        {ai_answer}
        </ai_answer>
        Here is the correct answer:
        <correct_answer>
        {correct_answer}
        </correct_answer>"""

                evaluator_result = evaluator.complete(prompt, system_message=system).text
                #print(ai_answer, correct_answer, evaluator_result)
                
                # Extract the score from the evaluator's response
                score_match = re.search(r'<score>(\d)</score>', evaluator_result)
                if score_match:
                    score = int(score_match.group(1))
                    scores.append(score)
                else:
                    raise ValueError(f"Could not extract score from evaluator's response for query {i}")

            return scores
    
        def evaluate_answers2(
            evaluator, query_answer_pairs: List[Tuple[str, str]]
        ) -> List[int]:
            system = """
You are given a problem and student's solution. If the correct answer is provided use it, otherwise first think about the solution yourself, then score the the student's solution with one of these scores:
0 - Student provided incorrect or no solution
3 - Student provided correct solution

Your output should be using always this template:
Score: #
"""
            scores = []
            for i, (query, hint, answer) in enumerate(query_answer_pairs, start=1):
                if not len(hint):
                    prompt = f"Problem: {query}\nStudent solution: {answer}"
                else:
                    prompt = f"Problem: {query}\nCorrect answer: {hint}\nStudent solution: {answer}"
                #                print(prompt)
                evaluator_result = evaluator.complete(
                    prompt, system_message=system
                ).text
                #print(prompt)
                #print(system)
                #print(evaluator_result)
                found = re.search(r"Score: (\d+)", evaluator_result)
                if found:
                    scores.append(int(found.group(1)))
                else:
                    #print("No score found!", evaluator_result)
                    scores.append(0)

            return scores

        model_results = {}

        def process_prompt(model, prompt, index, evaluator, evaluation_queue, **kwargs):
            print(model, index)
            result = model.complete(prompt[0], max_tokens=1000, temperature=0, **kwargs)
            output_data = {
                "text": result.text,
                "tokens": result.meta["tokens_completion"],
                "latency": result.meta["latency"],
                "cost": result.meta["cost"],
                "prompt_index": index,
            }
            
            if evaluator:
                evaluation_thread = threading.Thread(
                    target=lambda: evaluation_queue.put((index, evaluate_answers(evaluator, [(prompt[0], prompt[1], result.text)])[0]))
                )
                evaluation_thread.start()
                output_data['evaluation_thread'] = evaluation_thread
            
            return output_data

        def process_prompts_sequentially(model, prompts, evaluator, **kwargs):
            results = []
            evaluation_queue = queue.Queue()
            evaluation_threads = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                futures = [
                    executor.submit(process_prompt, model, prompt, index, evaluator, evaluation_queue, **kwargs)
                    for index, prompt in enumerate(prompts)
                ]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    results.append(result)
                    if evaluator:
                        evaluation_threads.append(result.get('evaluation_thread'))
            return model, results, evaluation_queue, evaluation_threads

        # Run completion tasks in parallel for each model, but sequentially for each prompt within a model
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_prompts_sequentially, model, problems, evaluator, **kwargs)
                for model in self._providers
            ]

            for future in as_completed(futures):
                model, outputs, evaluation_queue, evaluation_threads = future.result()
                model_results[model] = {
                    "outputs": outputs,
                    "total_latency": 0,
                    "total_cost": 0,
                    "evaluation": [None] * len(outputs),
                }

                for output_data in outputs:
                    model_results[model]["total_latency"] += output_data["latency"]
                    model_results[model]["total_cost"] += output_data["cost"]

                if evaluator:
                    # Wait for all evaluation threads to complete
                    for thread in evaluation_threads:
                        thread.join()

                    # Process all evaluation results
                    while not evaluation_queue.empty():
                        index, evaluation = evaluation_queue.get()
                        model_results[model]["evaluation"][index] = evaluation

        for model in model_results:
            outputs = model_results[model]["outputs"]
            model_results[model]["median_latency"] = statistics.median(
                [output["latency"] for output in outputs]
            )

            total_tokens = sum([output["tokens"] for output in outputs])
            total_latency = model_results[model]["total_latency"]
            model_results[model]["aggregated_speed"] = total_tokens / total_latency

        if evaluator:
            sorted_models = sorted(
                model_results,
                key=lambda x: model_results[x]["aggregated_speed"]
                * sum(filter(None, model_results[x]["evaluation"])),
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
                if valid_evaluations > 0:
                    acc = 100 * total_score / valid_evaluations
                    row_data.append(f"Accuracy: {acc:.2f}%")
                else:
                    row_data.append("Accuracy: N/A")

            table.add_row(row_data)

        if not html:
            return table
        else:
            return table.get_html_string()

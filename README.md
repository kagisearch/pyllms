
# PyLLMs

[![](https://dcbadge.vercel.app/api/server/aDNg6E9szy?compact=true&style=flat)](https://discord.gg/aDNg6E9szy) [![Twitter](https://img.shields.io/twitter/follow/KagiHQ?style=social)](https://twitter.com/KagiHQ) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/license/mit/) 

PyLLMs is a minimal Python library to connect to LLMs (OpenAI, Anthropic, AI21, Cohere, Aleph Alpha, HuggingfaceHub) with a built-in model performance benchmark. 

It is ideal for fast prototyping and evaluating different models thanks to:
- Connect to top LLMs in s few lines of code (currently OpenAI, Anthropic and AI21 are supported)
- Response meta includes tokens processed, cost and latency standardized across the models
- Multi-model support: Get completions from different models at the same time
- LLM benchmark: Evaluate models on quality, speed and cost

Feel free to reuse and expand. Pull requests are welcome.

## Installation

Clone this repository and install the package using pip:

```
pip3 install pyllms
```


## Usage


```
import llms

model = llms.init()
result = model.complete("what is 5+5")

print(result.text)  

```

Library will attempt to read the API keys and the default model from environment variables, which you can set like this:

```
export OPENAI_API_KEY="your_api_key_here"
export ANTHROPIC_API_KEY="your_api_key_here"
export AI21_API_KEY="your_api_key_here"

export LLMS_DEFAULT_MODEL="gpt-3.5-turbo"
```


Alternatively, you can pass initialization values to the init() method:

```
model=llms.init(openai_api_key='your_api_key_here', model='gpt-4')
```


You can also pass optional parameters to the complete method. 'temperature' and 'max_tokens' are standardized across all APIs and get converted to the corresponding API params. 

Any other parameters accepted by the original model are supported in their verbatim form.

```
result = model.complete(
    "what is the capital of country where mozzart was born",
    temperature=0.1,
    max_tokens=200
)
```

Note: By default, temperature for all models is set to 0, and max_tokens to 300.

The result meta will contain helpful information like tokens used, cost (which is automatically calculated using current pricing), and response latency:
```
>>> print(result.meta)
{'model': 'gpt-3.5-turbo', 'tokens': 15, 'tokens_prompt': 14, 'tokens_completion': 1, 'cost': 3e-05, 'latency': 0.48232388496398926}
```


## Multi-model usage

You can also initialize multiple models at once! This is very useful for testing and comparing output of different models in parallel. 

```
>>> models=llms.init(model=['gpt-3.5-turbo','claude-instant-v1'])
>>> result=models.complete('what is the capital of country where mozzart was born')
>>> print(result.text)
['The capital of the country where Mozart was born is Vienna, Austria.', 'Wolfgang Amadeus Mozart was born in Salzburg, Austria.\n\nSo the capital of the country where Mozart was born is Vienna, Austria.']

>>> print(result.meta)
[{'model': 'gpt-3.5-turbo', 'tokens': 34, 'tokens_prompt': 20, 'tokens_completion': 14, 'cost': 6.8e-05, 'latency': 0.7097790241241455}, {'model': 'claude-instant-v1', 'tokens': 54, 'tokens_prompt': 20, 'tokens_completion': 34, 'cost': 5.79e-05, 'latency': 0.7291600704193115}]
```

## Streaming support

PyLLMs supports streaming from compatible models. 'complete_stream' method will return generator object and all you have to do is iterate through it:

```
>>> model= llms.init('claude-v1')
>>> result = model.complete_stream("write an essay on civil war")
>>> for chunk in result:
...        if chunk is not None:
...          print(chunk, end='')   
... 

Here is a paragraph about civil rights:


Civil rights are the basic rights and freedoms that all citizens should have in a society. They include fundamental rights like the right to vote, the right to free speech, the right to practice the religion of one's choice, the right to equal treatment under the law, and the right to live free from discrimination. The civil rights movement in the United States fought to secure these rights for African Americans and other minorities in the face of institutionalized racism and discrimination. Leaders like Martin Luther King Jr. helped pass laws like the Civil Rights Act of 1964 and the Voting Rights Act of 1965 which outlawed discrimination and dismantled barriers to voting. The struggle for civil rights continues today as more work is still needed to promote racial equality and protect the rights of all citizens.

```

Current limitations:
- When streaming, 'meta' is not available
- Multi-models are not supported for streaming


Tip: if you are testing this in python3 CLI, run it with -u parameter to disable buffering:

```
python3 -u
```

## Other methods

You can count tokens using the model's tokenizer:

```
count=model.count_tokens('the quick brown fox jumped over the lazy dog')
```

## Model Benchmarks

Models are appearing like mushrooms after rain and everyone is interested in three things:

1) Quality
2) Speed
3) Cost

PyLLMs icludes an automated benchmark system. The quality of models is evaluated using a powerful model (for example gpt-4) on a range of predefined questions, or you can supply your own.


```
models=llms.init(model=['gpt-3.5-turbo', 'claude-instant-v1', 'j2-jumbo-instruct'])
gpt4=llms.init('gpt-4') # optional, evaluator can be ommited and in that case only speed and cost will be evaluated
models.benchmark(evaluator=gpt4)
```

```
+--------------------------------+--------------------+---------------------+----------------------+-------------------------+-----------------+
|             Model              |       Tokens       |       Cost ($)      |     Latency (s)      |    Speed (tokens/sec)   |    Evaluation   |
+--------------------------------+--------------------+---------------------+----------------------+-------------------------+-----------------+
| OpenAIProvider (gpt-3.5-turbo) |         37         |       0.00007       |         1.47         |          25.19          |        1        |
| OpenAIProvider (gpt-3.5-turbo) |         93         |       0.00019       |         4.13         |          22.53          |        0        |
| OpenAIProvider (gpt-3.5-turbo) |        360         |       0.00072       |        18.42         |          19.54          |        5        |
| OpenAIProvider (gpt-3.5-turbo) |        143         |       0.00029       |         6.76         |          21.15          |        5        |
| OpenAIProvider (gpt-3.5-turbo) |        112         |       0.00022       |         3.87         |          28.95          |        4        |
| OpenAIProvider (gpt-3.5-turbo) |         47         |       0.00009       |         1.57         |          29.86          |        5        |
| OpenAIProvider (gpt-3.5-turbo) |         78         |       0.00016       |         1.52         |          51.19          |        5        |
| OpenAIProvider (gpt-3.5-turbo) |        254         |       0.00051       |         1.08         |          235.22         |        5        |
| OpenAIProvider (gpt-3.5-turbo) |        284         |       0.00057       |        11.39         |          24.94          |        5        |
| OpenAIProvider (gpt-3.5-turbo) |        358         |       0.00072       |        15.77         |          22.71          |        5        |
| OpenAIProvider (gpt-3.5-turbo) |        485         |       0.00097       |        23.84         |          20.34          |        5        |
| OpenAIProvider (gpt-3.5-turbo) |        222         |       0.00044       |         3.87         |          57.37          |        5        |
| OpenAIProvider (gpt-3.5-turbo) | Total Tokens: 2473 | Total Cost: 0.00495 | Median Latency: 4.00 | Aggregated speed: 26.40 | Total Score: 50 |
+--------------------------------+--------------------+---------------------+----------------------+-------------------------+-----------------+
+---------------------------------------+--------------------+---------------------+----------------------+--------------------------+-----------------+
|                 Model                 |       Tokens       |       Cost ($)      |     Latency (s)      |    Speed (tokens/sec)    |    Evaluation   |
+---------------------------------------+--------------------+---------------------+----------------------+--------------------------+-----------------+
| AnthropicProvider (claude-instant-v1) |         33         |       0.00010       |         0.85         |          38.63           |        1        |
| AnthropicProvider (claude-instant-v1) |        152         |       0.00072       |         1.69         |          89.97           |        5        |
| AnthropicProvider (claude-instant-v1) |         59         |       0.00024       |         0.70         |          84.55           |        5        |
| AnthropicProvider (claude-instant-v1) |        112         |       0.00054       |         1.31         |          85.18           |        5        |
| AnthropicProvider (claude-instant-v1) |        191         |       0.00082       |         1.54         |          124.30          |        0        |
| AnthropicProvider (claude-instant-v1) |         65         |       0.00024       |         0.68         |          95.35           |        5        |
| AnthropicProvider (claude-instant-v1) |        190         |       0.00082       |         1.54         |          123.19          |        5        |
| AnthropicProvider (claude-instant-v1) |        276         |       0.00053       |         0.69         |          398.39          |        5        |
| AnthropicProvider (claude-instant-v1) |        220         |       0.00085       |         1.52         |          144.87          |        5        |
| AnthropicProvider (claude-instant-v1) |        189         |       0.00072       |         1.21         |          156.10          |        5        |
| AnthropicProvider (claude-instant-v1) |        326         |       0.00145       |         2.65         |          122.87          |        0        |
| AnthropicProvider (claude-instant-v1) |        281         |       0.00089       |         1.37         |          204.61          |        5        |
| AnthropicProvider (claude-instant-v1) | Total Tokens: 2094 | Total Cost: 0.00791 | Median Latency: 1.34 | Aggregated speed: 132.82 | Total Score: 46 |
+---------------------------------------+--------------------+---------------------+----------------------+--------------------------+-----------------+

+-----------------------------------------+--------------------+---------------------+----------------------+-------------------------+-----------------+
|                  Model                  |       Tokens       |       Cost ($)      |     Latency (s)      |    Speed (tokens/sec)   |    Evaluation   |
+-----------------------------------------+--------------------+---------------------+----------------------+-------------------------+-----------------+
| CohereProvider (command-xlarge-nightly) |         27         |       0.00068       |         0.73         |          37.23          |        1        |
| CohereProvider (command-xlarge-nightly) |         49         |       0.00122       |         1.04         |          47.03          |        5        |
| CohereProvider (command-xlarge-nightly) |         31         |       0.00077       |         0.67         |          46.10          |        0        |
| CohereProvider (command-xlarge-nightly) |         30         |       0.00075       |         0.73         |          41.35          |        0        |
| CohereProvider (command-xlarge-nightly) |        128         |       0.00320       |         2.89         |          44.27          |        0        |
| CohereProvider (command-xlarge-nightly) |         38         |       0.00095       |         0.70         |          54.29          |        4        |
| CohereProvider (command-xlarge-nightly) |         57         |       0.00143       |         0.51         |          111.13         |        5        |
| CohereProvider (command-xlarge-nightly) |        269         |       0.00673       |         0.98         |          274.23         |        3        |
| CohereProvider (command-xlarge-nightly) |        230         |       0.00575       |         4.55         |          50.54          |        0        |
| CohereProvider (command-xlarge-nightly) |        170         |       0.00425       |         2.45         |          69.41          |        0        |
| CohereProvider (command-xlarge-nightly) |        1502        |       0.03755       |        30.80         |          48.77          |        0        |
| CohereProvider (command-xlarge-nightly) |        218         |       0.00545       |         2.01         |          108.49         |        4        |
| CohereProvider (command-xlarge-nightly) | Total Tokens: 2749 | Total Cost: 0.06872 | Median Latency: 1.01 | Aggregated speed: 57.20 | Total Score: 22 |
+-----------------------------------------+--------------------+---------------------+----------------------+-------------------------+-----------------+
```

To evaluate models on your own prompts, simply pass a list of questions. The evaluator will automatically evaluate the responses:

```
models.benchmark(prompts=["what is the capital of finland", "who won superbowl in the year justin bieber was born"], evaluator=gpt4)
```

## Supported Models

To get a list of supported models, call list(). Models will be shown in the order of least expensive to most expensive.

```
>>> model=llms.init()

>>> model.list()

>>> model.list("gpt') # lists only models with 'gpt' in name/provider name

| Provider            | Name                   | Prompt Cost | Completion Cost | Token Limit |
|---------------------|------------------------|-------------|-----------------|-------------|
| AI21Provider        | j2-grande-instruct     |        10.0 |            10.0 |        8192 |
| AI21Provider        | j2-jumbo-instruct      |        15.0 |            15.0 |        8192 |
| AlephAlphaProvider  | luminous-base          |         6.6 |             7.6 |        2048 |
| AlephAlphaProvider  | luminous-extended      |         9.9 |            10.9 |        2048 |
| AlephAlphaProvider  | luminous-supreme       |        38.5 |            42.5 |        2048 |
| AlephAlphaProvider  | luminous-supreme-control |      48.5 |            53.6 |        2048 |
| AnthropicProvider   | claude-instant-v1      |        1.63 |            5.51 |        9000 |
| AnthropicProvider   | claude-v1              |       11.02 |           32.68 |        9000 |
| CohereProvider      | command-xlarge-beta    |          25 |              25 |        8192 |
| CohereProvider      | command-xlarge-nightly |          25 |              25 |        8192 |
| OpenAIProvider      | gpt-3.5-turbo          |         2.0 |             2.0 |        4000 |
| OpenAIProvider      | gpt-4                  |        30.0 |            60.0 |        8000 |

```

Useful links:\
[OpenAI documentation](https://platform.openai.com/docs/api-reference/completions)\
[Anthropic documentation](https://console.anthropic.com/docs/api/reference#-v1-complete)\
[AI21 documentation](https://docs.ai21.com/reference/j2-instruct-ref)
[Cohere documentation](https://cohere-sdk.readthedocs.io/en/latest/cohere.html#api)
[Aleph Alpha documentation](https://aleph-alpha-client.readthedocs.io/en/latest/aleph_alpha_client.html#aleph_alpha_client.CompletionRequest)

## License

This project is licensed under the MIT License.


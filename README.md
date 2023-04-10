# LLMS

A lightweight Python that strives to enable dead-simple interaction with popular language models from OpenAI, Anthropic, AI21 and others.

## Installation

Clone this repository and install the package using pip:

```
pip3 install llms
```


## Usage


```
import llms
model = llms.init()
result = model.complete("what is 5+5")
print(result.text)  

```

llms will read the API key from environment variables, which you can set like this:

```
export OPENAI_API_KEY="your_api_key_here"
export ANTHROPIC_API_KEY="your_api_key_here"
export AI21_API_KEY="your_api_key_here"
export LLMS_DEFAULT_MODEL="gpt-3.5-turbo"
```


Alternatively you can pass initialization values to the init() method:

```
model=llms.init(openai_api_key='your_api_key_here', model='gpt-4')
```


You can also pass optional parameters to the complete method. Any parameters accepted by the original model are supported automatically, in verbatim form:

```
result = model.complete(
    "what is the capital of country where mozzart was born",
    temperature=0.8,
)
```


Result will also contain useful information like tokens used, cost (which is automaticall calculated using current pricing) and result latency:

```
>>> print(result.meta)
{'model': 'gpt-3.5-turbo', 'tokens': 15, 'tokens_prompt': 14, 'tokens_completion': 1, 'cost': 3e-05, 'latency': 0.48232388496398926}
```








## Multi-model usage

You can initialize multiple models at once. This is useful for testing and comparing output. All models will be queried in parallel to save time. 

```
>>> models=llms.init(model=['gpt-3.5-turbo','claude-instant-v1'])
>>> result=models.complete('what is the capital of country where mozzart was born')
>>> print(result.text)
['The capital of the country where Mozart was born is Vienna, Austria.', 'Wolfgang Amadeus Mozart was born in Salzburg, Austria.\n\nSo the capital of the country where Mozart was born is Vienna, Austria.']

>>> print(result.meta)
[{'model': 'gpt-3.5-turbo', 'tokens': 34, 'tokens_prompt': 20, 'tokens_completion': 14, 'cost': 6.8e-05, 'latency': 0.7097790241241455}, {'model': 'claude-instant-v1', 'tokens': 54, 'tokens_prompt': 20, 'tokens_completion': 34, 'cost': 5.79e-05, 'latency': 0.7291600704193115}]
```

## Benchmarks

```
models=llms.init(model=['gpt-3.5-turbo', 'claude-instant-v1'])
gpt4=llms.init('gpt-4')
models.benchmark(evaluator=gpt4)
```

+---------------------------------------+--------------------+---------------------+----------------------+---------------------------+------------------+
|                 Model                 |       Tokens       |       Cost ($)      |     Latency (s)      |     Speed (tokens/sec)    |    Evaluation    |
+---------------------------------------+--------------------+---------------------+----------------------+---------------------------+------------------+
| AnthropicProvider (claude-instant-v1) |         33         |       0.00003       |         0.86         |           38.49           |        8         |
| AnthropicProvider (claude-instant-v1) |        152         |       0.00019       |         1.81         |           83.99           |        10        |
| AnthropicProvider (claude-instant-v1) |        248         |       0.00031       |         2.15         |           115.37          |        7         |
| AnthropicProvider (claude-instant-v1) |        209         |       0.00021       |         1.69         |           123.94          |        10        |
| AnthropicProvider (claude-instant-v1) |         59         |       0.00006       |         0.68         |           86.91           |        10        |
| AnthropicProvider (claude-instant-v1) |        140         |       0.00018       |         1.44         |           96.93           |        10        |
| AnthropicProvider (claude-instant-v1) |        245         |       0.00031       |         2.51         |           97.73           |        9         |
| AnthropicProvider (claude-instant-v1) |        250         |       0.00031       |         2.17         |           115.10          |        9         |
| AnthropicProvider (claude-instant-v1) |        248         |       0.00031       |         2.15         |           115.24          |        9         |
| AnthropicProvider (claude-instant-v1) |        323         |       0.00034       |         1.86         |           173.71          |        8         |
| AnthropicProvider (claude-instant-v1) |        168         |       0.00015       |         0.91         |           184.79          |        10        |
| AnthropicProvider (claude-instant-v1) | Total Tokens: 2075 | Total Cost: 0.00240 | Median Latency: 1.81 | Aggregrated speed: 113.85 | Total Score: 100 |
|     OpenAIProvider (gpt-3.5-turbo)    |         37         |       0.00007       |         1.22         |           30.39           |        8         |
|     OpenAIProvider (gpt-3.5-turbo)    |         93         |       0.00019       |         2.32         |           40.13           |        10        |
|     OpenAIProvider (gpt-3.5-turbo)    |        451         |       0.00090       |        14.70         |           30.69           |        10        |
|     OpenAIProvider (gpt-3.5-turbo)    |        204         |       0.00041       |         4.50         |           45.38           |        10        |
|     OpenAIProvider (gpt-3.5-turbo)    |        320         |       0.00064       |        10.51         |           30.44           |        10        |
|     OpenAIProvider (gpt-3.5-turbo)    |        109         |       0.00022       |         3.27         |           33.31           |        10        |
|     OpenAIProvider (gpt-3.5-turbo)    |        314         |       0.00063       |         9.96         |           31.52           |        10        |
|     OpenAIProvider (gpt-3.5-turbo)    |        294         |       0.00059       |         8.84         |           33.26           |        10        |
|     OpenAIProvider (gpt-3.5-turbo)    |        274         |       0.00055       |         8.35         |           32.82           |        10        |
|     OpenAIProvider (gpt-3.5-turbo)    |        501         |       0.00100       |        13.45         |           37.26           |        10        |
|     OpenAIProvider (gpt-3.5-turbo)    |        170         |       0.00034       |         2.72         |           62.52           |        10        |
|     OpenAIProvider (gpt-3.5-turbo)    | Total Tokens: 2767 | Total Cost: 0.00553 | Median Latency: 8.35 |  Aggregrated speed: 34.66 | Total Score: 108 |
+---------------------------------------+--------------------+---------------------+----------------------+---------------------------+------------------+

## Supported Models

To get a list of supported models, call list(). Models will be shown in the order of least expensive to most expensive.

```
>>> model=llms.init()
>>> model.list()

[{'provider': 'AnthropicProvider', 'name': 'claude-instant-v1', 'cost': {'prompt': 0.43, 'completion': 1.45}}, {'provider': 'OpenAIProvider', 'name': 'gpt-3.5-turbo', 'cost': {'prompt': 2.0, 'completion': 2.0}}, {'provider': 'AI21Provider', 'name': 'j2-large', 'cost': {'prompt': 3.0, 'completion': 3.0}}, {'provider': 'AnthropicProvider', 'name': 'claude-v1', 'cost': {'prompt': 2.9, 'completion': 8.6}}, {'provider': 'AI21Provider', 'name': 'j2-grande', 'cost': {'prompt': 10.0, 'completion': 10.0}}, {'provider': 'AI21Provider', 'name': 'j2-grande-instruct', 'cost': {'prompt': 10.0, 'completion': 10.0}}, {'provider': 'AI21Provider', 'name': 'j2-jumbo', 'cost': {'prompt': 15.0, 'completion': 15.0}}, {'provider': 'AI21Provider', 'name': 'j2-jumbo-instruct', 'cost': {'prompt': 15.0, 'completion': 15.0}}, {'provider': 'OpenAIProvider', 'name': 'gpt-4', 'cost': {'prompt': 30.0, 'completion': 60.0}}]
```

Useful links:\
[OpenAI documentation](https://platform.openai.com/docs/api-reference/completions)\
[Anthropic documentation](https://console.anthropic.com/docs/api/reference#-v1-complete)\
[AI21 documentation](https://docs.ai21.com/reference/j2-instruct-ref)


## License

This project is licensed under the MIT License.


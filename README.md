# PyLLMs

A lightweight Python that strives to enable dead-simple interaction with popular language models from OpenAI, Anthropic, AI21 and others.

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

llms will read the API key from environment variables, which you can set like this:

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


You can also pass optional parameters to the complete method. Any parameters accepted by the original model are supported automatically, in the verbatim form:

```
result = model.complete(
    "what is the capital of country where mozzart was born",
    temperature=0.1,
)
```

By default, temperature for all models is set to 0.

The result will also contain helpful information like tokens used, cost (which is automatically calculated using current pricing), and result latency:
```
>>> print(result.meta)
{'model': 'gpt-3.5-turbo', 'tokens': 15, 'tokens_prompt': 14, 'tokens_completion': 1, 'cost': 3e-05, 'latency': 0.48232388496398926}
```








## Multi-model usage

You can initialize multiple models at once, which is very useful for testing and comparing output of different models. All models will run in parallel to save time. 

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

```
+---------------------------------------+--------------------+---------------------+-----------------------+---------------------------+-----------------+
|                 Model                 |       Tokens       |       Cost ($)      |      Latency (s)      |     Speed (tokens/sec)    |    Evaluation   |
+---------------------------------------+--------------------+---------------------+-----------------------+---------------------------+-----------------+
| AnthropicProvider (claude-instant-v1) |         33         |       0.00003       |          0.40         |           82.19           |        3        |
| AnthropicProvider (claude-instant-v1) |        152         |       0.00019       |          1.65         |           91.89           |        10       |
| AnthropicProvider (claude-instant-v1) |        248         |       0.00031       |          2.18         |           113.74          |        9        |
| AnthropicProvider (claude-instant-v1) |        209         |       0.00021       |          1.86         |           112.11          |        10       |
| AnthropicProvider (claude-instant-v1) |         59         |       0.00006       |          0.87         |           68.18           |        10       |
| AnthropicProvider (claude-instant-v1) |        140         |       0.00018       |          1.46         |           96.10           |        10       |
| AnthropicProvider (claude-instant-v1) |        245         |       0.00031       |          2.45         |           100.16          |        9        |
| AnthropicProvider (claude-instant-v1) |        250         |       0.00031       |          2.29         |           109.35          |        9        |
| AnthropicProvider (claude-instant-v1) |        248         |       0.00031       |          2.17         |           114.50          |        9        |
| AnthropicProvider (claude-instant-v1) |        323         |       0.00034       |          1.95         |           165.57          |        8        |
| AnthropicProvider (claude-instant-v1) |        172         |       0.00014       |          0.97         |           177.63          |        10       |
| AnthropicProvider (claude-instant-v1) | Total Tokens: 2079 | Total Cost: 0.00240 |  Median Latency: 1.86 | Aggregrated speed: 113.98 | Total Score: 97 |
|     OpenAIProvider (gpt-3.5-turbo)    |         37         |       0.00007       |          1.20         |           30.86           |        7        |
|     OpenAIProvider (gpt-3.5-turbo)    |         93         |       0.00019       |          2.89         |           32.19           |        1        |
|     OpenAIProvider (gpt-3.5-turbo)    |        451         |       0.00090       |         18.10         |           24.91           |        10       |
|     OpenAIProvider (gpt-3.5-turbo)    |        204         |       0.00041       |          5.60         |           36.45           |        10       |
|     OpenAIProvider (gpt-3.5-turbo)    |        366         |       0.00073       |         16.51         |           22.17           |        10       |
|     OpenAIProvider (gpt-3.5-turbo)    |        109         |       0.00022       |          3.69         |           29.51           |        10       |
|     OpenAIProvider (gpt-3.5-turbo)    |        316         |       0.00063       |         11.96         |           26.43           |        10       |
|     OpenAIProvider (gpt-3.5-turbo)    |        294         |       0.00059       |         10.47         |           28.09           |        10       |
|     OpenAIProvider (gpt-3.5-turbo)    |        275         |       0.00055       |         10.02         |           27.45           |        10       |
|     OpenAIProvider (gpt-3.5-turbo)    |        501         |       0.00100       |         16.28         |           30.77           |        10       |
|     OpenAIProvider (gpt-3.5-turbo)    |        180         |       0.00036       |          3.23         |           55.79           |        10       |
|     OpenAIProvider (gpt-3.5-turbo)    | Total Tokens: 2826 | Total Cost: 0.00565 | Median Latency: 10.02 |  Aggregrated speed: 28.28 | Total Score: 98 |
|    AI21Provider (j2-jumbo-instruct)   |         27         |       0.00040       |          0.95         |           28.31           |        3        |
|    AI21Provider (j2-jumbo-instruct)   |        114         |       0.00171       |          3.10         |           36.81           |        1        |
|    AI21Provider (j2-jumbo-instruct)   |        195         |       0.00293       |          5.62         |           34.73           |        10       |
|    AI21Provider (j2-jumbo-instruct)   |        117         |       0.00176       |          2.08         |           56.12           |        10       |
|    AI21Provider (j2-jumbo-instruct)   |        216         |       0.00324       |          6.12         |           35.27           |        7        |
|    AI21Provider (j2-jumbo-instruct)   |         67         |       0.00101       |          2.01         |           33.39           |        10       |
|    AI21Provider (j2-jumbo-instruct)   |        229         |       0.00344       |          6.14         |           37.27           |        10       |
|    AI21Provider (j2-jumbo-instruct)   |        225         |       0.00337       |          6.21         |           36.26           |        5        |
|    AI21Provider (j2-jumbo-instruct)   |        218         |       0.00327       |          5.90         |           36.95           |        1        |
|    AI21Provider (j2-jumbo-instruct)   |        281         |       0.00421       |          6.25         |           44.97           |        1        |
|    AI21Provider (j2-jumbo-instruct)   |        149         |       0.00224       |          1.56         |           95.81           |        10       |
|    AI21Provider (j2-jumbo-instruct)   | Total Tokens: 1838 | Total Cost: 0.02757 |  Median Latency: 5.62 |  Aggregrated speed: 40.01 | Total Score: 68 |
+---------------------------------------+--------------------+---------------------+-----------------------+---------------------------+-----------------+
```

In addition, you can evaluate models on your own prompts:
```
models.benchmark(prompts=["what is the capital of finland", "who won superbowl in the year justin bieber was born"],evaluator=gpt4)
```

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


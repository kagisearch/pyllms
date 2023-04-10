# LLMS

A lightweight Python package for easy interaction with popular language models from OpenAI, Anthropic, AI21 and others.

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

useful information like tokens, cost and result latency is stored in 'meta' property of the result.

```
>>> print(result.meta)
{'model': 'gpt-3.5-turbo', 'tokens': 15, 'tokens_prompt': 14, 'tokens_completion': 1, 'cost': 3e-05, 'latency': 0.48232388496398926}
```


llms will read the API key from environment variables, which you can set like this:

```
export OPENAI_API_KEY="your_api_key_here"
export ANTHROPIC_API_KEY="your_api_key_here"
export AI21_API_KEY="your_api_key_here"
```

Alternatively you can pass it to the init() method:

```
model = llms.init(openai_api_key='your_api_key_here')
```

Other values are anthropic_api_key and ai21_api_key.

To select the model, use model param:

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

Model documentation:
[OpenAI documentation](https://platform.openai.com/docs/api-reference/completions)
[Anthropic documentation](https://console.anthropic.com/docs/api/reference#-v1-complete)
[AI21 documentation](https://docs.ai21.com/reference/j2-instruct-ref)


## Supported Models

To get a list of supported models, call list(). Models will be shown in the order of least expensive to most expensive.

```
>>> model=llms.init()
>>> model.list()

[{'provider': 'AnthropicProvider', 'name': 'claude-instant-v1', 'cost': {'prompt': 0.43, 'completion': 1.45}}, {'provider': 'OpenAIProvider', 'name': 'gpt-3.5-turbo', 'cost': {'prompt': 2.0, 'completion': 2.0}}, {'provider': 'AI21Provider', 'name': 'j2-large', 'cost': {'prompt': 3.0, 'completion': 3.0}}, {'provider': 'AnthropicProvider', 'name': 'claude-v1', 'cost': {'prompt': 2.9, 'completion': 8.6}}, {'provider': 'AI21Provider', 'name': 'j2-grande', 'cost': {'prompt': 10.0, 'completion': 10.0}}, {'provider': 'AI21Provider', 'name': 'j2-grande-instruct', 'cost': {'prompt': 10.0, 'completion': 10.0}}, {'provider': 'AI21Provider', 'name': 'j2-jumbo', 'cost': {'prompt': 15.0, 'completion': 15.0}}, {'provider': 'AI21Provider', 'name': 'j2-jumbo-instruct', 'cost': {'prompt': 15.0, 'completion': 15.0}}, {'provider': 'OpenAIProvider', 'name': 'gpt-4', 'cost': {'prompt': 30.0, 'completion': 60.0}}]
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

## License

This project is licensed under the MIT License.


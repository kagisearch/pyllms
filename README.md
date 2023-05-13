
# PyLLMs

[![](https://dcbadge.vercel.app/api/server/aDNg6E9szy?compact=true&style=flat)](https://discord.gg/aDNg6E9szy) [![Twitter](https://img.shields.io/twitter/follow/KagiHQ?style=social)](https://twitter.com/KagiHQ) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/license/mit/) 

PyLLMs is a minimal Python library to connect to LLMs (OpenAI, Anthropic, AI21, Cohere, Aleph Alpha, HuggingfaceHub) with a built-in model performance benchmark. 

It is ideal for fast prototyping and evaluating different models thanks to:
- Connect to top LLMs in a few lines of code
- Response meta includes tokens processed, cost and latency standardized across the models
- Multi-model support: Get completions from different models at the same time
- LLM benchmark: Evaluate models on quality, speed and cost

Feel free to reuse and expand. Pull requests are welcome.

## Installation

Clone this repository and install the package using pip:

```
pip install pyllms
```


## Usage


```
import llms

model = llms.init()
result = model.complete("what is 5+5")

print(result.text)  

```

Library will attempt to read the API keys and the default model from environment variables, which you can set like this (for the provider you are using):

```
export OPENAI_API_KEY="your_api_key_here"
export ANTHROPIC_API_KEY="your_api_key_here"
export AI21_API_KEY="your_api_key_here"
export COHERE_API_KEY="your_api_key_here"
export ALEPHALPHA_API_KEY="your_api_key_here"
export HUGGINFACEHUB_API_KEY="your_api_key_here"

export LLMS_DEFAULT_MODEL="gpt-3.5-turbo"
```


Alternatively, you can pass initialization values to the init() method:

```
model=llms.init(openai_api_key='your_api_key_here', model='gpt-4')
```

For using Google LLMs through Vertex AI API, see "Using Google AI models" below.

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
{
  'model': 'gpt-3.5-turbo', 
  'tokens': 34, 
  'tokens_prompt': 20, 
  'tokens_completion': 14, 
  'cost': '0.00007', 
  'latency': 1.4
}
```


## Multi-model usage

You can also initialize multiple models at once! This is very useful for testing and comparing output of different models in parallel. 

```
>>> models=llms.init(model=['gpt-3.5-turbo','claude-instant-v1'])
>>> result=models.complete('what is the capital of country where mozzart was born')
>>> print(result.text)
[
 'The capital of the country where Mozart was born is Vienna, Austria.', 
 'Wolfgang Amadeus Mozart was born in Salzburg, Austria.\n\nSo the capital of the country where Mozart was born is Vienna, Austria.'
]

>>> print(result.meta)
[
 {'model': 'gpt-3.5-turbo', 'tokens': 34, 'tokens_prompt': 20, 'tokens_completion': 14, 'cost': 6.8e-05, 'latency': 0.7097790241241455}, 
 {'model': 'claude-instant-v1', 'tokens': 54, 'tokens_prompt': 20, 'tokens_completion': 34, 'cost': 5.79e-05, 'latency': 0.7291600704193115}
]
```

## Async support
Async completion is supported for compatible models. It is not supported in multi-models mode yet.
```
result = await model.acomplete("what is the capital of country where mozzart was born")
```

## Streaming support

PyLLMs supports streaming from compatible models. 'complete_stream' method will return generator object and all you have to do is iterate through it:

```
model= llms.init('claude-v1')
result = model.complete_stream("write an essay on civil war")
for chunk in result:
   if chunk is not None:
      print(chunk, end='')   
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
models=llms.init(model=['gpt-3.5-turbo', 'claude-instant-v1', 'command-xlarge-nightly'])

gpt4=llms.init('gpt-4') # optional, evaluator can be ommited and in that case only speed and cost will be evaluated

models.benchmark(evaluator=gpt4)
```

```
+-----------------------------------------+--------------------+---------------------+----------------------+--------------------------+-----------------+
|                  Model                  |       Tokens       |       Cost ($)      |     Latency (s)      |    Speed (tokens/sec)    |    Evaluation   |
+-----------------------------------------+--------------------+---------------------+----------------------+--------------------------+-----------------+
| AnthropicProvider (claude-instant-v1.1) |         49         |       0.00019       |         0.98         |          50.00           |        5        |
| AnthropicProvider (claude-instant-v1.1) |        211         |       0.00104       |         2.50         |          84.40           |        0        |
| AnthropicProvider (claude-instant-v1.1) |        204         |       0.00103       |         2.06         |          99.03           |        5        |
| AnthropicProvider (claude-instant-v1.1) |        139         |       0.00068       |         1.66         |          83.73           |        5        |
| AnthropicProvider (claude-instant-v1.1) |        138         |       0.00053       |         3.85         |          35.84           |        4        |
| AnthropicProvider (claude-instant-v1.1) |        156         |       0.00074       |         1.75         |          89.14           |        5        |
| AnthropicProvider (claude-instant-v1.1) |        284         |       0.00134       |         2.94         |          96.60           |        3        |
| AnthropicProvider (claude-instant-v1.1) |        266         |       0.00048       |         0.57         |          466.67          |        5        |
| AnthropicProvider (claude-instant-v1.1) |        247         |       0.00100       |         1.73         |          142.77          |        0        |
| AnthropicProvider (claude-instant-v1.1) |        211         |       0.00084       |         1.53         |          137.91          |        5        |
| AnthropicProvider (claude-instant-v1.1) |        180         |       0.00064       |         1.37         |          131.39          |        0        |
| AnthropicProvider (claude-instant-v1.1) |        300         |       0.00099       |         1.43         |          209.79          |        5        |
| AnthropicProvider (claude-instant-v1.1) |         63         |       0.00029       |         0.91         |          69.23           |        5        |
| AnthropicProvider (claude-instant-v1.1) | Total Tokens: 2448 | Total Cost: 0.00979 | Median Latency: 1.66 | Aggregated speed: 105.15 | Total Score: 47 |
+-----------------------------------------+--------------------+---------------------+----------------------+--------------------------+-----------------+
+--------------------------------+--------------------+---------------------+----------------------+-------------------------+-----------------+
|             Model              |       Tokens       |       Cost ($)      |     Latency (s)      |    Speed (tokens/sec)   |    Evaluation   |
+--------------------------------+--------------------+---------------------+----------------------+-------------------------+-----------------+
| OpenAIProvider (gpt-3.5-turbo) |         37         |       0.00007       |         2.08         |          17.79          |        1        |
| OpenAIProvider (gpt-3.5-turbo) |         93         |       0.00019       |         5.69         |          16.34          |        0        |
| OpenAIProvider (gpt-3.5-turbo) |        360         |       0.00072       |        26.75         |          13.46          |        5        |
| OpenAIProvider (gpt-3.5-turbo) |        143         |       0.00029       |        10.39         |          13.76          |        5        |
| OpenAIProvider (gpt-3.5-turbo) |        112         |       0.00022       |         4.82         |          23.24          |        3        |
| OpenAIProvider (gpt-3.5-turbo) |         47         |       0.00009       |         1.78         |          26.40          |        5        |
| OpenAIProvider (gpt-3.5-turbo) |         78         |       0.00016       |         1.87         |          41.71          |        5        |
| OpenAIProvider (gpt-3.5-turbo) |        254         |       0.00051       |         1.17         |          217.09         |        5        |
| OpenAIProvider (gpt-3.5-turbo) |        284         |       0.00057       |        16.16         |          17.57          |        5        |
| OpenAIProvider (gpt-3.5-turbo) |        358         |       0.00072       |        25.45         |          14.07          |        5        |
| OpenAIProvider (gpt-3.5-turbo) |        485         |       0.00097       |        32.18         |          15.07          |        5        |
| OpenAIProvider (gpt-3.5-turbo) |        222         |       0.00044       |         6.13         |          36.22          |        5        |
| OpenAIProvider (gpt-3.5-turbo) |         66         |       0.00013       |         7.73         |           8.54          |        5        |
| OpenAIProvider (gpt-3.5-turbo) | Total Tokens: 2539 | Total Cost: 0.00508 | Median Latency: 6.13 | Aggregated speed: 17.86 | Total Score: 54 |
+--------------------------------+--------------------+---------------------+----------------------+-------------------------+-----------------+
+-----------------------------+----------------------+---------------------+----------------------+-------------------------+-----------------+
|            Model            |        Tokens        |       Cost ($)      |     Latency (s)      |    Speed (tokens/sec)   |    Evaluation   |
+-----------------------------+----------------------+---------------------+----------------------+-------------------------+-----------------+
| GoogleProvider (chat-bison) |        36.75         |       0.00007       |         1.86         |          19.76          |        5        |
| GoogleProvider (chat-bison) |        229.5         |       0.00046       |         5.65         |          40.62          |        0        |
| GoogleProvider (chat-bison) |        131.5         |       0.00026       |         4.84         |          27.17          |        5        |
| GoogleProvider (chat-bison) |        34.75         |       0.00007       |         2.22         |          15.65          |        4        |
| GoogleProvider (chat-bison) |        407.25        |       0.00081       |        12.62         |          32.27          |        0        |
| GoogleProvider (chat-bison) |         67.0         |       0.00013       |         9.49         |           7.06          |        5        |
| GoogleProvider (chat-bison) |        103.25        |       0.00021       |         2.60         |          39.71          |        4        |
| GoogleProvider (chat-bison) |        285.0         |       0.00057       |         1.55         |          183.87         |        5        |
| GoogleProvider (chat-bison) |        282.75        |       0.00057       |         7.44         |          38.00          |        5        |
| GoogleProvider (chat-bison) |        275.5         |       0.00055       |         7.39         |          37.28          |        0        |
| GoogleProvider (chat-bison) |        283.75        |       0.00057       |         9.22         |          30.78          |        0        |
| GoogleProvider (chat-bison) |        333.75        |       0.00067       |         8.21         |          40.65          |        5        |
| GoogleProvider (chat-bison) |        50.25         |       0.00010       |         2.22         |          22.64          |        5        |
| GoogleProvider (chat-bison) | Total Tokens: 2521.0 | Total Cost: 0.00504 | Median Latency: 5.65 | Aggregated speed: 33.47 | Total Score: 43 |
+-----------------------------+----------------------+---------------------+----------------------+-------------------------+-----------------+
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
```

Here is a pretty table of supported models (in alphabetical order).
```

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
| CohereProvider      | command-xlarge-beta    |        25.0 |            25.0 |        8192 |
| CohereProvider      | command-xlarge-nightly |        25.0 |            25.0 |        8192 |
| HuggingfaceHub      | hf_pythia              |         0.0 |             0.0 |        2048 |
| OpenAIProvider      | gpt-3.5-turbo          |         2.0 |             2.0 |        4000 |
| OpenAIProvider      | gpt-4                  |        30.0 |            60.0 |        8000 |

```

Useful links:\
[OpenAI documentation](https://platform.openai.com/docs/api-reference/completions)\
[Anthropic documentation](https://console.anthropic.com/docs/api/reference#-v1-complete)\
[AI21 documentation](https://docs.ai21.com/reference/j2-instruct-ref)\
[Cohere documentation](https://cohere-sdk.readthedocs.io/en/latest/cohere.html#api)\
[Aleph Alpha documentation](https://aleph-alpha-client.readthedocs.io/en/latest/aleph_alpha_client.html#aleph_alpha_client.CompletionRequest)
[Google AI documentation](https://cloud.google.com/vertex-ai/docs/generative-ai/chat/test-chat-prompts)


## Using Google LLM models

0. (Set up a GCP account and create a project)
1. Enable Vertex AI APIs in your GCP project - https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com
1. Install gcloud CLI tool - https://cloud.google.com/sdk/docs/install
2. Set up Application Default Credentials - https://cloud.google.com/docs/authentication/provide-credentials-adc#how-to
3. Now you should be able to init Google LLM 

```
model = llms.init('chat-bison')
result = model.complete("hello!")
```

## License

This project is licensed under the MIT License.


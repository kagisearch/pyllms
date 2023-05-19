
# PyLLMs

[![](https://dcbadge.vercel.app/api/server/aDNg6E9szy?compact=true&style=flat)](https://discord.gg/aDNg6E9szy) [![Twitter](https://img.shields.io/twitter/follow/KagiHQ?style=social)](https://twitter.com/KagiHQ) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/license/mit/) 

PyLLMs is a minimal Python library to connect to LLMs (OpenAI, Anthropic, Google, AI21, Cohere, Aleph Alpha, HuggingfaceHub) with a built-in model performance benchmark. 

It is ideal for fast prototyping and evaluating different models thanks to:
- Connect to top LLMs in a few lines of code
- Response meta includes tokens processed, cost and latency standardized across the models
- Multi-model support: Get completions from different models at the same time
- LLM benchmark: Evaluate models on quality, speed and cost

Feel free to reuse and expand. Pull requests are welcome.

## Installation

Install the package using pip:

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
models=llms.init(model=['gpt-3.5-turbo', 'claude-instant-v1'])

gpt4=llms.init('gpt-4') # optional, evaluator can be ommited and in that case only speed and cost will be evaluated

models.benchmark(evaluator=gpt4)
```

```
+---------------------------------------+--------------------+---------------------+----------------------+--------------------------+------------------+
|                 Model                 |       Tokens       |       Cost ($)      |     Latency (s)      |    Speed (tokens/sec)    |    Evaluation    |
+---------------------------------------+--------------------+---------------------+----------------------+--------------------------+------------------+
| AnthropicProvider (claude-instant-v1) |        211         |       0.00104       |         2.85         |          74.04           |        0         |
| AnthropicProvider (claude-instant-v1) |        204         |       0.00103       |         2.02         |          100.99          |        3         |
| AnthropicProvider (claude-instant-v1) |        139         |       0.00068       |         1.57         |          88.54           |        3         |
| AnthropicProvider (claude-instant-v1) |        138         |       0.00053       |         1.08         |          127.78          |        3         |
| AnthropicProvider (claude-instant-v1) |        139         |       0.00068       |         1.43         |          97.20           |        3         |
| AnthropicProvider (claude-instant-v1) |        284         |       0.00134       |         2.85         |          99.65           |        0         |
| AnthropicProvider (claude-instant-v1) |        266         |       0.00048       |         0.48         |          554.17          |        3         |
| AnthropicProvider (claude-instant-v1) |        247         |       0.00100       |         1.68         |          147.02          |        0         |
| AnthropicProvider (claude-instant-v1) |        211         |       0.00084       |         1.57         |          134.39          |        0         |
| AnthropicProvider (claude-instant-v1) |        180         |       0.00064       |         1.20         |          150.00          |        0         |
| AnthropicProvider (claude-instant-v1) |        247         |       0.00068       |         1.10         |          224.55          |        0         |
| AnthropicProvider (claude-instant-v1) |         62         |       0.00028       |         0.88         |          70.45           |        0         |
| AnthropicProvider (claude-instant-v1) |        175         |       0.00088       |         1.99         |          87.94           |        0         |
| AnthropicProvider (claude-instant-v1) |         55         |       0.00014       |         0.55         |          100.00          |        3         |
| AnthropicProvider (claude-instant-v1) |        188         |       0.00086       |         1.87         |          100.53          |        3         |
| AnthropicProvider (claude-instant-v1) |         79         |       0.00035       |         1.01         |          78.22           |        3         |
| AnthropicProvider (claude-instant-v1) |         49         |       0.00018       |         0.69         |          71.01           |        0         |
| AnthropicProvider (claude-instant-v1) |         40         |       0.00013       |         0.45         |          88.89           |        0         |
| AnthropicProvider (claude-instant-v1) |        165         |       0.00078       |         1.63         |          101.23          |        0         |
| AnthropicProvider (claude-instant-v1) |        215         |       0.00087       |         2.10         |          102.38          |        3         |
| AnthropicProvider (claude-instant-v1) |        375         |       0.00186       |         3.79         |          98.94           |        0         |
| AnthropicProvider (claude-instant-v1) |        188         |       0.00087       |         1.83         |          102.73          |        0         |
| AnthropicProvider (claude-instant-v1) | Total Tokens: 3857 | Total Cost: 0.01614 | Median Latency: 1.57 | Aggregated speed: 111.41 | Accuracy: 40.91% |
+---------------------------------------+--------------------+---------------------+----------------------+--------------------------+------------------+
+-----------------------------+----------------------+---------------------+----------------------+-------------------------+------------------+
|            Model            |        Tokens        |       Cost ($)      |     Latency (s)      |    Speed (tokens/sec)   |    Evaluation    |
+-----------------------------+----------------------+---------------------+----------------------+-------------------------+------------------+
| GoogleProvider (chat-bison) |        229.5         |       0.00046       |         7.24         |          31.70          |        0         |
| GoogleProvider (chat-bison) |        131.5         |       0.00026       |         4.83         |          27.23          |        3         |
| GoogleProvider (chat-bison) |        34.75         |       0.00007       |         2.36         |          14.72          |        0         |
| GoogleProvider (chat-bison) |        407.25        |       0.00081       |        14.02         |          29.05          |        0         |
| GoogleProvider (chat-bison) |         46.0         |       0.00009       |         1.81         |          25.41          |        0         |
| GoogleProvider (chat-bison) |        103.25        |       0.00021       |         2.73         |          37.82          |        0         |
| GoogleProvider (chat-bison) |        285.0         |       0.00057       |         1.61         |          177.02         |        0         |
| GoogleProvider (chat-bison) |        282.75        |       0.00057       |         6.82         |          41.46          |        0         |
| GoogleProvider (chat-bison) |        275.5         |       0.00055       |         8.49         |          32.45          |        0         |
| GoogleProvider (chat-bison) |        271.5         |       0.00054       |         8.67         |          31.31          |        0         |
| GoogleProvider (chat-bison) |        369.0         |       0.00074       |        11.50         |          32.09          |        3         |
| GoogleProvider (chat-bison) |        31.25         |       0.00006       |         2.69         |          11.62          |        3         |
| GoogleProvider (chat-bison) |        18.75         |       0.00004       |         1.24         |          15.12          |        0         |
| GoogleProvider (chat-bison) |         72.0         |       0.00014       |         2.35         |          30.64          |        0         |
| GoogleProvider (chat-bison) |        219.25        |       0.00044       |         5.81         |          37.74          |        0         |
| GoogleProvider (chat-bison) |         56.5         |       0.00011       |         2.74         |          20.62          |        0         |
| GoogleProvider (chat-bison) |        32.25         |       0.00006       |         2.00         |          16.12          |        0         |
| GoogleProvider (chat-bison) |         31.0         |       0.00006       |         1.16         |          26.72          |        0         |
| GoogleProvider (chat-bison) |        95.75         |       0.00019       |         3.36         |          28.50          |        3         |
| GoogleProvider (chat-bison) |        191.25        |       0.00038       |         4.65         |          41.13          |        0         |
| GoogleProvider (chat-bison) |        184.25        |       0.00037       |         5.51         |          33.44          |        0         |
| GoogleProvider (chat-bison) |        118.0         |       0.00024       |        10.66         |          11.07          |        0         |
| GoogleProvider (chat-bison) |        70.25         |       0.00014       |         2.48         |          28.33          |        0         |
| GoogleProvider (chat-bison) | Total Tokens: 3556.5 | Total Cost: 0.00711 | Median Latency: 3.36 | Aggregated speed: 31.00 | Accuracy: 17.39% |
+-----------------------------+----------------------+---------------------+----------------------+-------------------------+------------------+
+--------------------------------+--------------------+---------------------+----------------------+-------------------------+------------------+
|             Model              |       Tokens       |       Cost ($)      |     Latency (s)      |    Speed (tokens/sec)   |    Evaluation    |
+--------------------------------+--------------------+---------------------+----------------------+-------------------------+------------------+
| OpenAIProvider (gpt-3.5-turbo) |         93         |       0.00019       |         5.32         |          17.48          |        0         |
| OpenAIProvider (gpt-3.5-turbo) |        360         |       0.00072       |        26.39         |          13.64          |        3         |
| OpenAIProvider (gpt-3.5-turbo) |        143         |       0.00029       |         9.81         |          14.58          |        0         |
| OpenAIProvider (gpt-3.5-turbo) |        112         |       0.00022       |         5.07         |          22.09          |        0         |
| OpenAIProvider (gpt-3.5-turbo) |         34         |       0.00007       |         1.95         |          17.44          |        3         |
| OpenAIProvider (gpt-3.5-turbo) |         78         |       0.00016       |         1.84         |          42.39          |        3         |
| OpenAIProvider (gpt-3.5-turbo) |        254         |       0.00051       |         0.96         |          264.58         |        0         |
| OpenAIProvider (gpt-3.5-turbo) |        237         |       0.00047       |        12.18         |          19.46          |        3         |
| OpenAIProvider (gpt-3.5-turbo) |        358         |       0.00072       |        22.44         |          15.95          |        3         |
| OpenAIProvider (gpt-3.5-turbo) |        485         |       0.00097       |        32.37         |          14.98          |        3         |
| OpenAIProvider (gpt-3.5-turbo) |        217         |       0.00043       |         6.47         |          33.54          |        3         |
| OpenAIProvider (gpt-3.5-turbo) |         68         |       0.00014       |         4.56         |          14.91          |        3         |
| OpenAIProvider (gpt-3.5-turbo) |         33         |       0.00007       |         1.49         |          22.15          |        0         |
| OpenAIProvider (gpt-3.5-turbo) |         44         |       0.00009       |         0.48         |          91.67          |        0         |
| OpenAIProvider (gpt-3.5-turbo) |        157         |       0.00031       |         9.07         |          17.31          |        0         |
| OpenAIProvider (gpt-3.5-turbo) |         25         |       0.00005       |         0.76         |          32.89          |        0         |
| OpenAIProvider (gpt-3.5-turbo) |         27         |       0.00005       |         0.57         |          47.37          |        0         |
| OpenAIProvider (gpt-3.5-turbo) |         42         |       0.00008       |         2.13         |          19.72          |        3         |
| OpenAIProvider (gpt-3.5-turbo) |         44         |       0.00009       |         1.13         |          38.94          |        0         |
| OpenAIProvider (gpt-3.5-turbo) |        204         |       0.00041       |         9.98         |          20.44          |        3         |
| OpenAIProvider (gpt-3.5-turbo) |        279         |       0.00056       |        19.25         |          14.49          |        3         |
| OpenAIProvider (gpt-3.5-turbo) |        333         |       0.00067       |        23.85         |          13.96          |        3         |
| OpenAIProvider (gpt-3.5-turbo) |         81         |       0.00016       |         1.40         |          57.86          |        0         |
| OpenAIProvider (gpt-3.5-turbo) | Total Tokens: 3708 | Total Cost: 0.00743 | Median Latency: 5.07 | Aggregated speed: 18.59 | Accuracy: 52.17% |
+--------------------------------+--------------------+---------------------+----------------------+-------------------------+------------------+
+-------------------------------+--------------------+---------------------+----------------------+-------------------------+------------------+
|             Model             |       Tokens       |       Cost ($)      |     Latency (s)      |    Speed (tokens/sec)   |    Evaluation    |
+-------------------------------+--------------------+---------------------+----------------------+-------------------------+------------------+
| AnthropicProvider (claude-v1) |        316         |       0.00966       |        10.32         |          30.62          |        3         |
| AnthropicProvider (claude-v1) |        177         |       0.00529       |         4.30         |          41.16          |        3         |
| AnthropicProvider (claude-v1) |        116         |       0.00334       |         2.52         |          46.03          |        0         |
| AnthropicProvider (claude-v1) |        135         |       0.00311       |         2.96         |          45.61          |        3         |
| AnthropicProvider (claude-v1) |        214         |       0.00652       |         5.15         |          41.55          |        3         |
| AnthropicProvider (claude-v1) |        249         |       0.00688       |         4.87         |          51.13          |        3         |
| AnthropicProvider (claude-v1) |        261         |       0.00301       |         0.93         |          280.65         |        0         |
| AnthropicProvider (claude-v1) |        223         |       0.00527       |         3.37         |          66.17          |        3         |
| AnthropicProvider (claude-v1) |        220         |       0.00537       |         4.70         |          46.81          |        3         |
| AnthropicProvider (claude-v1) |        390         |       0.01080       |         7.83         |          49.81          |        3         |
| AnthropicProvider (claude-v1) |        263         |       0.00480       |         3.08         |          85.39          |        3         |
| AnthropicProvider (claude-v1) |         68         |       0.00185       |         2.41         |          28.22          |        3         |
| AnthropicProvider (claude-v1) |         36         |       0.00072       |         1.01         |          35.64          |        0         |
| AnthropicProvider (claude-v1) |        112         |       0.00275       |         2.20         |          50.91          |        0         |
| AnthropicProvider (claude-v1) |        243         |       0.00694       |         6.88         |          35.32          |        0         |
| AnthropicProvider (claude-v1) |         85         |       0.00230       |         2.98         |          28.52          |        0         |
| AnthropicProvider (claude-v1) |         83         |       0.00219       |         1.91         |          43.46          |        0         |
| AnthropicProvider (claude-v1) |         39         |       0.00078       |         0.80         |          48.75          |        3         |
| AnthropicProvider (claude-v1) |         67         |       0.00145       |         1.02         |          65.69          |        0         |
| AnthropicProvider (claude-v1) |        183         |       0.00420       |         3.74         |          48.93          |        3         |
| AnthropicProvider (claude-v1) |        360         |       0.01062       |         8.67         |          41.52          |        0         |
| AnthropicProvider (claude-v1) |        220         |       0.00624       |         5.19         |          42.39          |        0         |
| AnthropicProvider (claude-v1) |        258         |       0.00700       |         5.36         |          48.13          |        0         |
| AnthropicProvider (claude-v1) | Total Tokens: 4318 | Total Cost: 0.11109 | Median Latency: 3.37 | Aggregated speed: 46.83 | Accuracy: 52.17% |
+-------------------------------+--------------------+---------------------+----------------------+-------------------------+------------------+
+------------------------+--------------------+---------------------+-----------------------+------------------------+------------------+
|         Model          |       Tokens       |       Cost ($)      |      Latency (s)      |   Speed (tokens/sec)   |    Evaluation    |
+------------------------+--------------------+---------------------+-----------------------+------------------------+------------------+
| OpenAIProvider (gpt-4) |         82         |       0.00402       |         18.27         |          4.49          |        0         |
| OpenAIProvider (gpt-4) |        123         |       0.00660       |         18.00         |          6.83          |        3         |
| OpenAIProvider (gpt-4) |        147         |       0.00813       |         21.49         |          6.84          |        3         |
| OpenAIProvider (gpt-4) |        109         |       0.00489       |         10.95         |          9.95          |        3         |
| OpenAIProvider (gpt-4) |         44         |       0.00198       |          4.63         |          9.50          |        3         |
| OpenAIProvider (gpt-4) |         76         |       0.00285       |          3.31         |         22.96          |        3         |
| OpenAIProvider (gpt-4) |        258         |       0.00813       |          3.46         |         74.57          |        3         |
| OpenAIProvider (gpt-4) |        250         |       0.01227       |         52.10         |          4.80          |        3         |
| OpenAIProvider (gpt-4) |        182         |       0.00840       |         21.51         |          8.46          |        3         |
| OpenAIProvider (gpt-4) |        449         |       0.02418       |         59.39         |          7.56          |        3         |
| OpenAIProvider (gpt-4) |        198         |       0.00729       |         11.44         |         17.31          |        3         |
| OpenAIProvider (gpt-4) |         69         |       0.00366       |         10.97         |          6.29          |        3         |
| OpenAIProvider (gpt-4) |         30         |       0.00123       |          2.91         |         10.31          |        0         |
| OpenAIProvider (gpt-4) |         43         |       0.00135       |          1.44         |         29.86          |        0         |
| OpenAIProvider (gpt-4) |        317         |       0.01761       |         60.61         |          5.23          |        0         |
| OpenAIProvider (gpt-4) |         24         |       0.00078       |          1.16         |         20.69          |        0         |
| OpenAIProvider (gpt-4) |         53         |       0.00246       |          6.17         |          8.59          |        0         |
| OpenAIProvider (gpt-4) |         41         |       0.00177       |          5.26         |          7.79          |        0         |
| OpenAIProvider (gpt-4) |         41         |       0.00144       |          2.16         |         18.98          |        3         |
| OpenAIProvider (gpt-4) |        186         |       0.00867       |         17.17         |         10.83          |        3         |
| OpenAIProvider (gpt-4) |        188         |       0.00963       |         22.88         |          8.22          |        0         |
| OpenAIProvider (gpt-4) |        143         |       0.00717       |         16.05         |          8.91          |        0         |
| OpenAIProvider (gpt-4) |         89         |       0.00330       |          3.81         |         23.36          |        3         |
| OpenAIProvider (gpt-4) | Total Tokens: 3142 | Total Cost: 0.14781 | Median Latency: 10.97 | Aggregated speed: 8.38 | Accuracy: 60.87% |
+------------------------+--------------------+---------------------+-----------------------+------------------------+------------------+
+--------------------------+--------------------+---------------------+----------------------+-------------------------+------------------+
|          Model           |       Tokens       |       Cost ($)      |     Latency (s)      |    Speed (tokens/sec)   |    Evaluation    |
+--------------------------+--------------------+---------------------+----------------------+-------------------------+------------------+
| CohereProvider (command) |         58         |       0.00145       |         1.32         |          43.94          |        3         |
| CohereProvider (command) |        123         |       0.00308       |         3.38         |          36.39          |        0         |
| CohereProvider (command) |         19         |       0.00047       |        34.05         |           0.56          |        0         |
| CohereProvider (command) |        118         |       0.00295       |         2.63         |          44.87          |        0         |
| CohereProvider (command) |         37         |       0.00093       |         0.91         |          40.66          |        0         |
| CohereProvider (command) |         60         |       0.00150       |         0.60         |          100.00         |        3         |
| CohereProvider (command) |        271         |       0.00677       |         1.02         |          265.69         |        0         |
| CohereProvider (command) |        329         |       0.00822       |         5.41         |          60.81          |        0         |
| CohereProvider (command) |        1267        |       0.03168       |        30.64         |          41.35          |        0         |
| CohereProvider (command) |        431         |       0.01077       |         6.71         |          64.23          |        0         |
| CohereProvider (command) |        223         |       0.00558       |         1.92         |          116.15         |        0         |
| CohereProvider (command) |         26         |       0.00065       |         0.77         |          33.77          |        0         |
| CohereProvider (command) |         23         |       0.00057       |         0.63         |          36.51          |        0         |
| CohereProvider (command) |         37         |       0.00093       |         0.35         |          105.71         |        0         |
| CohereProvider (command) |         47         |       0.00118       |         0.54         |          87.04          |        0         |
| CohereProvider (command) |         18         |       0.00045       |         0.38         |          47.37          |        0         |
| CohereProvider (command) |         38         |       0.00095       |         0.78         |          48.72          |        3         |
| CohereProvider (command) |         34         |       0.00085       |         0.79         |          43.04          |        0         |
| CohereProvider (command) |         30         |       0.00075       |         0.37         |          81.08          |        0         |
| CohereProvider (command) |         79         |       0.00198       |         0.44         |          179.55         |        0         |
| CohereProvider (command) |         60         |       0.00150       |         0.71         |          84.51          |        0         |
| CohereProvider (command) |         47         |       0.00118       |         0.57         |          82.46          |        0         |
| CohereProvider (command) |         69         |       0.00172       |         0.54         |          127.78         |        0         |
| CohereProvider (command) | Total Tokens: 3444 | Total Cost: 0.08611 | Median Latency: 0.78 | Aggregated speed: 36.08 | Accuracy: 13.04% |
+--------------------------+--------------------+---------------------+----------------------+-------------------------+------------------+
+-----------------------------------------------+--------------------+---------------------+----------------------+-------------------------+-----------------+
|                     Model                     |       Tokens       |       Cost ($)      |     Latency (s)      |    Speed (tokens/sec)   |    Evaluation   |
+-----------------------------------------------+--------------------+---------------------+----------------------+-------------------------+-----------------+
| AlephAlphaProvider (luminous-supreme-control) |         24         |       0.00117       |         1.21         |          19.83          |        0        |
| AlephAlphaProvider (luminous-supreme-control) |         24         |       0.00119       |         1.27         |          18.90          |        0        |
| AlephAlphaProvider (luminous-supreme-control) |         21         |       0.00104       |         1.13         |          18.58          |        0        |
| AlephAlphaProvider (luminous-supreme-control) |        114         |       0.00587       |         7.18         |          15.88          |        0        |
| AlephAlphaProvider (luminous-supreme-control) |         18         |       0.00089       |         0.91         |          19.78          |        0        |
| AlephAlphaProvider (luminous-supreme-control) |         52         |       0.00253       |         0.85         |          61.18          |        0        |
| AlephAlphaProvider (luminous-supreme-control) |        277         |       0.01363       |         5.57         |          49.73          |        0        |
| AlephAlphaProvider (luminous-supreme-control) |        149         |       0.00756       |        10.03         |          14.86          |        0        |
| AlephAlphaProvider (luminous-supreme-control) |        108         |       0.00540       |         5.79         |          18.65          |        0        |
| AlephAlphaProvider (luminous-supreme-control) |        191         |       0.00980       |        15.48         |          12.34          |        0        |
| AlephAlphaProvider (luminous-supreme-control) |        148         |       0.00719       |         0.88         |          168.18         |        0        |
| AlephAlphaProvider (luminous-supreme-control) |         30         |       0.00156       |         3.33         |           9.01          |        0        |
| AlephAlphaProvider (luminous-supreme-control) |         13         |       0.00064       |         0.71         |          18.31          |        0        |
| AlephAlphaProvider (luminous-supreme-control) |         36         |       0.00176       |         0.81         |          44.44          |        0        |
| AlephAlphaProvider (luminous-supreme-control) |         41         |       0.00199       |         0.72         |          56.94          |        0        |
| AlephAlphaProvider (luminous-supreme-control) |         29         |       0.00148       |         2.21         |          13.12          |        0        |
| AlephAlphaProvider (luminous-supreme-control) |         24         |       0.00120       |         1.40         |          17.14          |        0        |
| AlephAlphaProvider (luminous-supreme-control) |         17         |       0.00083       |         0.71         |          23.94          |        0        |
| AlephAlphaProvider (luminous-supreme-control) |         29         |       0.00142       |         0.83         |          34.94          |        3        |
| AlephAlphaProvider (luminous-supreme-control) |         77         |       0.00374       |         0.73         |          105.48         |        0        |
| AlephAlphaProvider (luminous-supreme-control) |         49         |       0.00238       |         0.68         |          72.06          |        0        |
| AlephAlphaProvider (luminous-supreme-control) |         41         |       0.00199       |         0.67         |          61.19          |        0        |
| AlephAlphaProvider (luminous-supreme-control) |        103         |       0.00521       |         5.58         |          18.46          |        3        |
| AlephAlphaProvider (luminous-supreme-control) | Total Tokens: 1615 | Total Cost: 0.08047 | Median Latency: 1.13 | Aggregated speed: 23.51 | Accuracy: 8.70% |
+-----------------------------------------------+--------------------+---------------------+----------------------+-------------------------+-----------------+

```

To evaluate models on your own prompts, simply pass a list of questions and optional answers as tuple. The evaluator will automatically evaluate the responses:

```
models.benchmark(prompts=[("what is the capital of finland", "helsinki")], evaluator=gpt4)
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
| CohereProvider      | command                |        25.0 |            25.0 |        8192 |
| CohereProvider      | command-nightly        |        25.0 |            25.0 |        8192 |
| GoogleProvider      | chat-bison             |         0.5 |             0.5 |        2048 |
| HuggingfaceHub      | hf_pythia              |         0.0 |             0.0 |        2048 |
| OpenAIProvider      | gpt-3.5-turbo          |         2.0 |             2.0 |        4000 |
| OpenAIProvider      | gpt-4                  |        30.0 |            60.0 |        8000 |

```

Useful links:\
[OpenAI documentation](https://platform.openai.com/docs/api-reference/completions)\
[Anthropic documentation](https://console.anthropic.com/docs/api/reference#-v1-complete)\
[AI21 documentation](https://docs.ai21.com/reference/j2-instruct-ref)\
[Cohere documentation](https://cohere-sdk.readthedocs.io/en/latest/cohere.html#api)\
[Aleph Alpha documentation](https://aleph-alpha-client.readthedocs.io/en/latest/aleph_alpha_client.html#aleph_alpha_client.CompletionRequest)\
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


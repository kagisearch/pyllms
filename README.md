# PyLLMs

[![PyPI version](https://badge.fury.io/py/pyllms.svg)](https://badge.fury.io/py/pyllms)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/license/mit/)
[![](https://dcbadge.vercel.app/api/server/aDNg6E9szy?compact=true&style=flat)](https://discord.gg/aDNg6E9szy)
[![Twitter](https://img.shields.io/twitter/follow/KagiHQ?style=social)](https://twitter.com/KagiHQ)

PyLLMs is a minimal Python library to connect to various Language Models (LLMs) with a built-in model performance benchmark.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Basic Usage](#basic-usage)
  - [Multi-model Usage](#multi-model-usage)
  - [Async Support](#async-support)
  - [Streaming Support](#streaming-support)
  - [Chat History and System Message](#chat-history-and-system-message)
  - [Other Methods](#other-methods)
- [Configuration](#configuration)
- [Model Benchmarks](#model-benchmarks)
- [Supported Models](#supported-models)
- [Advanced Usage](#advanced-usage)
  - [Using OpenAI API on Azure](#using-openai-api-on-azure)
  - [Using Google Vertex LLM models](#using-google-vertex-llm-models)
  - [Using Local Ollama LLM models](#using-local-ollama-llm-models)
- [Contributing](#contributing)
- [License](#license)

## Features

- Connect to top LLMs in a few lines of code
- Response meta includes tokens processed, cost, and latency standardized across models
- Multi-model support: Get completions from different models simultaneously
- LLM benchmark: Evaluate models on quality, speed, and cost
- Async and streaming support for compatible models

## Installation

Install the package using pip:

```bash
pip install pyllms
```

## Quick Start

```python
import llms

model = llms.init('gpt-4o')
result = model.complete("What is 5+5?")

print(result.text)
```

## Usage

### Basic Usage

```python
import llms

model = llms.init('gpt-4o')
result = model.complete(
    "What is the capital of the country where Mozart was born?",
    temperature=0.1,
    max_tokens=200
)

print(result.text)
print(result.meta)
```

### Multi-model Usage

```python
models = llms.init(model=['gpt-3.5-turbo', 'claude-instant-v1'])
result = models.complete('What is the capital of the country where Mozart was born?')

print(result.text)
print(result.meta)
```

### Async Support

```python
result = await model.acomplete("What is the capital of the country where Mozart was born?")
```

### Streaming Support

```python
model = llms.init('claude-v1')
result = model.complete_stream("Write an essay on the Civil War")
for chunk in result.stream:
   if chunk is not None:
      print(chunk, end='')
```

### Chat History and System Message

```python
history = []
history.append({"role": "user", "content": user_input})
history.append({"role": "assistant", "content": result.text})

model.complete(prompt=prompt, history=history)

# For OpenAI chat models
model.complete(prompt=prompt, system_message=system, history=history)
```

### Other Methods

```python
count = model.count_tokens('The quick brown fox jumped over the lazy dog')
```

## Configuration

PyLLMs will attempt to read API keys and the default model from environment variables. You can set them like this:

```bash
export OPENAI_API_KEY="your_api_key_here"
export ANTHROPIC_API_KEY="your_api_key_here"
export AI21_API_KEY="your_api_key_here"
export COHERE_API_KEY="your_api_key_here"
export ALEPHALPHA_API_KEY="your_api_key_here"
export HUGGINFACEHUB_API_KEY="your_api_key_here"
export GOOGLE_API_KEY="your_api_key_here"
export MISTRAL_API_KEY="your_api_key_here"
export REKA_API_KEY="your_api_key_here"
export TOGETHER_API_KEY="your_api_key_here"
export GROQ_API_KEY="your_api_key_here"
export DEEPSEEK_API_KEY="your_api_key_here"

export LLMS_DEFAULT_MODEL="gpt-3.5-turbo"
```

Alternatively, you can pass initialization values to the `init()` method:

```python
model = llms.init(openai_api_key='your_api_key_here', model='gpt-4')
```

## Model Benchmarks

PyLLMs includes an automated benchmark system. The quality of models is evaluated using a powerful model (e.g., GPT-4) on a range of predefined questions, or you can supply your own.

```python
model = llms.init(model=['claude-3-haiku-20240307', 'gpt-4o-mini', 'claude-3-5-sonnet-20240620', 'gpt-4o', 'mistral-large-latest', 'open-mistral-nemo', 'gpt-4', 'gpt-3.5-turbo', 'deepseek-coder', 'deepseek-chat', 'llama-3.1-8b-instant', 'llama-3.1-70b-versatile'])

gpt4 = llms.init('gpt-4o')

models.benchmark(evaluator=gpt4)
```

Check [Kagi LLM Benchmarking Project](https://help.kagi.com/kagi/ai/llm-benchmark.html) for the latest benchmarks!

To evaluate models on your own prompts:

```python
models.benchmark(prompts=[("What is the capital of Finland?", "Helsinki")], evaluator=gpt4)
```

## Supported Models

To get a full list of supported models:

```python
model = llms.init()
model.list() # list all models

model.list("gpt")  # lists only models with 'gpt' in name/provider name
```

Currently supported models (may be outdated):

| **Provider**             | **Models**                                                                                                                                                                                                                                                                                             |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| OpenAIProvider           | gpt-3.5-turbo, gpt-3.5-turbo-1106, gpt-3.5-turbo-instruct, gpt-4, gpt-4-1106-preview, gpt-4-turbo-preview, gpt-4-turbo, gpt-4o, gpt-4o-mini, gpt-4o-2024-08-06, gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, gpt-4.5-preview, chatgpt-4o-latest, o1-preview, o1-mini, o1, o1-pro, o3-mini, o3, o3-pro, o4-mini |
| AnthropicProvider        | claude-2.1, claude-3-5-sonnet-20240620, claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022, claude-3-7-sonnet-20250219, claude-sonnet-4-20250514, claude-opus-4-20250514                                                                                                                            |
| BedrockAnthropicProvider | anthropic.claude-instant-v1, anthropic.claude-v1, anthropic.claude-v2, anthropic.claude-3-haiku-20240307-v1:0, anthropic.claude-3-sonnet-20240229-v1:0, anthropic.claude-3-5-sonnet-20240620-v1:0                                                                                                      |
| AI21Provider             | j2-grande-instruct, j2-jumbo-instruct                                                                                                                                                                                                                                                                  |
| CohereProvider           | command, command-nightly                                                                                                                                                                                                                                                                               |
| AlephAlphaProvider       | luminous-base, luminous-extended, luminous-supreme, luminous-supreme-control                                                                                                                                                                                                                           |
| HuggingfaceHubProvider   | hf_pythia, hf_falcon40b, hf_falcon7b, hf_mptinstruct, hf_mptchat, hf_llava, hf_dolly, hf_vicuna                                                                                                                                                                                                        |
| GoogleGenAIProvider      | gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite-preview-06-17, gemini-2.0-flash, gemini-2.0-flash-lite, gemini-1.5-pro, gemini-1.5-flash, gemini-1.5-flash-8b                                                                                                                                  |
| GoogleVertexAIProvider   | gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite-preview-06-17, gemini-2.0-flash, gemini-2.0-flash-lite, gemini-1.5-pro, gemini-1.5-flash, gemini-1.5-flash-8b                                                                                                                                  |
| OllamaProvider           | vanilj/Phi-4:latest, falcon3:10b, smollm2:latest, llama3.2:3b-instruct-q8_0, qwen2:1.5b, mistral:7b-instruct-v0.2-q4_K_S, phi3:latest, phi3:3.8b, phi:latest, tinyllama:latest, magicoder:latest, deepseek-coder:6.7b, deepseek-coder:latest, dolphin-phi:latest, stablelm-zephyr:latest               |
| DeepSeekProvider         | deepseek-chat, deepseek-coder                                                                                                                                                                                                                                                                          |
| GroqProvider             | llama-3.1-405b-reasoning, llama-3.1-70b-versatile, llama-3.1-8b-instant, gemma2-9b-it                                                                                                                                                                                                                  |
| RekaProvider             | reka-edge, reka-flash, reka-core                                                                                                                                                                                                                                                                       |
| TogetherProvider         | meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo                                                                                                                                                                                                                                                          |
| OpenRouterProvider       | nvidia/llama-3.1-nemotron-70b-instruct, x-ai/grok-2, nousresearch/hermes-3-llama-3.1-405b:free, google/gemini-flash-1.5-exp, liquid/lfm-40b, mistralai/ministral-8b, qwen/qwen-2.5-72b-instruct                                                                                                        |
| MistralProvider          | mistral-tiny, open-mistral-7b, mistral-small, open-mixtral-8x7b, mistral-small-latest, mistral-medium-latest, mistral-large-latest, open-mistral-nemo                                                                                                                                                  |

## Advanced Usage

### Using OpenAI API on Azure

```python
import llms
AZURE_API_BASE = "{insert here}"
AZURE_API_KEY = "{insert here}"

model = llms.init('gpt-4')

azure_args = {
    "engine": "gpt-4",  # Azure deployment_id
    "api_base": AZURE_API_BASE,
    "api_type": "azure",
    "api_version": "2023-05-15",
    "api_key": AZURE_API_KEY,
}

azure_result = model.complete("What is 5+5?", **azure_args)
```

### Using Google AI Models

PyLLMs supports Google's AI models through two providers:

#### Option 1: Gemini API (GoogleGenAI)

Uses direct Gemini API with API key authentication:

```python
# Set your API key
export GOOGLE_API_KEY="your_api_key_here"

# Use any Gemini model
model = llms.init('gemini-2.5-flash')
result = model.complete("Hello!")
```

#### Option 2: Vertex AI (GoogleVertexAI)

Uses Google Cloud Vertex AI with Application Default Credentials:

1. Set up a GCP account and create a project
2. Enable Vertex AI APIs in your GCP project
3. Install gcloud CLI tool
4. Set up Application Default Credentials:
   ```bash
   gcloud auth application-default login
   gcloud config set project YOUR_PROJECT_ID
   ```

Then use models through Vertex AI:

```python
# Option A: Direct provider usage for Vertex AI
from llms.providers.google_genai import GoogleVertexAIProvider
provider = GoogleVertexAIProvider()
result = provider.complete("Hello!")

# Option B: Unified provider with Vertex AI flag
from llms.providers.google_genai import GoogleGenAIProvider
provider = GoogleGenAIProvider(use_vertexai=True)
result = provider.complete("Hello!")
```

**Note:** Both providers support the same model names. If both `GOOGLE_API_KEY` and gcloud credentials are configured, `llms.init('gemini-2.5-flash')` will use both providers simultaneously.

### Using Local Ollama LLM models

1. Ensure Ollama is running and you've pulled the desired model
2. Get the name of the LLM you want to use
3. Initialize PyLLMs:

```python
model = llms.init("tinyllama:latest")
result = model.complete("Hello!")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

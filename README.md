# LLMS

A Python package for easy interaction with popular language models like OpenAI's GPT models.

## Installation

Clone this repository and install the package using pip:

```bash
git clone https://github.com/yourusername/llms.git
cd llms
pip install -e .


## Usage

To use the llms package, you'll need to set the required environment variables. For OpenAI, you'll need to set the OPENAI_API_KEY:

```
export OPENAI_API_KEY="your_api_key_here"


Here's an example of how to use the llms package:

```
import llms

model = llms.LLMS()

result = model.complete("write a blog post about social cognitive theory")

print(result.text)
print(result.meta)
```

You can also pass optional parameters to the complete method:

```
result = model.complete(
    "write a blog post about social cognitive theory",
    temperature=0.8,
    max_tokens=100,
    stop=["\n"]
)
```

## Supported Models

Currently, the package supports OpenAI's GPT models:

gpt-4
gpt-3.5-turbo

## License

This project is licensed under the MIT License.


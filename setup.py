from setuptools import setup, find_packages
import pathlib

# Get the long description from the README file
here = pathlib.Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

_project_homepage = "https://github.com/kagisearch/pyllms"

setup(
    name="pyllms",
    version="0.1.9",
    description="Minimal Python library to connect to LLMs (OpenAI, Anthropic, AI21, Cohere, Aleph-Alpha, HuggingfaceHub), with a built-in model performance benchmark.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Vladimir Prelovac",
    author_email="vlad@kagi.com",
    packages=find_packages(),
    install_requires=[
        "openai",
        "tiktoken",
        "anthropic",
        "ai21",
        "cohere",
        "aleph-alpha-client",
        "huggingface_hub",
        "prettytable",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Human Machine Interfaces",
        "Topic :: Text Processing",
    ],
    python_requires=">=3.7",
    keywords="llm, llms, large language model, AI, NLP, natural language processing, gpt, chatgpt, openai, anthropic, ai21, cohere, aleph alpha, huggingface hub",
    project_urls={
        "Documentation": _project_homepage,
        "Source Code": _project_homepage,
        "Issue Tracker": _project_homepage+"/issues",
    },
)

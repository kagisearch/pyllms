from .llms import LLMS


def init(*args, **kwargs):
    if len(args) > 1 and not kwargs.get("model"):
        msg = "Please provide a list of models, like this: model=['j2-grande-instruct', 'claude-v1', 'gpt-3.5-turbo']"
        raise ValueError(msg)
    return LLMS(*args, **kwargs)

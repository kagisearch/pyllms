from .llms import LLMS

def init(*args, **kwargs):
    return LLMS(*args, **kwargs)
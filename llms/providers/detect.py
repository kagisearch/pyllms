from typing import Dict
from llms.llms import LLMS

def check_feed(url: str) -> bool:
    """Check if a URL is a valid RSS feed."""
    # Implementation would go here
    pass

def report_model_accuracy() -> Dict[str, Dict[str, int]]:
    """Report accuracy statistics for different models."""
    model = LLMS()
    model_decisions = {model_name: {"correct": 0, "total": 0} for model_name in model._provider_map}
    return model_decisions

bad_domains = set()
bad_rss = set()

"""Run the websearch pipeline."""
from mmore.websearch.pipeline import WebsearchPipeline
from mmore.websearch.config import WebsearchConfig

def run_websearch(config):
    """Run the websearch pipeline."""
    pipeline = WebsearchPipeline(WebsearchConfig.from_dict(config))
    pipeline.run() 
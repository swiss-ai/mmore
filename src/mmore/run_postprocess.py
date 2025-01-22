from typing import List, Union

from mmore.process.post_processor.pipeline import PPPipelineConfig, PPPipeline

from mmore.type import MultimodalSample
from mmore.utils import load_config

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Global logging configuration
#logging.basicConfig(format='%(asctime)s: %(message)s')
#logging.basicConfig(format='%(message)s')
logging.basicConfig(format='[PP 🧹-- %(asctime)s] %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

# TODO: We should find a way to load the dataset in a more generic way
def _load_dataset(data_path: List[str]) -> List[MultimodalSample]:
    samples = [s for path in data_path for s in MultimodalSample.from_jsonl(path)]
    return samples

def postprocess(config_file, input_data):
    """Run post-processors pipeline."""
    if isinstance(input_data, str):
            input_data = [input_data]
    # Load config
    config = load_config(config_file, PPPipelineConfig)

    # Load post-processors pipeline
    pipeline = PPPipeline.from_config(config)

    # Load samples
    samples = _load_dataset(config.data_path)

    # Run pipeline
    samples = pipeline(samples)

if __name__ == "__main__":
    postprocess()
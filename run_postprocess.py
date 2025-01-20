import argparse

from dataclasses import dataclass
from typing import List

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

@dataclass
class PPInferenceConfig:
    data_path: str | List[str]
    pipeline: PPPipelineConfig

    def __post_init__(self):
        if isinstance(self.data_path, str):
            self.data_path = [self.data_path]

def get_args():
    parser = argparse.ArgumentParser(description='Run RAG pipeline with API or CLI mode')
    parser.add_argument('--config-file', type=str, required=True, help='Path to a config file')
    return parser.parse_args()

# TODO: We should find a way to load the dataset in a more generic way
def _load_dataset(data_path: List[str]) -> List[MultimodalSample]:
    samples = [s for path in data_path for s in MultimodalSample.from_jsonl(path)]
    return samples

if __name__ == "__main__":
    args = get_args()

    # Load config
    config = load_config(args.config_file, PPInferenceConfig)

    # Load post-processors pipeline
    pipeline = PPPipeline.from_config(config.pipeline)

    # Load samples
    samples = _load_dataset(config.data_path)

    # Run pipeline
    samples = pipeline(samples)
from typing import List, Union
import argparse
import logging

PP_EMOJI = "🧹"
logger = logging.getLogger(__name__)
logging.basicConfig(format=f'[PP {PP_EMOJI}-- %(asctime)s] %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

from mmore.process.post_processor.pipeline import PPPipelineConfig, PPPipeline
from mmore.type import MultimodalSample
from mmore.utils import load_config

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
    samples = _load_dataset(input_data)

    # Run pipeline
    samples = pipeline(samples)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True, help="Path to the postprocess configuration file.")
    parser.add_argument("--input_data", required=True, help="Path to the jsonl of the documents.")

    args = parser.parse_args()
    postprocess(args.config_file, args.input_data)
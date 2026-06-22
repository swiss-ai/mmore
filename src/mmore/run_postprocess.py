import argparse
from typing import List

from mmore.process.post_processor.pipeline import PPPipeline, PPPipelineConfig
from mmore.profiler import enable_profiling_from_env, profile_function
from mmore.type import MultimodalSample
from mmore.utils import load_config
from mmore.ux import setup_logging

PP_EMOJI = "🧹"
logger = setup_logging("POST-PROCESS", PP_EMOJI)


def _load_dataset(data_path: List[str]) -> List[MultimodalSample]:
    return [s for path in data_path for s in MultimodalSample.from_jsonl(path)]


@profile_function()
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
    if len(samples) == 0:
        logger.warning("⚠️ Found no file to postprocess")

    # Run pipeline
    samples = pipeline(samples)


if __name__ == "__main__":
    enable_profiling_from_env()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        required=True,
        help="Path to the postprocess configuration file.",
    )
    parser.add_argument(
        "--input_data", required=True, help="Path to the jsonl of the documents."
    )

    args = parser.parse_args()
    postprocess(args.config_file, args.input_data)

"""Helper script to compare table chunking modes on real data.

Usage:
    uv run python test_table_chunker.py <path_to_jsonl> [sample_index]

Example:
    uv run python test_table_chunker.py examples/process/outputs/merged/merged_results.jsonl 22 2>/dev/null

Note:
    2>/dev/null redirects stderr to suppress warnings
"""

import sys

from mmore.process.post_processor.chunker.multimodal import (
    MultimodalChunker,
    MultimodalChunkerConfig,
)
from mmore.type import MultimodalSample


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <path_to_jsonl> [sample_index]")
        sys.exit(1)

    jsonl_path = sys.argv[1]
    sample_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    samples = MultimodalSample.from_jsonl(jsonl_path)
    if not samples:
        print(f"No samples found in {jsonl_path}")
        sys.exit(1)

    if sample_idx >= len(samples):
        print(f"Sample index {sample_idx} out of range (total: {len(samples)})")
        sys.exit(1)

    sample = samples[sample_idx]
    print(f"Sample {sample_idx} | id={sample.id} | text length={len(sample.text)}")
    print(f"First 200 chars: {sample.text[:200]}")
    print("=" * 80)

    for mode in ("none", "preserve_headers", "keep_whole"):
        config = MultimodalChunkerConfig(
            chunking_strategy="sentence",
            text_chunker_config={"chunk_size": 512, "chunk_overlap": 0},
            table_handling=mode,
        )
        chunker = MultimodalChunker.from_config(config)
        chunks = chunker.chunk(sample)

        print(f"\n--- {mode} ({len(chunks)} chunks) ---")
        for i, c in enumerate(chunks):
            is_table = c.metadata.get("is_table_chunk", False)
            preview = c.text[:120].replace("\n", "\\n")
            print(
                f"  [{i}] table={is_table} | tokens~{len(c.text.split())} | {preview}..."
            )


if __name__ == "__main__":
    main()

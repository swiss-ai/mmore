import json

import pytest

from mmore.type import MultimodalSample


@pytest.fixture
def make_sample():
    def _make(file_path: str, text: str = "x", **metadata) -> MultimodalSample:
        return MultimodalSample.from_dict(
            {
                "text": text,
                "modalities": [],
                "metadata": {"file_path": file_path, **metadata},
            }
        )

    return _make


@pytest.fixture
def write_jsonl():
    def _write(path: str, samples: list[MultimodalSample]) -> None:
        with open(path, "w") as f:
            for s in samples:
                f.write(json.dumps(s.to_dict()) + "\n")

    return _write

import json

import pytest

from mmore.privacy.agents.base import clear_llm_cache
from mmore.privacy.agents.registry import tool_registry
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


@pytest.fixture
def isolate_llm_cache():
    clear_llm_cache()
    yield
    clear_llm_cache()


@pytest.fixture
def isolated_tool_registry():
    snapshot = dict(tool_registry)
    tool_registry.clear()
    yield
    tool_registry.clear()
    tool_registry.update(snapshot)

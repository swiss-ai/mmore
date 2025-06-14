import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.mmore.process.post_processor import BasePostProcessorConfig, load_postprocessor
from src.mmore.process.post_processor.chunker.multimodal import (
    MultimodalChunker,
    MultimodalChunkerConfig,
)
from src.mmore.process.post_processor.filter import FILTER_TYPES, FILTERS_LOADERS_MAP
from src.mmore.process.post_processor.filter.base import BaseFilter, BaseFilterConfig
from src.mmore.process.post_processor.ner import NERecognizer, NERExtractorConfig
from src.mmore.process.post_processor.tagger import load_tagger
from src.mmore.process.post_processor.tagger.base import BaseTaggerConfig
from src.mmore.process.post_processor.tagger.lang_detector import LangDetector
from src.mmore.process.post_processor.tagger.modalities import ModalitiesCounter
from src.mmore.process.post_processor.tagger.words import WordsCounter
from src.mmore.rag.llm import LLM
from src.mmore.type import MultimodalSample


# ------------------ Chunker Tests ------------------
def test_chunker_from_load_postprocessor():
    """
    Verify that load_postprocessor returns a MultimodalChunker when given a chunker config.
    """
    config_args = {"chunking_strategy": "sentence", "text_chunker_config": {}}
    base_config = BasePostProcessorConfig(type="chunker", args=config_args)
    processor = load_postprocessor(base_config)
    assert isinstance(processor, MultimodalChunker), (
        "Expected a MultimodalChunker instance."
    )


def test_chunker_process():
    """
    Test that the chunker splits a simple sentence-based text into multiple chunks.
    """
    config = MultimodalChunkerConfig(
        chunking_strategy="sentence",
        text_chunker_config={"chunk_size": 5, "chunk_overlap": 0},
    )
    chunker = MultimodalChunker.from_config(config)
    sample = MultimodalSample(
        text="Hello world. This is a test.", modalities=[], metadata={}
    )
    chunks = chunker.process(sample)
    print(f"chunks: {chunks}")
    # Expect 2 chunks for the 2 sentences
    assert len(chunks) == 2, f"Expected 2 chunks, got {len(chunks)}"
    assert chunks[0].text.strip() == "Hello world.", (
        f"Unexpected first chunk: {chunks[0].text}"
    )
    assert chunks[1].text.strip() == "This is a test.", (
        f"Unexpected second chunk: {chunks[1].text}"
    )


# ------------------ Filter Tests ------------------


# Define a dummy filter to use with the unified loader.
class DummyFilter(BaseFilter):
    def __init__(self, name: str):
        super().__init__(name)

    @classmethod
    def from_config(cls, config: BaseFilterConfig) -> "DummyFilter":
        # Use the config name if available, otherwise default to "dummy_filter"
        return cls(name=config.name or "dummy_filter")

    def filter(self, sample: MultimodalSample) -> bool:
        return True


# Patch the filter loaders mapping and supported types for the dummy filter.
_original_filters_loaders_map = FILTERS_LOADERS_MAP.copy()
_original_filter_type = FILTER_TYPES[:]
FILTERS_LOADERS_MAP["dummy_filter"] = DummyFilter
FILTER_TYPES.append("dummy_filter")


def test_filter_from_load_postprocessor():
    """
    Verify that load_postprocessor returns a DummyFilter when given a dummy filter config.
    """
    config_args = {"type": "dummy_filter", "args": {}}
    base_config = BasePostProcessorConfig(type="dummy_filter", args=config_args)
    processor = load_postprocessor(base_config)
    assert isinstance(processor, DummyFilter), "Expected a DummyFilter instance."

    # Restore the original mappings to avoid side effects.
    FILTERS_LOADERS_MAP.clear()
    FILTERS_LOADERS_MAP.update(_original_filters_loaders_map)
    FILTER_TYPES[:] = _original_filter_type


def test_filter_process():
    """
    Test that the filter post processor correctly processes a sample.
    Two dummy filters are defined:
      - One that always accepts the sample.
      - One that always rejects the sample.
    """

    # Dummy filter that always accepts the sample.
    class DummyAcceptFilter(BaseFilter):
        def filter(self, sample: MultimodalSample) -> bool:
            return True

    # Dummy filter that always rejects the sample.
    class DummyRejectFilter(BaseFilter):
        def filter(self, sample: MultimodalSample) -> bool:
            return False

    sample = MultimodalSample(text="Sample text", modalities=[], metadata={}, id="1")

    accept_filter = DummyAcceptFilter("dummy_accept")
    accepted = accept_filter.process(sample)
    # When filter returns True, process() should return the sample wrapped in a list.
    assert accepted == [sample], (
        f"Expected sample to be kept when filter returns True, got {accepted}"
    )

    reject_filter = DummyRejectFilter("dummy_reject")
    rejected = reject_filter.process(sample)
    # When filter returns False, process() should return an empty list.
    assert rejected == [], (
        f"Expected sample to be rejected when filter returns False, got {rejected}"
    )


# ------------------ NER Tests ------------------


# Dummy LLM that always returns a fixed extraction output.
class DummyLLM:
    def __call__(self, input, config=None):
        # This output string has one entity record.
        # It uses the specified delimiters: tuple_delimiter = "<|>", record_delimiter = "##"
        # and no extra record is added.
        return '("entity"<|>HELLO WORLD<|>ORGANIZATION<|>A SAMPLE ORGANIZATION)'


def test_ner_from_config():
    """
    Verify that NERecognizer.from_config returns an instance of NERecognizer.
    """
    # Patch LLM.from_config to return our dummy LLM regardless of input.
    original_llm_from_config = LLM.from_config
    LLM.from_config = lambda cfg: DummyLLM()

    config = NERExtractorConfig(
        llm={"dummy": "dummy"},  # dummy config; our lambda ignores it
        prompt="dummy prompt",  # a simple string; PromptTemplate.from_template() will be used
        entity_types=["ORGANIZATION"],
        tuple_delimiter="<|>",
        record_delimiter="##",
        completion_delimiter="<|COMPLETE|>",
    )
    recognizer = NERecognizer.from_config(config)
    assert isinstance(recognizer, NERecognizer), "Expected NERecognizer instance."

    # Restore the original method.
    LLM.from_config = original_llm_from_config


def test_ner_process():
    """
    Test that NERecognizer.process extracts entities correctly from a sample.
    The dummy LLM always returns an output with one entity:
      ("entity"<|>HELLO WORLD<|>ORGANIZATION<|>A SAMPLE ORGANIZATION)
    which should add to the sample's metadata a list with one dictionary.
    """
    original_llm_from_config = LLM.from_config
    LLM.from_config = lambda cfg: DummyLLM()

    config = NERExtractorConfig(
        llm={"dummy": "dummy"},
        prompt="dummy prompt",
        entity_types=["ORGANIZATION"],
        tuple_delimiter="<|>",
        record_delimiter="##",
        completion_delimiter="<|COMPLETE|>",
    )
    recognizer = NERecognizer.from_config(config)

    sample = MultimodalSample(
        text="Some irrelevant text", modalities=[], metadata={}, id="1"
    )
    processed_samples = recognizer.process(sample)

    # The process() method should return a list with one sample.
    assert isinstance(processed_samples, list), "Expected process() to return a list."
    # The sample's metadata should include an 'ner' key.
    assert "ner" in sample.metadata, "Expected sample.metadata to include key 'ner'."

    ner_entities = sample.metadata["ner"]
    # We expect one entity: HELLO WORLD as an ORGANIZATION with the given description.
    assert len(ner_entities) == 1, f"Expected 1 entity, got {len(ner_entities)}."
    entity_info = ner_entities[0]
    assert entity_info.get("entity") == "HELLO WORLD", (
        f"Unexpected entity name: {entity_info.get('entity')}"
    )
    assert entity_info.get("type") == "ORGANIZATION", (
        f"Unexpected entity type: {entity_info.get('type')}"
    )
    assert entity_info.get("description") == ["A SAMPLE ORGANIZATION"], (
        f"Unexpected entity description: {entity_info.get('description')}"
    )

    # Restore the original LLM.from_config
    LLM.from_config = original_llm_from_config


# ------------- Loader Tests -------------

# ---------------------------------------------------------------------------
# Monkey-patch the tagger classes to add a minimal from_config method.
# This enables load_tagger() to instantiate them.
# ---------------------------------------------------------------------------
if not hasattr(WordsCounter, "from_config"):
    WordsCounter.from_config = classmethod(lambda cls, config: cls())
if not hasattr(ModalitiesCounter, "from_config"):
    ModalitiesCounter.from_config = classmethod(lambda cls, config: cls())
if not hasattr(LangDetector, "from_config"):
    LangDetector.from_config = classmethod(lambda cls, config: cls())


def test_tagger_from_load_tagger_words():
    """
    Verify that load_tagger returns a WordsCounter when given a words_counter config.
    """
    config = BaseTaggerConfig(type="words_counter", args={})
    tagger = load_tagger(config)
    assert isinstance(tagger, WordsCounter), "Expected a WordsCounter instance."


def test_tagger_from_load_tagger_modalities():
    """
    Verify that load_tagger returns a ModalitiesCounter when given a modalities_counter config.
    """
    config = BaseTaggerConfig(type="modalities_counter", args={})
    tagger = load_tagger(config)
    assert isinstance(tagger, ModalitiesCounter), (
        "Expected a ModalitiesCounter instance."
    )


def test_tagger_from_load_tagger_lang_detector():
    """
    Verify that load_tagger returns a LangDetector when given a lang_detector config.
    """
    config = BaseTaggerConfig(type="lang_detector", args={})
    tagger = load_tagger(config)
    assert isinstance(tagger, LangDetector), "Expected a LangDetector instance."


def test_tagger_load_invalid_type():
    """
    Verify that load_tagger raises a ValueError when given an unrecognized tagger type.
    """
    config = BaseTaggerConfig(type="unknown_tagger", args={})
    with pytest.raises(ValueError, match="Unrecognized tagger type"):
        load_tagger(config)


# ------------- Process Tests -------------


def test_tagger_process_words_counter():
    """
    Test that the WordsCounter tagger computes the word count correctly.
    The process() method should add a "word_count" metadata key to the sample.
    """
    config = BaseTaggerConfig(type="words_counter", args={})
    tagger = load_tagger(config)
    sample = MultimodalSample(
        text="Hello world, this is a test", modalities=[], metadata={}, id="1"
    )
    processed = tagger.process(sample)
    expected_count = len(sample.text.split())
    # WordsCounter's default metadata_key is set in its __init__ to 'word_count'
    assert sample.metadata.get("word_count") == expected_count, (
        f"Expected word_count {expected_count}, got {sample.metadata.get('word_count')}"
    )
    assert isinstance(processed, list), "Expected process() to return a list."


def test_tagger_process_modalities_counter():
    """
    Test that the ModalitiesCounter tagger returns the correct count of modalities.
    The process() method should add a "modalities_count" metadata key to the sample.
    """
    config = BaseTaggerConfig(type="modalities_counter", args={})
    tagger = load_tagger(config)
    sample = MultimodalSample(
        text="Some text", modalities=["img1", "img2", "video1"], metadata={}, id="2"
    )
    processed = tagger.process(sample)
    expected_count = len(sample.modalities)
    # ModalitiesCounter's default metadata_key is 'modalities_count'
    assert sample.metadata.get("modalities_count") == expected_count, (
        f"Expected modalities_count {expected_count}, got {sample.metadata.get('modalities_count')}"
    )
    assert isinstance(processed, list), "Expected process() to return a list."


def test_tagger_process_lang_detector():
    """
    Test that the LangDetector tagger detects the language of the sample text.
    The process() method should add a "lang" metadata key to the sample.
    """
    config = BaseTaggerConfig(type="lang_detector", args={})
    tagger = load_tagger(config)
    # Provide text clearly in English.
    sample = MultimodalSample(
        text="Hello world, this is an English sentence.",
        modalities=[],
        metadata={},
        id="3",
    )
    processed = tagger.process(sample)
    detected_lang = sample.metadata.get("lang")
    # langdetect typically returns "en" for English.
    assert detected_lang in [
        "en",
        "EN",
    ], f"Expected detected language 'en', got {detected_lang}"
    assert isinstance(processed, list), "Expected process() to return a list."

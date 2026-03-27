from typing import Dict, List, cast

import pytest

from mmore.process.post_processor import BasePostProcessorConfig, load_postprocessor
from mmore.process.post_processor.chunker.multimodal import (
    MultimodalChunker,
    MultimodalChunkerConfig,
)
from mmore.process.post_processor.chunker.utils import (
    _strip_separator_cell,
    _strip_table_row,
    _strip_table_text,
    chunk_table,
    chunk_table_single_row,
    detect_markdown_tables,
)
from mmore.process.post_processor.filter import FILTER_TYPES, FILTERS_LOADERS_MAP
from mmore.process.post_processor.filter.base import BaseFilter, BaseFilterConfig
from mmore.process.post_processor.ner import NERecognizer, NERExtractorConfig
from mmore.process.post_processor.tagger import load_tagger
from mmore.process.post_processor.tagger.base import BaseTaggerConfig
from mmore.process.post_processor.tagger.lang_detector import LangDetector
from mmore.process.post_processor.tagger.modalities import ModalitiesCounter
from mmore.process.post_processor.tagger.words import WordsCounter
from mmore.rag.llm import LLM, LLMConfig
from mmore.type import MultimodalRawInput, MultimodalSample


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
FILTERS_LOADERS_MAP["dummy_filter"] = DummyFilter  # pyright: ignore[reportArgumentType]
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
    LLM.from_config = lambda cfg: DummyLLM()  # pyright: ignore[reportAttributeAccessIssue]

    config = NERExtractorConfig(
        llm=LLMConfig("dummy"),  # dummy config; our lambda ignores it
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
    LLM.from_config = lambda cfg: DummyLLM()  # pyright: ignore[reportAttributeAccessIssue]

    config = NERExtractorConfig(
        llm=LLMConfig("dummy"),
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

    ner_entities: List[Dict[str, str]] = cast(
        List[Dict[str, str]], sample.metadata["ner"]
    )
    # We expect one entity: HELLO WORLD as an ORGANIZATION with the given description.
    assert len(ner_entities) == 1, f"Expected 1 entity, got {len(ner_entities)}."
    entity_info: dict[str, str] = ner_entities[0]
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
    WordsCounter.from_config = (  # pyright: ignore[reportAttributeAccessIssue]
        classmethod(lambda cls, config: cls())
    )
if not hasattr(ModalitiesCounter, "from_config"):
    ModalitiesCounter.from_config = (  # pyright: ignore[reportAttributeAccessIssue]
        classmethod(lambda cls, config: cls())
    )
if not hasattr(LangDetector, "from_config"):
    LangDetector.from_config = (  # pyright: ignore[reportAttributeAccessIssue]
        classmethod(lambda cls, config: cls())
    )


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
        text="Some text",
        modalities=[
            MultimodalRawInput(type="image", value="img1"),
            MultimodalRawInput(type="image", value="img2"),
            MultimodalRawInput(type="video", value="video1"),
        ],
        metadata={},
        id="2",
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


# ------------------ Table-Aware Chunker Tests ------------------


SIMPLE_TABLE = """\
| Name | Age | City |
|------|-----|------|
| Alice | 30 | Paris |
| Bob | 25 | London |
| Carol | 35 | Berlin |"""


LARGE_TABLE_HEADER = "| Col A | Col B | Col C |"
LARGE_TABLE_SEP_RAW = "| ------- | ------- | ------- |"
LARGE_TABLE_SEP = "| --- | --- | --- |"


def _make_long_table(num_rows: int) -> str:
    rows = [f"| val{i}_a | val{i}_b | val{i}_c |" for i in range(num_rows)]
    return LARGE_TABLE_HEADER + "\n" + LARGE_TABLE_SEP_RAW + "\n" + "\n".join(rows)


MIXED_TEXT = f"""\
This is a paragraph of regular text before the table.

{SIMPLE_TABLE}

This is a paragraph of regular text after the table."""


class TestDetectAndStripMarkdownTables:
    def test_detects_simple_table(self):
        tables = detect_markdown_tables(SIMPLE_TABLE)
        assert len(tables) == 1
        table = tables[0]
        assert table.start_index == 0
        assert len(table.body_rows) == 3
        assert "Name" in table.header
        assert "---" in table.header

    def test_detects_table_in_mixed_text(self):
        tables = detect_markdown_tables(MIXED_TEXT)
        assert len(tables) == 1
        table = tables[0]
        assert table.start_index != 0
        assert len(table.body_rows) == 3
        assert "Name" in table.header
        assert "Age" in table.header

    def test_no_tables(self):
        text = "Just some regular text.\nNo tables here.\nAnother line."
        tables = detect_markdown_tables(text)
        assert len(tables) == 0

    def test_header_only_table(self):
        text = "| A | B |\n|---|---|\n"
        tables = detect_markdown_tables(text)
        assert len(tables) == 1
        assert len(tables[0].body_rows) == 0

    def test_multiple_tables(self):
        text = (
            "| A | B |\n|---|---|\n| 1 | 2 |\n\n"
            "Some text in between.\n\n"
            "| X | Y |\n|---|---|\n| 3 | 4 |\n"
        )
        tables = detect_markdown_tables(text)
        assert len(tables) == 2

    def test_detects_table_with_empty_cells(self):
        text = "| A | B |\n|---|---|\n| | val |\n| x | |\n"
        tables = detect_markdown_tables(text)
        assert len(tables) == 1
        assert len(tables[0].body_rows) == 2
        assert "| | val |" in tables[0].body_rows[0]
        assert "| x | |" in tables[0].body_rows[1]

    def test_detects_table_with_alignment_colons(self):
        text = "| Left | Center | Right |\n| :--- | :---: | ---: |\n| a | b | c |\n"
        tables = detect_markdown_tables(text)
        assert len(tables) == 1
        assert len(tables[0].body_rows) == 1
        assert "Left" in tables[0].header

    def test_not_a_table_without_separator(self):
        text = "| looks like | a table |\n| but no | separator |\n"
        tables = detect_markdown_tables(text)
        assert len(tables) == 0

    def test_strip_row_removes_cell_padding(self):
        row = "|  Alice  |  30  |  Paris  |"
        assert _strip_table_row(row) == "| Alice | 30 | Paris |"

    def test_strip_row_ignores_non_table_line(self):
        row = "This is just text."
        assert _strip_table_row(row) == row

    def test_strip_separator_cell_all_alignments(self):
        assert _strip_separator_cell("----------") == "---"
        assert _strip_separator_cell(":----------") == ":---"
        assert _strip_separator_cell("----------:") == "---:"
        assert _strip_separator_cell(":----------:") == ":---:"
        assert _strip_separator_cell("---") == "---"
        assert _strip_separator_cell("  :---:  ") == ":---:"

    def test_strip_row_normalizes_separator_dashes(self):
        row = "| ---------- | ---- | ------- |"
        assert _strip_table_row(row) == "| --- | --- | --- |"

    def test_strip_row_preserves_alignment_colons(self):
        row = "| :---------- | :----: | -------: |"
        assert _strip_table_row(row) == "| :--- | :---: | ---: |"

    def test_strips_full_table_text(self):
        table = "|  Name  |  Age  |\n| ---------- | ---- |\n|  Alice  |  30  |"
        expected = "| Name | Age |\n| --- | --- |\n| Alice | 30 |"
        assert _strip_table_text(table) == expected

    def test_strips_table_text_preserves_alignment(self):
        table = (
            "| Left | Center | Right |\n"
            "| :---------- | :--------: | ----------: |\n"
            "| a | b | c |"
        )
        result = _strip_table_text(table)
        lines = result.split("\n")
        assert lines[1] == "| :--- | :---: | ---: |"


class TestChunkTableSingleRow:
    def _simple_counter(self, text: str) -> int:
        return len(text.split())

    def test_one_chunk_per_row(self):
        tables = detect_markdown_tables(SIMPLE_TABLE)
        assert len(tables) == 1
        chunks = chunk_table_single_row(tables[0], count_tokens=self._simple_counter)
        assert len(chunks) == 3
        assert "Alice" in chunks[0].text
        assert "Bob" in chunks[1].text
        assert "Carol" in chunks[2].text

    def test_header_prepended_to_each_chunk(self):
        tables = detect_markdown_tables(SIMPLE_TABLE)
        chunks = chunk_table_single_row(tables[0], count_tokens=self._simple_counter)
        for chunk in chunks:
            assert chunk.text.startswith("| Name | Age | City |")
            assert "---" in chunk.text

    def test_header_only_table_returns_single_chunk(self):
        text = "| A | B |\n|---|---|\n"
        tables = detect_markdown_tables(text)
        assert len(tables) == 1
        chunks = chunk_table_single_row(tables[0], count_tokens=self._simple_counter)
        assert len(chunks) == 1
        assert chunks[0].text == "| A | B |\n| --- | --- |"

    def test_chunk_indices_in_range(self):
        tables = detect_markdown_tables(SIMPLE_TABLE)
        chunks = chunk_table_single_row(tables[0], count_tokens=self._simple_counter)
        for chunk in chunks:
            assert chunk.start_index >= tables[0].start_index
            assert chunk.end_index <= tables[0].end_index


class TestChunkTable:
    def _simple_counter(self, text: str) -> int:
        """Simple word-based token counter for tests."""
        return len(text.split())

    def test_small_table_single_chunk(self):
        tables = detect_markdown_tables(SIMPLE_TABLE)
        assert len(tables) == 1
        chunks = chunk_table(
            tables[0], max_tokens=1000, count_tokens=self._simple_counter
        )
        assert len(chunks) == 1
        assert "Alice" in chunks[0].text
        assert "Carol" in chunks[0].text

    def test_large_table_split_preserves_headers(self):
        big_table = _make_long_table(50)
        tables = detect_markdown_tables(big_table)
        assert len(tables) == 1
        # Small max_tokens to force splitting
        chunks = chunk_table(
            tables[0], max_tokens=30, count_tokens=self._simple_counter
        )
        assert len(chunks) > 1
        # Every chunk must start with the header
        for chunk in chunks:
            assert chunk.text.startswith(LARGE_TABLE_HEADER)
            assert LARGE_TABLE_SEP in chunk.text

    def test_chunk_indices_in_range(self):
        big_table = _make_long_table(20)
        tables = detect_markdown_tables(big_table)
        chunks = chunk_table(
            tables[0], max_tokens=30, count_tokens=self._simple_counter
        )
        for chunk in chunks:
            assert chunk.start_index >= tables[0].start_index
            assert chunk.end_index <= tables[0].end_index


class TestMultimodalChunkerTableHandling:
    def _make_chunker(
        self, table_handling: str = "single_row", chunk_size: int = 20
    ) -> MultimodalChunker:
        config = MultimodalChunkerConfig(
            chunking_strategy="sentence",
            text_chunker_config={"chunk_size": chunk_size, "chunk_overlap": 0},
            table_handling=table_handling,
        )
        return MultimodalChunker.from_config(config)

    def test_no_tables_unchanged_behavior(self):
        chunker = self._make_chunker()
        sample = MultimodalSample(
            text="Hello world. This is a test.", modalities=[], metadata={}
        )
        chunks = chunker.chunk(sample)
        assert len(chunks) >= 1
        for c in chunks:
            assert "is_table_chunk" not in c.metadata

    def test_large_table_split_with_headers(self):
        big_table = _make_long_table(50)
        chunker = self._make_chunker(table_handling="multi_rows", chunk_size=30)
        sample = MultimodalSample(text=big_table, modalities=[], metadata={})
        chunks = chunker.chunk(sample)
        assert len(chunks) > 1
        # Every chunk has the header prepended
        for c in chunks:
            assert c.metadata.get("is_table_chunk") is True
            assert c.text.startswith(LARGE_TABLE_HEADER)

    def test_table_handling_single_row(self):
        chunker = self._make_chunker(table_handling="single_row", chunk_size=512)
        sample = MultimodalSample(text=SIMPLE_TABLE, modalities=[], metadata={})
        chunks = chunker.chunk(sample)
        assert len(chunks) == 3
        # Every chunk has the header prepended
        for c in chunks:
            assert c.metadata.get("is_table_chunk") is True
            assert c.text.startswith("| Name | Age | City |")
        assert "Alice" in chunks[0].text
        assert "Bob" in chunks[1].text
        assert "Carol" in chunks[2].text

    def test_mixed_content_chunking(self):
        chunker = self._make_chunker(chunk_size=512)
        sample = MultimodalSample(text=MIXED_TEXT, modalities=[], metadata={})
        chunks = chunker.chunk(sample)
        assert len(chunks) >= 2  # at least text + table
        table_chunks = [c for c in chunks if c.metadata.get("is_table_chunk")]
        non_table_chunks = [c for c in chunks if not c.metadata.get("is_table_chunk")]
        assert len(table_chunks) >= 1
        assert len(non_table_chunks) >= 1

    def test_table_handling_none(self):
        chunker = self._make_chunker(table_handling="none", chunk_size=512)
        sample = MultimodalSample(text=SIMPLE_TABLE, modalities=[], metadata={})
        chunks = chunker.chunk(sample)
        for c in chunks:
            assert "is_table_chunk" not in c.metadata

    def test_table_handling_keep_whole(self):
        big_table = _make_long_table(50)
        chunker = self._make_chunker(table_handling="keep_whole", chunk_size=30)
        sample = MultimodalSample(text=big_table, modalities=[], metadata={})
        chunks = chunker.chunk(sample)
        # Should be a single chunk even though it exceeds chunk_size
        assert len(chunks) == 1
        assert chunks[0].metadata.get("is_table_chunk") is True

    def test_invalid_table_handling_mode(self):
        with pytest.raises(ValueError, match="Invalid table_handling mode"):
            MultimodalChunker(
                text_chunker=MultimodalChunker.from_config(
                    MultimodalChunkerConfig()
                ).text_chunker,
                table_handling="invalid_mode",
            )

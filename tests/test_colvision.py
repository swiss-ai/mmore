import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from mmore.colvision.milvuscolvision import MilvusColvisionManager
from mmore.colvision.model_utils import _patched_key_mapping, resolve_model_classes
from mmore.colvision.retriever import (
    ColVisionRetriever,
    ColVisionRetrieverConfig,
    load_text_mapping,
)
from mmore.colvision.run_index import index
from mmore.colvision.run_process import (
    ColVisionEmbedder,
    PDFConverter,
    process_single_pdf,
)

"""
If you get an error when running tests with pytest, run tests with: PYTHONPATH=$(pwd) pytest tests/test_colvision.py.
This is required because the project follows a "src" layout, and setting PYTHONPATH ensures Python correctly resolves imports like "from mmore.colvision...".
"""

SAMPLES_DIR = "examples/sample_data/"


# ------------------ Model Resolution Tests ------------------
def test_resolve_model_classes_colpali():
    """Test that ColPali model names resolve to ColPali classes."""
    model_cls, proc_cls = resolve_model_classes("vidore/colpali-v1.3")
    assert model_cls.__name__ == "ColPali"
    assert proc_cls.__name__ == "ColPaliProcessor"


def test_resolve_model_classes_colqwen2():
    """Test that ColQwen2 model names resolve to ColQwen2 classes."""
    model_cls, proc_cls = resolve_model_classes("vidore/colqwen2-v1.0")
    assert model_cls.__name__ == "ColQwen2"
    assert proc_cls.__name__ == "ColQwen2Processor"


def test_resolve_model_classes_colqwen2_5():
    """Test that ColQwen2.5 model names resolve to ColQwen2_5 classes."""
    model_cls, proc_cls = resolve_model_classes("vidore/colqwen2.5-v0.2")
    assert model_cls.__name__ == "ColQwen2_5"
    assert proc_cls.__name__ == "ColQwen2_5_Processor"


def test_resolve_model_classes_colqwen3():
    """Test that ColQwen3 model names resolve to ColQwen3 classes."""
    model_cls, proc_cls = resolve_model_classes("vidore/colqwen3-v0.1")
    assert model_cls.__name__ == "ColQwen3"
    assert proc_cls.__name__ == "ColQwen3Processor"


def test_resolve_model_classes_colgemma():
    """Test that ColGemma model names resolve to ColGemma3 classes."""
    model_cls, proc_cls = resolve_model_classes("Cognitive-Lab/ColNetraEmbed")
    assert model_cls.__name__ == "ColGemma3"
    assert proc_cls.__name__ == "ColGemmaProcessor3"


def test_resolve_model_classes_unknown_raises():
    """Test that unknown model names raise ValueError."""
    with pytest.raises(ValueError, match="Unknown model"):
        resolve_model_classes("some/unknown-model")


# ------------------ _patched_key_mapping Tests ------------------


def test_patched_key_mapping_colgemma3_adds_vision_tower_rule():
    """
    Regression: ColGemma3 (Cognitive-Lab/ColNetraEmbed) was serialised under
    transformers 4.x with vision_tower.vision_model.* keys.  transformers 5.x
    flattened the layout to vision_tower.*.  colpali-engine's mapping omits
    this rule, leaving the vision encoder randomly initialised.

    _patched_key_mapping must add vision_tower.vision_model.* → vision_tower.*
    so the checkpoint keys are remapped correctly at load time.
    """
    import colpali_engine.models as colpali_models

    mapping = _patched_key_mapping(colpali_models.ColGemma3)

    assert mapping is not None, (
        "_patched_key_mapping must return a mapping for ColGemma3 "
        "(vision tower gap not yet fixed upstream)"
    )
    vision_rules = {k: v for k, v in mapping.items() if "vision_tower" in k}
    assert vision_rules, (
        "Mapping must contain at least one rule covering vision_tower keys; "
        f"got keys: {list(mapping.keys())}"
    )
    # The rule must remap vision_tower.vision_model.* → vision_tower.*
    import re

    sample_key = "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.weight"
    remapped = sample_key
    for pattern, replacement in mapping.items():
        remapped = re.sub(pattern, replacement, remapped)
    assert remapped == "vision_tower.encoder.layers.0.self_attn.q_proj.weight", (
        f"vision_tower key was not correctly remapped; got: {remapped!r}"
    )


def test_patched_key_mapping_colgemma3_self_disables():
    """
    If colpali-engine adds a vision_tower rule natively, _patched_key_mapping
    must not add a duplicate and may return None (no extra patch needed).
    We verify the guard: when a vision_tower key already exists in the mapping,
    our patch is skipped.
    """
    import colpali_engine.models as colpali_models

    # Temporarily inject a vision_tower rule into ColGemma3's mapping.
    original = colpali_models.ColGemma3._checkpoint_conversion_mapping.copy()
    colpali_models.ColGemma3._checkpoint_conversion_mapping = {
        **original,
        r"^vision_tower\.vision_model\.": "vision_tower.",
    }
    try:
        mapping = _patched_key_mapping(colpali_models.ColGemma3)
        if mapping is not None:
            vision_rules = [k for k in mapping if "vision_tower" in k]
            assert len(vision_rules) == 1, (
                "Only one vision_tower rule must be present when upstream already provides it; "
                f"got: {vision_rules}"
            )
    finally:
        colpali_models.ColGemma3._checkpoint_conversion_mapping = original


def test_patched_key_mapping_colqwen2_5_still_works():
    """
    Regression guard: the ColQwen2/2.5 embed_tokens/norm patch must still be
    applied after the ColGemma3 patch was added to _patched_key_mapping.
    """
    import re

    import colpali_engine.models as colpali_models

    mapping = _patched_key_mapping(colpali_models.ColQwen2_5)

    assert mapping is not None, (
        "_patched_key_mapping must return a mapping for ColQwen2_5"
    )
    assert r"^model\.embed_tokens" in mapping, "embed_tokens rule must be present"
    assert r"^model\.norm" in mapping, "norm rule must be present"

    for src, expected in [
        ("model.embed_tokens.weight", "language_model.embed_tokens.weight"),
        ("model.norm.weight", "language_model.norm.weight"),
    ]:
        remapped = src
        for pattern, replacement in mapping.items():
            remapped = re.sub(pattern, replacement, remapped)
        assert remapped == expected, (
            f"Key {src!r} was not correctly remapped; got: {remapped!r}"
        )


def test_patched_key_mapping_skips_on_transformers_4x(monkeypatch):
    """
    Regression: the `key_mapping` from_pretrained argument only exists on
    transformers 5.x. On the colvision-legacy stack (transformers 4.x, ColPali
    v1.3), ColPali is PaliGemma-based, so the Gemma vision-tower patch would
    otherwise fire and pass an unsupported `key_mapping` to from_pretrained.

    _patched_key_mapping must return None whenever transformers < 5.3 so that
    no key_mapping is ever injected on the legacy stack.
    """
    import colpali_engine.models as colpali_models
    import transformers

    monkeypatch.setattr(transformers, "__version__", "4.44.2")
    for cls_name in ("ColPali", "ColGemma3", "ColQwen2_5"):
        cls = getattr(colpali_models, cls_name)
        assert _patched_key_mapping(cls) is None, (
            f"{cls_name} must not be patched on transformers 4.x"
        )


# ------------------ PDFConverter Tests ------------------
def test_pdf_converter_convert_to_pngs():
    """Test that PDFConverter correctly converts PDF pages to PNG images."""
    sample_file = os.path.join(SAMPLES_DIR, "pdf", "calendar.pdf")
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"

    converter = PDFConverter(dpi=200)
    try:
        png_paths = converter.convert_to_pngs(Path(sample_file))
        assert len(png_paths) > 0, "Should generate at least one PNG file"
        assert all(p.exists() for p in png_paths), (
            "All generated PNG files should exist"
        )
        assert all(p.suffix == ".png" for p in png_paths), (
            "All generated files should have .png extension"
        )
    finally:
        converter.cleanup()


# ------------------ ColVisionEmbedder Tests ------------------
def test_colvision_embedder_embed_images():
    """Test that embed_images correctly processes images in batches and returns embeddings with expected shape."""
    from unittest.mock import MagicMock

    from mmore.colvision import run_process

    # Create temporary image files
    with tempfile.TemporaryDirectory() as tmpdir:
        image_paths = []
        for i in range(5):  # 5 images to test batch processing with batch_size=2
            img = Image.new("RGB", (100, 100), color=(i * 10, i * 20, i * 30))
            img_path = Path(tmpdir) / f"test_image_{i}.png"
            img.save(img_path)
            image_paths.append(str(img_path))

        # Mock the model loader to bypass actual HuggingFace download
        original_loader = run_process.load_model_and_processor

        embedding_dim = 128
        mock_embeddings = [
            torch.randn(embedding_dim, dtype=torch.bfloat16) for _ in range(5)
        ]

        # Track batch calls
        batch_calls = []

        def mock_process_images(images):
            """Mock processor.process_images that returns processed batch."""
            batch_size = len(images)
            batch_calls.append(batch_size)
            return {
                "pixel_values": torch.randn(
                    batch_size, 3, 224, 224, dtype=torch.bfloat16
                )
            }

        def mock_model_forward(**kwargs):
            """Mock model forward that returns embeddings."""
            batch_size = kwargs["pixel_values"].shape[0]
            return torch.stack(mock_embeddings[:batch_size])

        # Use MagicMock for the model
        mock_model = MagicMock()
        mock_model.device = torch.device("cpu")
        mock_model.eval.return_value = mock_model
        mock_model.side_effect = mock_model_forward

        # Use MagicMock for the processor
        mock_processor_instance = MagicMock()
        mock_processor_instance.process_images.side_effect = mock_process_images

        run_process.load_model_and_processor = lambda name, device: (
            mock_model,
            mock_processor_instance,
        )

        try:
            embedder = ColVisionEmbedder(model_name="vidore/colpali-v1.3", device="cpu")
            embedder.processor = mock_processor_instance

            # Test with batch_size=2 (should create 3 batches: 2, 2, 1)
            embeddings = embedder.embed_images(image_paths, batch_size=2)

            # Verify correct number of embeddings
            assert len(embeddings) == 5, (
                f"Should return 5 embeddings, got {len(embeddings)}"
            )

            # Verify batch processing occurred (should have 3 batches: 2, 2, 1)
            assert len(batch_calls) == 3, (
                f"Should process in 3 batches, got {len(batch_calls)}"
            )
            assert batch_calls == [2, 2, 1], (
                f"Batch sizes should be [2, 2, 1], got {batch_calls}"
            )

            # Verify embedding shape and type
            for i, emb in enumerate(embeddings):
                assert isinstance(emb, np.ndarray), (
                    f"Embedding {i} should be numpy array"
                )
                assert emb.ndim == 1, (
                    f"Embedding {i} should be 1D, got shape {emb.shape}"
                )
                assert emb.shape[0] == embedding_dim, (
                    f"Embedding {i} should have dimension {embedding_dim}, got {emb.shape[0]}"
                )
                assert emb.dtype == np.float32, (
                    f"Embedding {i} should be float32, got {emb.dtype}"
                )

        finally:
            run_process.load_model_and_processor = original_loader


def test_colvision_embedder_embed_images_invalid_input():
    """Test that embed_images handles invalid image paths correctly."""
    from unittest.mock import MagicMock

    from mmore.colvision import run_process

    # Mock the model loader to bypass actual HuggingFace download
    original_loader = run_process.load_model_and_processor

    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")
    mock_model.eval.return_value = mock_model

    mock_processor_instance = MagicMock()
    mock_processor_instance.process_images.return_value = {
        "pixel_values": torch.empty(0, 3, 224, 224)
    }

    run_process.load_model_and_processor = lambda name, device: (
        mock_model,
        mock_processor_instance,
    )

    try:
        embedder = ColVisionEmbedder(model_name="vidore/colpali-v1.3", device="cpu")
        embedder.processor = mock_processor_instance

        # Test with non-existent file path
        with pytest.raises((FileNotFoundError, OSError)):
            embedder.embed_images(["/nonexistent/path/image.png"], batch_size=5)

        # Test with corrupted image file
        with tempfile.TemporaryDirectory() as tmpdir:
            corrupted_path = Path(tmpdir) / "corrupted.png"
            with open(corrupted_path, "wb") as f:
                f.write(b"This is not a valid image file")

            with pytest.raises((OSError, IOError)):
                embedder.embed_images([str(corrupted_path)], batch_size=5)

    finally:
        run_process.load_model_and_processor = original_loader


# ------------------ Process Single PDF Tests ------------------
def test_process_single_pdf_success():
    """Test that process_single_pdf correctly processes a PDF file."""
    from unittest.mock import MagicMock

    sample_file = os.path.join(SAMPLES_DIR, "pdf", "calendar.pdf")
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"

    converter = PDFConverter()
    converter.convert_to_pngs = lambda pdf_file: [
        Path("page_1.png"),
        Path("page_2.png"),
    ]

    # Override embed_images to bypass model loading
    mock_embedder = MagicMock()
    mock_embedder.embed_images.return_value = [
        np.array([0.1] * 128, dtype=np.float32),
        np.array([0.2] * 128, dtype=np.float32),
    ]

    # Override fitz.open to return a mock document
    from mmore.colvision import run_process

    original_fitz_open = run_process.fitz.open

    mock_page1 = MagicMock()
    mock_page1.get_text.return_value = "Page 1 text"

    mock_page2 = MagicMock()
    mock_page2.get_text.return_value = "Page 2 text"

    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 2
    mock_doc.__getitem__.side_effect = lambda idx: [mock_page1, mock_page2][idx]
    mock_doc.__enter__.return_value = mock_doc
    mock_doc.__exit__.return_value = None

    run_process.fitz.open = MagicMock(return_value=mock_doc)

    try:
        page_records, text_records = process_single_pdf(
            Path(sample_file), mock_embedder, converter
        )

        assert len(page_records) == 2, "Should process 2 pages"
        assert len(text_records) == 2, "Should extract text from 2 pages"
        assert page_records[0]["pdf_path"] == str(Path(sample_file)), (
            "PDF path should be correct"
        )
        assert page_records[0]["page_number"] == 1, "Page number should be 1"
        assert "embedding" in page_records[0], "Should include embedding"
        assert text_records[0]["text"] == "Page 1 text", (
            "Text should be extracted correctly"
        )
    finally:
        converter.cleanup()
        run_process.fitz.open = original_fitz_open


# ------------------ MilvusColvisionManager Tests ------------------
def test_milvus_colvision_manager_init_error():
    """Test that MilvusColvisionManager raises error when collection doesn't exist."""
    from mmore.colvision import milvuscolvision

    original_milvus_client = milvuscolvision.MilvusClient

    mock_client_instance = type(
        "obj",
        (object,),
        {
            "has_collection": lambda self, name: False,
        },
    )()

    milvuscolvision.MilvusClient = lambda *args, **kwargs: mock_client_instance

    try:
        with pytest.raises(ValueError, match="does not exist"):
            MilvusColvisionManager(
                db_path="./test_milvus",
                collection_name="test_collection",
                dim=128,
                create_collection=False,
            )
    finally:
        milvuscolvision.MilvusClient = original_milvus_client


def test_milvus_colvision_manager_insert_from_dataframe():
    """Test that MilvusColvisionManager correctly inserts data from DataFrame."""
    from mmore.colvision import milvuscolvision

    original_milvus_client = milvuscolvision.MilvusClient

    insert_calls = []
    mock_client_instance = type(
        "obj",
        (object,),
        {
            "has_collection": lambda self, name: True,
            "load_collection": lambda self, name: None,
            "insert": lambda self, collection, data: insert_calls.append(data),
        },
    )()

    milvuscolvision.MilvusClient = lambda *args, **kwargs: mock_client_instance

    try:
        manager = MilvusColvisionManager(
            db_path="./test_milvus",
            collection_name="test_collection",
            dim=128,
            create_collection=False,
        )

        test_df = pd.DataFrame(
            {
                "pdf_path": ["test1.pdf", "test2.pdf"],
                "page_number": [1, 2],
                "embedding": [
                    np.array([0.1] * 128, dtype=np.float32),
                    np.array([0.2] * 128, dtype=np.float32),
                ],
            }
        )

        manager.insert_from_dataframe(test_df, batch_size=1)

        assert len(insert_calls) == 2, (
            "Should insert 2 batches for 2 rows with batch_size=1"
        )
    finally:
        milvuscolvision.MilvusClient = original_milvus_client


def test_milvus_colvision_manager_search_embeddings_search_error():
    """Test that search_embeddings handles search errors correctly."""
    from mmore.colvision import milvuscolvision

    original_milvus_client = milvuscolvision.MilvusClient

    def mock_search_error(self, **kwargs):
        raise Exception("Search failed")

    mock_client_instance = type(
        "obj",
        (object,),
        {
            "has_collection": lambda self, name: True,
            "load_collection": lambda self, name: None,
            "search": mock_search_error,
        },
    )()

    milvuscolvision.MilvusClient = lambda *args, **kwargs: mock_client_instance

    try:
        manager = MilvusColvisionManager(
            db_path="./test_milvus",
            collection_name="test_collection",
            dim=128,
            create_collection=False,
        )

        query_embedding = np.array([[0.1] * 128], dtype=np.float32)

        # Should raise the exception
        with pytest.raises(Exception, match="Search failed"):
            manager.search_embeddings(query_embedding, top_k=3)

    finally:
        milvuscolvision.MilvusClient = original_milvus_client


# ------------------ Index Function Tests ------------------
def test_index_function():
    """Test that index function loads config and indexes data correctly."""
    from mmore.colvision import run_index

    original_load_config = run_index.load_config
    original_read_parquet = run_index.pd.read_parquet
    original_manager_class = run_index.MilvusColvisionManager

    mock_config = type(
        "obj",
        (object,),
        {
            "parquet_path": "test.parquet",
            "milvus": type(
                "obj",
                (object,),
                {
                    "db_path": "./test_milvus",
                    "collection_name": "test_collection",
                    "metric_type": "IP",
                    "create_collection": True,
                },
            )(),
        },
    )()

    mock_df = pd.DataFrame(
        {
            "pdf_path": ["test1.pdf"],
            "page_number": [1],
            "embedding": [np.array([0.1] * 128, dtype=np.float32)],
        }
    )

    insert_called = []
    create_index_called = []

    mock_manager = type(
        "obj",
        (object,),
        {
            "insert_from_dataframe": lambda self, df: insert_called.append(df),
            "create_index": lambda self: create_index_called.append(True),
        },
    )()

    run_index.load_config = lambda *args, **kwargs: mock_config
    run_index.pd.read_parquet = lambda *args, **kwargs: mock_df
    run_index.MilvusColvisionManager = lambda *args, **kwargs: mock_manager

    try:
        index("test_config.yml")

        assert len(insert_called) > 0, "Should call insert_from_dataframe"
        assert len(create_index_called) > 0, "Should call create_index"
    finally:
        run_index.load_config = original_load_config
        run_index.pd.read_parquet = original_read_parquet
        run_index.MilvusColvisionManager = original_manager_class


# ------------------ ColVisionRetriever Tests ------------------
def test_colvision_retriever_get_relevant_documents_with_text_map():
    """Test that _get_relevant_documents correctly integrates text content from text_map."""
    from mmore.colvision import milvuscolvision, retriever

    original_milvus_client = milvuscolvision.MilvusClient
    original_embed_queries = retriever.embed_queries

    # Mock model and processor
    mock_model = type("obj", (object,), {"device": torch.device("cpu")})()
    mock_processor = type("obj", (object,), {})()

    # Mock search results
    mock_search_results = [
        {
            "pdf_path": "test1.pdf",
            "page_number": 1,
            "score": 0.95,
            "rank": 1,
        },
        {
            "pdf_path": "test1.pdf",
            "page_number": 2,
            "score": 0.85,
            "rank": 2,
        },
    ]

    # Mock text mapping
    text_map = {
        ("test1.pdf", 1): "This is the text content from page 1",
        ("test1.pdf", 2): "This is the text content from page 2",
    }

    # Mock manager - inherit from MilvusColvisionManager to pass Pydantic validation
    class MockMilvusColvisionManager(MilvusColvisionManager):
        def __init__(self, *args, **kwargs):
            # Skip parent initialization to avoid MilvusClient setup
            # Set minimal required attributes
            self.uri = "./test_milvus"
            self.collection_name = "test_collection"
            self.dim = 128
            self.metric_type = "IP"
            self.client = None

        def search_embeddings(self, query_embeddings, top_k=3, max_workers=4):
            return mock_search_results[:top_k]

    mock_manager = MockMilvusColvisionManager()

    # Mock embed_queries
    def mock_embed_queries(texts, model, processor):
        return [np.array([0.1] * 128, dtype=np.float32)]

    milvuscolvision.MilvusClient = lambda *args, **kwargs: type(
        "obj",
        (object,),
        {
            "has_collection": lambda self, name: True,
            "load_collection": lambda self, name: None,
        },
    )()
    retriever.embed_queries = mock_embed_queries

    try:
        config = ColVisionRetrieverConfig(
            db_path="./test_milvus",
            collection_name="test_collection",
            model_name="vidore/colpali-v1.3",
            top_k=2,
        )

        retriever_instance = ColVisionRetriever(
            model=mock_model,
            processor=mock_processor,
            manager=mock_manager,
            config=config,
            text_map=text_map,
        )

        # Test retrieval
        documents = retriever_instance._get_relevant_documents("test query")

        # Verify results
        assert len(documents) == 2, f"Should return 2 documents, got {len(documents)}"

        # Verify text content integration
        assert documents[0].page_content == "This is the text content from page 1", (
            "First document should have correct text content"
        )
        assert documents[1].page_content == "This is the text content from page 2", (
            "Second document should have correct text content"
        )

        # Verify metadata includes pdf_name
        assert documents[0].metadata["pdf_name"] == "test1.pdf", (
            "Document should have pdf_name in metadata"
        )
        assert documents[0].metadata["pdf_path"] == "test1.pdf", (
            "Document should have pdf_path in metadata"
        )

    finally:
        milvuscolvision.MilvusClient = original_milvus_client
        retriever.embed_queries = original_embed_queries


# ------------------ Load Text Mapping Tests ------------------
def test_load_text_mapping():
    """Test that load_text_mapping correctly loads text from parquet file."""
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
        test_df = pd.DataFrame(
            {
                "pdf_path": ["test1.pdf", "test2.pdf"],
                "page_number": [1, 2],
                "text": ["Page 1 text", "Page 2 text"],
            }
        )
        test_df.to_parquet(tmp_file.name, index=False)

        try:
            text_map = load_text_mapping(tmp_file.name)

            assert text_map is not None, "Text map should not be None"
            assert ("test1.pdf", 1) in text_map, "Should contain first page mapping"
            assert text_map[("test1.pdf", 1)] == "Page 1 text", "Text should match"
        finally:
            os.unlink(tmp_file.name)


def test_load_text_mapping_not_found():
    """Test that load_text_mapping handles missing file gracefully."""
    text_map = load_text_mapping("nonexistent.parquet")
    assert text_map is None, "Should return None when file doesn't exist"


# ------------------ Regression: rerank_page filter syntax ------------------
def test_rerank_page_filter_uses_fstring_not_dollar_syntax():
    """
    Regression test for the pymilvus $variable filter bug.

    pymilvus does not support the "$variable" syntax in filters (this is not MongoDB).
    The params= kwarg was silently ignored, the invalid filter raised an exception
    swallowed by the try/except → score=None → page excluded → context:[].
    The fix interpolates the values directly via f-string.
    """
    from mmore.colvision import milvuscolvision

    original_milvus_client = milvuscolvision.MilvusClient

    dim = 4
    pdf_path = "docs/report.pdf"
    page_number = 3
    captured_filters = []

    def mock_search(self, **kwargs):
        return [[{"entity": {"pdf_path": pdf_path, "page_number": page_number}}]]

    def mock_query(self, **kwargs):
        captured_filters.append(kwargs.get("filter", ""))
        return [
            {"embedding": [0.1] * dim, "pdf_path": pdf_path},
            {"embedding": [0.2] * dim, "pdf_path": pdf_path},
        ]

    mock_client = type(
        "MockClient",
        (object,),
        {
            "has_collection": lambda self, name: True,
            "load_collection": lambda self, name: None,
            "search": mock_search,
            "query": mock_query,
        },
    )()

    milvuscolvision.MilvusClient = lambda *args, **kwargs: mock_client

    try:
        manager = MilvusColvisionManager(
            db_path="./test_milvus",
            collection_name="test_collection",
            dim=dim,
            create_collection=False,
        )

        query_embedding = np.array([[0.1] * dim], dtype=np.float32)
        results = manager.search_embeddings(query_embedding, top_k=1)

        assert len(captured_filters) == 1, "query() must be called once"
        used_filter = captured_filters[0]

        assert "$pdf_path" not in used_filter, (
            "The filter must not use $variable — pymilvus silently ignores params="
        )
        assert "$page_number" not in used_filter, (
            "The filter must not use $variable — pymilvus silently ignores params="
        )
        assert pdf_path in used_filter, (
            "The pdf_path value must be interpolated into the filter"
        )
        assert str(page_number) in used_filter, (
            "The page_number value must be interpolated into the filter"
        )

        # The symptom of the bug was context:[] — check that the results are not empty
        assert len(results) == 1, "Reranking must return a result, not an empty list"
        assert results[0]["pdf_path"] == pdf_path
        assert results[0]["page_number"] == page_number

        # Secondary regression: the score must be a JSON-serializable Python float.
        # np.float32 crashes run_retriever.save_results() (json.dump) → the whole
        # retrieve subprocess fails on the first result (ndcg=None on the benchmark side).
        import json

        assert isinstance(results[0]["score"], float), (
            f"score must be a Python float, not {type(results[0]['score']).__name__}"
        )
        json.dumps(results)  # must not raise TypeError

    finally:
        milvuscolvision.MilvusClient = original_milvus_client

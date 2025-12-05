import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from mmore.colpali.milvuscolpali import MilvusColpaliManager
from mmore.colpali.retriever import (
    ColPaliRetriever,
    ColPaliRetrieverConfig,
    load_text_mapping,
)
from mmore.colpali.run_index import index
from mmore.colpali.run_process import ColPaliEmbedder, PDFConverter, process_single_pdf

"""
If you get an error when running tests with pytest, run tests with: PYTHONPATH=$(pwd) pytest tests/test_colpali.py.
This is required because the project follows a "src" layout, and setting PYTHONPATH ensures Python correctly resolves imports like "from mmore.colpali...".
"""

SAMPLES_DIR = "examples/sample_data/"


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


# ------------------ ColPaliEmbedder Tests ------------------
def test_colpali_embedder_embed_images():
    """Test that embed_images correctly processes images in batches and returns embeddings with expected shape."""
    from mmore.colpali import run_process

    # Create temporary image files
    with tempfile.TemporaryDirectory() as tmpdir:
        image_paths = []
        for i in range(5):  # 5 images to test batch processing with batch_size=2
            img = Image.new("RGB", (100, 100), color=(i * 10, i * 20, i * 30))
            img_path = Path(tmpdir) / f"test_image_{i}.png"
            img.save(img_path)
            image_paths.append(str(img_path))

        # Mock the model and processor
        original_colpali = run_process.ColPali.from_pretrained
        original_processor = run_process.ColPaliProcessor.from_pretrained

        embedding_dim = 128
        mock_embeddings = [
            torch.randn(embedding_dim, dtype=torch.bfloat16) for _ in range(5)
        ]

        # Track batch calls
        batch_calls = []

        def mock_process_images(self, images):
            """Mock processor.process_images that returns processed batch."""
            batch_size = len(images)
            batch_calls.append(batch_size)
            return {
                "pixel_values": torch.randn(
                    batch_size, 3, 224, 224, dtype=torch.bfloat16
                )
            }

        def mock_model_forward(self, **kwargs):
            """Mock model forward that returns embeddings."""
            batch_size = kwargs["pixel_values"].shape[0]
            return torch.stack(mock_embeddings[:batch_size])

        mock_model = type(
            "obj",
            (object,),
            {
                "device": torch.device("cpu"),
                "eval": lambda self: self,
                "__call__": mock_model_forward,
            },
        )()

        mock_processor_instance = type(
            "obj",
            (object,),
            {
                "process_images": mock_process_images,
            },
        )()

        run_process.ColPali.from_pretrained = lambda *args, **kwargs: mock_model
        run_process.ColPaliProcessor.from_pretrained = (
            lambda *args, **kwargs: mock_processor_instance
        )

        try:
            embedder = ColPaliEmbedder(model_name="vidore/colpali-v1.3", device="cpu")
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
            run_process.ColPali.from_pretrained = original_colpali
            run_process.ColPaliProcessor.from_pretrained = original_processor


def test_colpali_embedder_embed_images_invalid_input():
    """Test that embed_images handles invalid image paths correctly."""
    from mmore.colpali import run_process

    # Mock the model and processor
    original_colpali = run_process.ColPali.from_pretrained
    original_processor = run_process.ColPaliProcessor.from_pretrained

    mock_model = type(
        "obj",
        (object,),
        {
            "device": torch.device("cpu"),
            "eval": lambda self: self,
        },
    )()

    mock_processor_instance = type(
        "obj",
        (object,),
        {
            "process_images": lambda x: {"pixel_values": torch.empty(0, 3, 224, 224)},
        },
    )()

    run_process.ColPali.from_pretrained = lambda *args, **kwargs: mock_model
    run_process.ColPaliProcessor.from_pretrained = (
        lambda *args, **kwargs: mock_processor_instance
    )

    try:
        embedder = ColPaliEmbedder(model_name="vidore/colpali-v1.3", device="cpu")
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
        run_process.ColPali.from_pretrained = original_colpali
        run_process.ColPaliProcessor.from_pretrained = original_processor


# ------------------ Process Single PDF Tests ------------------
def test_process_single_pdf_success():
    """Test that process_single_pdf correctly processes a PDF file."""
    sample_file = os.path.join(SAMPLES_DIR, "pdf", "calendar.pdf")
    assert os.path.exists(sample_file), f"Sample file not found: {sample_file}"

    converter = PDFConverter()
    converter.convert_to_pngs = lambda pdf_path: [
        Path("page_1.png"),
        Path("page_2.png"),
    ]

    # Override embed_images to bypass model loading
    mock_embedder = type(
        "obj",
        (object,),
        {
            "embed_images": lambda self, paths, batch_size=5: [
                np.array([0.1] * 128, dtype=np.float32),
                np.array([0.2] * 128, dtype=np.float32),
            ]
        },
    )()

    # Override fitz.open to return a mock document
    from mmore.colpali import run_process

    original_fitz_open = run_process.fitz.open

    mock_page1 = type("obj", (object,), {"get_text": lambda self: "Page 1 text"})()
    mock_page2 = type("obj", (object,), {"get_text": lambda self: "Page 2 text"})()
    mock_doc = type(
        "obj",
        (object,),
        {
            "__len__": lambda self: 2,
            "__getitem__": lambda self, idx: [mock_page1, mock_page2][idx],
            "close": lambda self: None,
            "__enter__": lambda self: self,
            "__exit__": lambda self, exc_type, exc_val, exc_tb: None,
        },
    )()

    run_process.fitz.open = lambda *args, **kwargs: mock_doc

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


# ------------------ MilvusColpaliManager Tests ------------------
def test_milvus_colpali_manager_init_error():
    """Test that MilvusColpaliManager raises error when collection doesn't exist."""
    from mmore.colpali import milvuscolpali

    original_milvus_client = milvuscolpali.MilvusClient

    mock_client_instance = type(
        "obj",
        (object,),
        {
            "has_collection": lambda self, name: False,
        },
    )()

    milvuscolpali.MilvusClient = lambda *args, **kwargs: mock_client_instance

    try:
        with pytest.raises(ValueError, match="does not exist"):
            MilvusColpaliManager(
                db_path="./test_milvus",
                collection_name="test_collection",
                dim=128,
                create_collection=False,
            )
    finally:
        milvuscolpali.MilvusClient = original_milvus_client


def test_milvus_colpali_manager_insert_from_dataframe():
    """Test that MilvusColpaliManager correctly inserts data from DataFrame."""
    from mmore.colpali import milvuscolpali

    original_milvus_client = milvuscolpali.MilvusClient

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

    milvuscolpali.MilvusClient = lambda *args, **kwargs: mock_client_instance

    try:
        manager = MilvusColpaliManager(
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
        milvuscolpali.MilvusClient = original_milvus_client


def test_milvus_colpali_manager_search_embeddings_search_error():
    """Test that search_embeddings handles search errors correctly."""
    from mmore.colpali import milvuscolpali

    original_milvus_client = milvuscolpali.MilvusClient

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

    milvuscolpali.MilvusClient = lambda *args, **kwargs: mock_client_instance

    try:
        manager = MilvusColpaliManager(
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
        milvuscolpali.MilvusClient = original_milvus_client


# ------------------ Index Function Tests ------------------
def test_index_function():
    """Test that index function loads config and indexes data correctly."""
    from mmore.colpali import run_index

    original_load_config = run_index.load_config
    original_read_parquet = run_index.pd.read_parquet
    original_manager_class = run_index.MilvusColpaliManager

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
                    "dim": 128,
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
    run_index.MilvusColpaliManager = lambda *args, **kwargs: mock_manager

    try:
        index("test_config.yml")

        assert len(insert_called) > 0, "Should call insert_from_dataframe"
        assert len(create_index_called) > 0, "Should call create_index"
    finally:
        run_index.load_config = original_load_config
        run_index.pd.read_parquet = original_read_parquet
        run_index.MilvusColpaliManager = original_manager_class


# ------------------ ColPaliRetriever Tests ------------------
def test_colpali_retriever_get_relevant_documents_with_text_map():
    """Test that _get_relevant_documents correctly integrates text content from text_map."""
    from mmore.colpali import milvuscolpali, retriever

    original_load_model = retriever.load_model
    original_milvus_client = milvuscolpali.MilvusClient
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

    # Mock manager - inherit from MilvusColpaliManager to pass Pydantic validation
    class MockMilvusColpaliManager(MilvusColpaliManager):
        def __init__(self, *args, **kwargs):
            # Skip parent initialization to avoid MilvusClient setup
            # Set minimal required attributes
            self.uri = "./test_milvus"
            self.collection_name = "test_collection"
            self.dim = 128
            self.metric_type = "IP"
            self.client = None

        def search_embeddings(self, query_embeddings, top_k, max_workers):
            return mock_search_results[:top_k]

    mock_manager = MockMilvusColpaliManager()

    # Mock embed_queries
    def mock_embed_queries(texts, model, processor):
        return [np.array([0.1] * 128, dtype=np.float32)]

    milvuscolpali.MilvusClient = lambda *args, **kwargs: type(
        "obj",
        (object,),
        {
            "has_collection": lambda self, name: True,
            "load_collection": lambda self, name: None,
        },
    )()
    retriever.load_model = lambda *args, **kwargs: (mock_model, mock_processor)
    retriever.embed_queries = mock_embed_queries

    try:
        config = ColPaliRetrieverConfig(
            db_path="./test_milvus",
            collection_name="test_collection",
            model_name="vidore/colpali-v1.3",
            top_k=2,
            dim=128,
        )

        retriever_instance = ColPaliRetriever(
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
        retriever.load_model = original_load_model
        milvuscolpali.MilvusClient = original_milvus_client
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

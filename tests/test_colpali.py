import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mmore.colpali.milvuscolpali import MilvusColpaliManager
from mmore.colpali.retriever import (
    ColPaliRetriever,
    ColPaliRetrieverConfig,
    load_text_mapping,
)
from mmore.colpali.run_index import index
from mmore.colpali.run_process import (
    ColPaliEmbedder,
    PDFConverter,
    process_single_pdf,
)

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


def test_pdf_converter_cleanup():
    """Test that PDFConverter cleanup removes temporary directory."""
    converter = PDFConverter()
    tmp_root = converter.tmp_root
    assert tmp_root.exists(), "Temporary directory should exist"

    converter.cleanup()
    assert not tmp_root.exists(), "Temporary directory should be removed after cleanup"


# ------------------ ColPaliEmbedder Tests ------------------
def test_colpali_embedder_init():
    """Test that ColPaliEmbedder initializes correctly."""
    from mmore.colpali import run_process

    # Override ColPali.from_pretrained to bypass model loading
    original_colpali = run_process.ColPali.from_pretrained
    original_processor = run_process.ColPaliProcessor.from_pretrained

    mock_model = type(
        "obj",
        (object,),
        {
            "device": "cpu",
            "eval": lambda self: self,
        },
    )()

    run_process.ColPali.from_pretrained = lambda *args, **kwargs: mock_model
    run_process.ColPaliProcessor.from_pretrained = lambda *args, **kwargs: type(
        "obj", (object,), {}
    )()

    try:
        embedder = ColPaliEmbedder(model_name="vidore/colpali-v1.3", device="cpu")
        assert embedder.device == "cpu", "Device should be set correctly"
        assert embedder.model is not None, "Model should be initialized"
        assert embedder.processor is not None, "Processor should be initialized"
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


def test_milvus_colpali_manager_create_collection():
    """Test that MilvusColpaliManager creates collection correctly."""
    from mmore.colpali import milvuscolpali

    original_milvus_client = milvuscolpali.MilvusClient

    create_collection_called = []
    mock_schema = type(
        "obj", (object,), {"add_field": lambda self, *args, **kwargs: None}
    )()

    mock_client_instance = type(
        "obj",
        (object,),
        {
            "has_collection": lambda self, name: False,
            "create_schema": lambda self, **kwargs: mock_schema,
            "create_collection": lambda self, **kwargs: create_collection_called.append(
                True
            ),
        },
    )()

    milvuscolpali.MilvusClient = lambda *args, **kwargs: mock_client_instance

    try:
        MilvusColpaliManager(
            db_path="./test_milvus",
            collection_name="test_collection",
            dim=128,
            create_collection=True,
        )

        assert len(create_collection_called) > 0, (
            "Should call create_collection when create_collection=True"
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
def test_colpali_retriever_from_config():
    """Test that ColPaliRetriever.from_config creates instance correctly."""
    from mmore.colpali import milvuscolpali, retriever

    original_load_model = retriever.load_model
    original_milvus_client = milvuscolpali.MilvusClient

    mock_model = type("obj", (object,), {})()
    mock_processor = type("obj", (object,), {})()

    # Create a real manager instance with mocked MilvusClient
    mock_client_instance = type(
        "obj",
        (object,),
        {
            "has_collection": lambda self, name: True,
            "load_collection": lambda self, name: None,
        },
    )()

    milvuscolpali.MilvusClient = lambda *args, **kwargs: mock_client_instance
    retriever.load_model = lambda *args, **kwargs: (mock_model, mock_processor)

    try:
        config = ColPaliRetrieverConfig(
            db_path="./test_milvus",
            collection_name="test_collection",
            model_name="vidore/colpali-v1.3",
            top_k=3,
            dim=128,
        )

        retriever_instance = ColPaliRetriever.from_config(config)

        assert retriever_instance.model is not None, "Model should be set"
        assert retriever_instance.processor is not None, "Processor should be set"
        assert retriever_instance.manager is not None, "Manager should be set"
        assert isinstance(retriever_instance.manager, MilvusColpaliManager), (
            "Manager should be MilvusColpaliManager instance"
        )
        assert retriever_instance.config == config, "Config should be set"
    finally:
        retriever.load_model = original_load_model
        milvuscolpali.MilvusClient = original_milvus_client


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


def test_load_text_mapping_none():
    """Test that load_text_mapping returns None when path is None."""
    text_map = load_text_mapping(None)
    assert text_map is None, "Should return None when path is None"


def test_load_text_mapping_not_found():
    """Test that load_text_mapping handles missing file gracefully."""
    text_map = load_text_mapping("nonexistent.parquet")
    assert text_map is None, "Should return None when file doesn't exist"

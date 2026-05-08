from unittest.mock import patch

import pytest
from conftest import SAMPLE_DOCS, FakeSparseEmbedding
from langchain_community.embeddings import FakeEmbeddings
from langchain_core.documents import Document
from pymilvus import MilvusClient

from pymilvus.exceptions import MilvusException
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

from mmore.index.indexer import Indexer
from mmore.rag.llm import LLMConfig

from mmore.rag.retriever import Retriever, _parse_image_paths
from mmore.rag.model.dense.base import DenseModelConfig
from mmore.rag.model.sparse.base import SparseModelConfig

_COLLECTION = "test_col"


@pytest.fixture(scope="module")
def populated_db(tmp_path_factory):
    db_path = str(tmp_path_factory.mktemp("rag_db") / "test.db")
    with patch(
        "mmore.index.indexer.SparseModel.from_config",
        return_value=FakeSparseEmbedding(),
    ):
        client = MilvusClient(db_path, enable_sparse=True)
        indexer = Indexer(
            dense_model_config=DenseModelConfig(model_name="debug"),
            sparse_model_config=SparseModelConfig(
                model_name="naver/splade-cocondenser-selfdistil"
            ),
            client=client,
        )
        indexer.index_documents(SAMPLE_DOCS, collection_name=_COLLECTION)
    return db_path


def test_retriever_initialization_real(tmp_path):
    """Check that retriever initializes correctly the Milvus Lite client."""
    client = MilvusClient(str(tmp_path / "test.db"), enable_sparse=True)
    retriever = Retriever(
        dense_model=FakeEmbeddings(size=2048),
        sparse_model=FakeSparseEmbedding(),
        client=client,
        hybrid_search_weight=0.5,
        k=2,
        use_web=False,
        reranker_model=None,
        reranker_tokenizer=None,
    )
    assert isinstance(retriever, Retriever)


@patch("mmore.rag.retriever.Retriever.rerank")
def test_rerank_batch(mock_rerank):
    """Test the reranking logic and ensure docs are sorted correctly by mock model scores."""

    docs = [
        Document(page_content="doc1", metadata={"id": "1"}),
        Document(page_content="doc2", metadata={"id": "2"}),
    ]

    def mock_rerank_side_effect(query, docs):
        scores = [0.1, 2.0]
        scored_docs = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        reranked_docs = []
        for doc, score in scored_docs:
            new_doc = doc.copy()
            new_doc.metadata["similarity"] = score
            reranked_docs.append(new_doc)
        return reranked_docs

    mock_rerank.side_effect = mock_rerank_side_effect

    retriever = Retriever(
        dense_model=MockEmbeddings(),
        sparse_model=MockSparse(),
        client=MockMilvus(),
        hybrid_search_weight=0.5,
        k=2,
        use_web=False,
        reranker_model=MockModel(),
        reranker_tokenizer=MockTokenizer(),
    )

    reranked = retriever.rerank("test query", docs)

    # Assertions
    assert isinstance(reranked, list)
    assert reranked[0].page_content == "doc2"
    assert reranked[1].page_content == "doc1"
    assert reranked[0].metadata["similarity"] == pytest.approx(2.0)
    mock_rerank.assert_called_once()


@patch("mmore.rag.retriever.Retriever.retrieve")
@patch("mmore.rag.retriever.Retriever.rerank")
def test_get_relevant_documents(mock_rerank, mock_retrieve):
    """Test that _get_relevant_documents integrates retrieval + reranking and transforms Milvus results to Documents."""

    # 1. Setup Mocks for Dependencies
    mock_retrieve.return_value = [
        {"id": "1", "distance": 0.1, "entity": {"text": "doc1 content"}},
        {"id": "2", "distance": 0.3, "entity": {"text": "doc2 content"}},
    ]

    def mock_rerank_side_effect(query, docs, **kwargs):
        assert all(isinstance(d, Document) for d in docs)
        docs[0].metadata["similarity"] = 0.95
        docs[1].metadata["similarity"] = 0.85
        return [docs[0], docs[1]]

    mock_rerank.side_effect = mock_rerank_side_effect

    # 2. Initialize the Retriever (Real class)
    retriever = Retriever(
        dense_model=MockEmbeddings(),
        sparse_model=MockSparse(),
        client=MockMilvus(),
        hybrid_search_weight=0.5,
        k=2,
        use_web=False,
        reranker_model=MockModel(),
        reranker_tokenizer=MockTokenizer(),
    )

    # 3. Call the actual method
    docs = retriever._get_relevant_documents("query", run_manager=MagicMock())

    # 4. Assertions
    assert len(docs) == 2
    assert all(isinstance(d, Document) for d in docs)
    mock_retrieve.assert_called_once()
    mock_rerank.assert_called_once()

    assert docs[0].page_content == "doc1 content"
    assert docs[0].metadata["similarity"] == pytest.approx(0.95)
    assert docs[1].page_content == "doc2 content"
    assert docs[1].metadata["similarity"] == pytest.approx(0.85)


@pytest.mark.parametrize(
    "raw,expected",
    [
        (None, []),
        (["/tmp/a.png", "/tmp/b.png"], ["/tmp/a.png", "/tmp/b.png"]),
        ('["/tmp/a.png", "/tmp/b.png"]', ["/tmp/a.png", "/tmp/b.png"]),
        ('"/tmp/single.png"', ["/tmp/single.png"]),
        ("/tmp/literal.png", ["/tmp/literal.png"]),
        ("", []),
        ("   ", []),
    ],
)
def test_parse_image_paths(raw, expected):
    assert _parse_image_paths(raw) == expected


@patch("mmore.rag.retriever.Retriever.rerank")
@patch("mmore.rag.retriever.Retriever.retrieve")
def test_get_relevant_documents_fallback_when_image_paths_field_missing(
    mock_retrieve, mock_rerank
):
    fallback_results = [
        {
            "id": "1",
            "distance": 0.7,
            "entity": {
                "text": "doc content",
                "paragraph_positions": [[1, 0], [1, 1]],
                "page_numbers": [1],
                "paragraph_numbers": [0, 1],
            },
        }
    ]
    mock_retrieve.side_effect = [
        MilvusException(message="field image_paths not found in schema"),
        fallback_results,
    ]
    mock_rerank.side_effect = lambda query, docs, **kwargs: docs

    retriever = Retriever(
        dense_model=MockEmbeddings(),
        sparse_model=MockSparse(),
        client=MockMilvus(),
        hybrid_search_weight=0.5,
        k=2,
        use_web=False,
        reranker_model=MockModel(),
        reranker_tokenizer=MockTokenizer(),
    )

    docs = retriever._get_relevant_documents("query", run_manager=MagicMock())

    assert len(docs) == 1
    assert docs[0].metadata["image_paths"] == []
    assert docs[0].metadata["paragraph_positions"] == [[1, 0], [1, 1]]
    assert docs[0].metadata["page_numbers"] == [1]
    assert docs[0].metadata["paragraph_numbers"] == [0, 1]
    assert mock_retrieve.call_count == 2

    first_call_output_fields = mock_retrieve.call_args_list[0].kwargs["output_fields"]
    second_call_output_fields = mock_retrieve.call_args_list[1].kwargs["output_fields"]
    assert first_call_output_fields == [
        "text",
        "image_paths",
        "paragraph_positions",
        "page_numbers",
        "paragraph_numbers",
    ]
    assert second_call_output_fields == [
        "text",
        "paragraph_positions",
        "page_numbers",
        "paragraph_numbers",
    ]


@patch("mmore.rag.retriever.Retriever.rerank")
@patch("mmore.rag.retriever.Retriever.retrieve")
def test_get_relevant_documents_reraises_unexpected_retrieve_errors(
    mock_retrieve, mock_rerank
):
    mock_retrieve.side_effect = RuntimeError("network failure")
    mock_rerank.side_effect = lambda query, docs, **kwargs: docs

    retriever = Retriever(
        dense_model=MockEmbeddings(),
        sparse_model=MockSparse(),
        client=MockMilvus(),
        hybrid_search_weight=0.5,
        k=2,
        use_web=False,
        reranker_model=MockModel(),
        reranker_tokenizer=MockTokenizer(),
    )

    with pytest.raises(RuntimeError, match="network failure"):
        retriever._get_relevant_documents("query", run_manager=MagicMock())

    assert mock_retrieve.call_count == 1


def test_llm_config_generation_kwargs():
    """LLMConfig.generation_kwargs returns correct parameter names per provider."""
    mistral_config = LLMConfig(llm_name="mistral-large-3", max_new_tokens=1200)
    assert mistral_config.provider == "MISTRAL"
    assert mistral_config.generation_kwargs["max_tokens"] == 1200

    anthropic_config = LLMConfig(llm_name="claude-sonnet-4-6", max_new_tokens=1500)
    assert anthropic_config.provider == "ANTHROPIC"
    assert anthropic_config.generation_kwargs["max_tokens"] == 1500

    cohere_config = LLMConfig(llm_name="command-r-08-2024", max_new_tokens=1000)
    assert cohere_config.provider == "COHERE"
    assert cohere_config.generation_kwargs["max_tokens"] == 1000

    hf_config = LLMConfig(llm_name="gpt2", max_new_tokens=800)
    assert hf_config.provider == "HF"
    assert hf_config.generation_kwargs["max_new_tokens"] == 800

    openai_config = LLMConfig(llm_name="gpt-4o", max_new_tokens=2000)
    assert openai_config.provider == "OPENAI"
    assert openai_config.generation_kwargs["max_completion_tokens"] == 2000


@pytest.mark.gpu
def test_rerank(populated_db):
    """Reranking with bge-reranker-base on a real GPU runner."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available (this test requires a GPU)")
    device = "cuda"
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
    model = AutoModelForSequenceClassification.from_pretrained(
        "BAAI/bge-reranker-base"
    ).to(device)

    client = MilvusClient(populated_db, enable_sparse=True)
    retriever = Retriever(
        dense_model=FakeEmbeddings(size=2048),
        sparse_model=FakeSparseEmbedding(),
        client=client,
        hybrid_search_weight=0.5,
        k=2,
        use_web=False,
        reranker_model=model,
        reranker_tokenizer=tokenizer,
    )

    docs = [
        Document(
            page_content="Paris is the capital of France.",
            metadata={
                "rank": 1,
                "similarity": 0.9,
                "id": "1",
                "paragraph_positions": [],
            },
        ),
        Document(
            page_content="Milvus is an open-source vector database.",
            metadata={
                "rank": 2,
                "similarity": 0.8,
                "id": "2",
                "paragraph_positions": [],
            },
        ),
    ]

    reranked = retriever.rerank("France capital", docs)
    assert len(reranked) == 2
    assert all(isinstance(d, Document) for d in reranked)

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from PIL import Image

from mmore.rag.llm import LLMConfig
from mmore.rag.model.vision.adapters import (
    DEFAULT_HF_VISION_MODEL,
    HuggingFaceVisionAdapter,
    OpenAIMultimodalAdapter,
    get_multimodal_llm,
)
from mmore.rag.model.vision.image_utils import (
    aggregate_image_paths,
    build_vision_content,
    images_to_base64_data_urls,
    load_images_from_paths,
)
from mmore.rag.pipeline import RAGPipeline


class RecordingMultimodalLLM:
    def __init__(self, answer: str = "vision-answer"):
        self.answer = answer
        self.calls = []

    def invoke_with_images(self, text, images=None, system_prompt=None):
        self.calls.append(
            {
                "text": text,
                "images": images or [],
                "system_prompt": system_prompt,
            }
        )
        return self.answer


def _make_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [("system", "Context:\n{context}"), ("human", "{input}")]
    )


def _make_retriever_with_images():
    return RunnableLambda(
        lambda _: [
            Document(
                page_content="retrieved snippet",
                metadata={"rank": 1, "image_paths": ["/tmp/image-a.png"]},
            )
        ]
    )


def test_pipeline_vision_branch_calls_multimodal_adapter(monkeypatch):
    adapter = RecordingMultimodalLLM(answer="vision-ok")
    retriever = _make_retriever_with_images()

    monkeypatch.setattr(
        "mmore.rag.pipeline.aggregate_image_paths", lambda docs: ["/tmp/image-a.png"]
    )
    monkeypatch.setattr(
        "mmore.rag.pipeline.load_images_from_paths",
        lambda paths, max_images=20: ["img"],
    )

    pipeline = RAGPipeline(
        retriever=retriever,
        prompt_template=_make_prompt(),
        llm=None,
        use_vision=True,
        multimodal_llm=adapter,
        max_images_per_request=5,
    )

    result = pipeline(
        {"input": "What is in the image?", "collection_name": "test"},
        return_dict=True,
    )[0]

    assert result["answer"] == "vision-ok"
    assert result["image_paths"] == ["/tmp/image-a.png"]
    assert len(adapter.calls) == 1
    assert adapter.calls[0]["images"] == ["img"]
    assert "What is in the image?" in adapter.calls[0]["text"]


def test_pipeline_text_only_branch_does_not_use_multimodal_adapter():
    class FailingMultimodalLLM:
        def invoke_with_images(self, text, images=None, system_prompt=None):
            raise AssertionError("Vision adapter must not be called in text-only mode")

    retriever = _make_retriever_with_images()
    llm = RunnableLambda(lambda _: "text-only-answer")

    pipeline = RAGPipeline(
        retriever=retriever,
        prompt_template=_make_prompt(),
        llm=llm,
        use_vision=False,
        multimodal_llm=FailingMultimodalLLM(),
        max_images_per_request=5,
    )

    result = pipeline(
        {"input": "Text mode question", "collection_name": "test"},
        return_dict=True,
    )[0]

    assert result["answer"] == "text-only-answer"
    assert result["image_paths"] == []


def test_aggregate_image_paths_filters_and_deduplicates():
    docs = [
        Document(page_content="a", metadata={"image_paths": ["/tmp/a.png", "  "]}),
        Document(
            page_content="b", metadata={"image_paths": ["/tmp/a.png", "/tmp/b.png"]}
        ),
        Document(page_content="c", metadata={}),
    ]

    assert aggregate_image_paths(docs) == ["/tmp/a.png", "/tmp/b.png"]


def test_load_images_from_paths_loads_existing_images(tmp_path):
    img_path = tmp_path / "img.png"
    Image.new("RGB", (4, 4), color="red").save(img_path)
    missing_path = tmp_path / "missing.png"

    loaded = load_images_from_paths([str(img_path), str(missing_path)], max_images=2)

    assert len(loaded) == 1
    assert loaded[0].mode == "RGB"


def test_images_to_base64_data_urls_skips_invalid_objects():
    image = Image.new("RGB", (2, 2), color="blue")

    urls = images_to_base64_data_urls([image, object()])

    assert len(urls) == 1
    assert urls[0].startswith("data:image/png;base64,")


def test_build_vision_content_puts_text_then_images():
    content = build_vision_content("question", ["data:image/png;base64,abc"])

    assert content[0] == {"type": "text", "text": "question"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == "data:image/png;base64,abc"


def test_openai_multimodal_adapter_invokes_chat_model():
    model = MagicMock()
    model.invoke.return_value = SimpleNamespace(content="ok")
    adapter = OpenAIMultimodalAdapter(model=model)
    image = Image.new("RGB", (2, 2), color="green")

    answer = adapter.invoke_with_images("hello", images=[image], system_prompt="system")

    assert answer == "ok"
    model.invoke.assert_called_once()
    messages = model.invoke.call_args.args[0]
    assert len(messages) == 2
    assert messages[0].content == "system"
    assert messages[1].content[0]["text"] == "hello"


def test_get_multimodal_llm_builds_hf_adapter_with_default_model():
    config = LLMConfig(llm_name="gpt2", max_new_tokens=123, provider="HF")

    adapter = get_multimodal_llm(config)

    assert isinstance(adapter, HuggingFaceVisionAdapter)
    assert adapter.model_id == DEFAULT_HF_VISION_MODEL
    assert adapter.max_new_tokens == 123


@patch("mmore.rag.model.vision.adapters.LLM.from_config")
def test_get_multimodal_llm_builds_openai_wrapper(mock_from_config):
    base_model = MagicMock()
    mock_from_config.return_value = base_model
    config = LLMConfig(llm_name="gpt-4o")

    adapter = get_multimodal_llm(config)

    assert isinstance(adapter, OpenAIMultimodalAdapter)
    assert adapter._model is base_model
    mock_from_config.assert_called_once_with(config)

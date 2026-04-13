# Summary of Changes - Multimodal RAG

## Goal

Enable a multimodal RAG flow where the final LLM receives:
- text context from retrieved chunks
- images associated with those chunks

The legacy text-only mode remains available when `use_vision: false`.

## Modified files

- `src/mmore/index/indexer.py`
- `src/mmore/rag/retriever.py`
- `src/mmore/rag/llm.py`
- `src/mmore/rag/pipeline.py`
- `src/mmore/rag/multimodal_llm.py`
- `examples/rag/config.yaml`

## Change details

### 1) Indexing: store image paths

In `indexer.py`:
- added `json` import
- added dynamic Milvus field `image_paths` during insert:
  - serialized as a JSON string
  - computed from modalities where `type == "image"`

Result: each indexed chunk can now carry its associated image paths.

### 2) Retrieval: robust `image_paths` handling

In `retriever.py`:
- added `json` import
- retrieval now tries `output_fields=["text", "image_paths"]`
- automatic fallback to `output_fields=["text"]` for older collection schemas
- robust parsing of `image_paths` from:
  - `None`
  - `list`
  - JSON `str`
  - plain `str`
- parsed output is stored in `Document.metadata["image_paths"]`

Result: compatibility with both old and new Milvus collections.

### 3) LLM config: vision mode toggle

In `llm.py`:
- extended `LLMConfig` with:
  - `use_vision: bool = False`
  - `organization: Optional[str] = None` (compatibility)
- synchronized `organization` and `provider`

Result: pipeline can route to vision mode without breaking text mode.

### 4) RAG pipeline: text-only vs vision branch

In `pipeline.py`:
- added `max_images_per_request` to `RAGConfig`
- changed `llm` type to optional (`Optional[BaseChatModel]`)
- added attributes:
  - `use_vision`
  - `multimodal_llm`
  - `max_images_per_request`
- in `from_config`:
  - if `use_vision=true`: do not instantiate HF text-only `HuggingFacePipeline(..., task="text-generation")`
  - build multimodal adapter via `get_multimodal_llm`
  - otherwise keep existing text-only path
- in `_build_chain`:
  - vision branch:
    1. retrieve docs
    2. `format_docs_multimodal`
    3. `load_images_from_paths(..., max_images_per_request)`
    4. build final prompt text
    5. call `invoke_with_images(...)`
  - text branch:
    - keep `prompt | llm | StrOutputParser()`
    - keep `assert llm is not None`
- output cleanup:
  - split on `<|im_start|>assistant\n` only when present

Result: avoids `Qwen2_5_VLConfig` errors caused by using the HF text-only path in vision mode.

### 5) Multimodal adapter factory

In `multimodal_llm.py`:
- updated `get_multimodal_llm` to use `organization` or `provider`

Result: more robust adapter selection across configurations.

### 6) Example config update

In `examples/rag/config.yaml`:
- enabled vision mode:
  - `llm_name: Qwen/Qwen2.5-VL-3B-Instruct`
  - `use_vision: true`
  - `max_new_tokens: 512`
- added:
  - `max_images_per_request: 20`

## Verification performed

- Lint check on modified files: OK
- Python compilation (`python3 -m compileall`): OK

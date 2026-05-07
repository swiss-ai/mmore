# Multimodal RAG

## Overview

The multimodal RAG workflow extends standard RAG by passing both:

- retrieved text chunks
- images linked to these chunks

to a vision-capable LLM.

This lets the model answer using textual context and visual evidence from your indexed documents.

## How to use

To use multimodal RAG:

1. Process documents that contain images, then **index** them. Set `indexer.dense_model.is_multimodal: true` only if you want multimodal **dense** retrieval; re-index after changing the dense model.
2. In the RAG config, pick a vision-capable `llm_name` (for example `Qwen/Qwen2.5-VL-3B-Instruct`) when using vision generation.
3. Set `rag.llm.use_vision: true` so answers receive loaded images; use `rag.max_images_per_request` to cap how many images per query. With `use_vision: false`, generation stays text-only.
4. Run `python3 -m mmore index --config_file /path/to/index.yaml`, then `python3 -m mmore rag --config-file /path/to/rag.yaml` (batch/API as in [RAG](rag.md)).

`is_multimodal` (index → dense embeddings) and `use_vision` (RAG → generation) are independent.

### Minimal Example

#### 1. Configure indexer for vision mode

To enable retrieval-time multimodal embeddings, set `is_multimodal: true` in your indexing config, then re-index.
Use the same structure as `examples/index/config.yaml`:

```yaml
indexer:
  dense_model:
    model_name: Qwen/Qwen2.5-VL-3B-Instruct
    is_multimodal: true
  sparse_model:
    model_name: splade
    is_multimodal: false
  db:
    uri: ./proc_demo.db
    name: my_db
collection_name: my_docs
documents_path: examples/postprocessor/outputs/merged/results.jsonl
```

#### 2. Configure RAG for vision mode

Start from `[examples/rag/config.yaml](https://github.com/swiss-ai/mmore/blob/master/examples/rag/config.yaml)` and update the `rag` section:

```yaml
rag:
  llm:
    llm_name: Qwen/Qwen2.5-VL-3B-Instruct
    max_new_tokens: 1200
    # To enable vision mode
    use_vision: true
  retriever:
    db:
      uri: ./proc_demo.db
      name: 'my_db'
    hybrid_search_weight: 0.5
    k: 5
    use_web: false
    reranker_model_name: BAAI/bge-reranker-base
  # For vision mode only:
  max_images_per_request: 20
  system_prompt: "Use the following context to answer the questions.\n\nContext:\n{context}"
mode: local
mode_args:
  input_file: examples/rag/queries.jsonl
  output_file: examples/rag/output.json
```

#### 3. Run the RAG pipeline

```bash
python3 -m mmore rag --config-file /path/to/config.yaml
```

You can run in batch mode (from an input file) or API mode, exactly like the standard [RAG](rag.md) page.

## How it works

**Indexing** persists each chunk’s text and `**image_paths`** in Milvus. Dense vectors come from `indexer.dense_model` (`is_multimodal` affects dense only). Sparse vectors are text-based; `**search_type: hybrid**` combines dense (possibly multimodal) with sparse; `**search_type: sparse**` alone does not use image signals for retrieval.

**At inference**, MMORE retrieves chunks, builds the text prompt from them. If `use_vision`is activated, it loads up to `max_images_per_request` images from metadata and passes text + images to the multimodal adapter. If no images load, the model still gets the text context only.

## Notes

- `use_vision` is the switch between text-only and multimodal execution.
- `is_multimodal` is configured in the indexing config (dense model) and controls multimodal embeddings for retrieval.
- `max_images_per_request` controls memory/latency trade-offs.
- You can also use larger vision models (for example `Qwen/Qwen2.5-VL-32B-Instruct` or `Qwen/Qwen2.5-VL-72B-Instruct`), provided your GPU resources are sufficient.

## Use Qwen2.5-VL

Simply install vision dependencies:

```bash
python3 -m pip install "transformers>=4.45.0" accelerate qwen-vl-utils torch
```


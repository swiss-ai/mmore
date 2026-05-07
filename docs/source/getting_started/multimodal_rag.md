# Multimodal RAG

## Overview

The multimodal RAG workflow extends standard RAG by passing both:

- retrieved text chunks
- images linked to these chunks

to a vision-capable LLM.

This lets the model answer using textual context and visual evidence from your indexed documents.

## How to use

To use multimodal RAG:

1. process and index documents that contain images
2. choose a vision-capable model in `llm_name` (for example `Qwen/Qwen2.5-VL-3B-Instruct`)
3. enable vision mode in the RAG config with `use_vision: true`
4. run the regular RAG command

When `use_vision: false`, MMORE falls back to text-only RAG.

## Minimal Example

### 1. Configure RAG for vision mode

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

### 2. Run the RAG pipeline

```bash
python3 -m mmore rag --config-file /path/to/config.yaml
```

You can run in batch mode (from an input file) or API mode, exactly like the standard [RAG](rag.md) page.

## How it works

At inference time, MMORE:

1. retrieves relevant chunks
2. extracts associated image paths from metadata
3. loads up to `max_images_per_request` images
4. sends text context + images to the multimodal adapter

If no image is available for a query, the model still receives the text context.

## Notes

- `use_vision` is the switch between text-only and multimodal execution.
- `max_images_per_request` controls memory/latency trade-offs.
- You can also use larger vision models (for example `Qwen/Qwen2.5-VL-32B-Instruct` or `Qwen/Qwen2.5-VL-72B-Instruct`), provided your GPU resources are sufficient.

## Use Qwen2.5-VL

Simply install vision dependencies:

```bash
python3 -m pip install "transformers>=4.45.0" accelerate qwen-vl-utils torch
```


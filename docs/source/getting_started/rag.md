# 🤖 RAG 

## Overview

The `rag` module enables the creation of a modular RAG inference pipeline for indexed multimodal documents.

It supports two main execution modes:

1. **API mode**: runs the pipeline as a server and exposes an API
2. **Batch mode**: runs inference from an input file of queries, for example a JSONL file

Different parts of the pipeline can be customized through a RAG inference configuration file.

## 💡 TL;DR

The RAG module lets you combine retrieval and generation over indexed multimodal documents.

In practice, it supports:

- a batch mode for file-based inference
- an API mode for serving the pipeline
- configurable retriever and LLM components
- optional WebRAG and CLI usage in batch mode

You can customize various parts of the pipeline by defining an inference RAG configuration file at
[`examples/rag/api/rag_api.yaml`](https://github.com/swiss-ai/mmore/blob/master/examples/rag/api/rag_api.yaml).


## 💻 Minimal Example:

Here is a minimal example to create a RAG pipeline hosted through [LangGraph](https://python.langchain.com/docs/langgraph/) servers.

### 1. Create a RAG inference config file

Create your RAG Inference config file based on the [batch example `examples/rag/config.yaml`](https://github.com/swiss-ai/mmore/blob/master/examples/rag/config.yaml) or the [API example `examples/rag/config_api.yaml`](https://github.com/swiss-ai/mmore/blob/master/examples/rag/config_api.yaml).

You can check the structure of the configuration file with the dataclass [RAGConfig]( https://github.com/swiss-ai/mmore/blob/master/src/mmore/rag/pipeline.py).

### 2. Start the RAG pipeline

Start your RAG pipeline using the `run_rag.py` script and your config file
```bash
python3 -m mmore rag --config-file /path/to/config.yaml
```

### 3. Query the server in API mode

In API mode, the RAG server exposes a health endpoint and a configurable RAG endpoint. By default, the RAG endpoint is `/rag`.

Check that the server is running:
```bash
curl --location --request GET http://localhost:8000/health
```

Send a RAG query:
```bash
curl --location --request POST http://localhost:8000/rag \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "What is Meditron?",
    "collection_name": "my_docs"
  }'
```
In batch mode, the pipeline is run directly with the input data specified in the configuration file, and the result is saved to the specified path.

See [`examples/rag`](https://github.com/swiss-ai/mmore/blob/master/examples/rag/) for other use cases.

## 🔎 Main modules

The RAG pipeline is built around two main modules:
1. The `Retriever`, which retrieves multimodal documents from the database. 
2. The `LLM`, which wraps different types of multimodal-able LLMs.

### Retriever

Here is an example on how to use the retriever module on its own. Note that it assumes that you already created a database using the [Indexing](indexing.md) workflow.

#### 1. Create a config

Start from the [example config file `examples/index/config.yaml`](https://github.com/swiss-ai/mmore/blob/master/examples/index/config.yaml).

#### 2. Retrieve from the vector store

```python
from mmore.rag.retriever import Retriever

# Create the Retriever
retriever = Retriever.from_config('/path/to/your/retriever_config.yaml')

# Retrieves the top 3 documents using an hybrid approach (e.g. dense + sparse embeddings)
retriever.retrieve(
    'What is Meditron?',
    k=3,
    collection_name="my_docs",
    search_type="hybrid"  # Options: "dense", "sparse", "hybrid"
)
```

### LLM

Here is an example on how to use the `LLM` module on its own. This also assumes that the indexing workflow has already been completed.

#### 1. Create a config file
```yaml
llm_name: gpt-4o-mini
max_new_tokens: 150
temperature: 0.7
```

#### 2. Query the LLM
```python
from mmore.rag.llm import LLM

# Create the LLM
llm = LLM.from_config('/path/to/your/llm_config.yaml')

# Create your messages
messages = [
(
    "system",
    "You are a helpful assistant that translates English to French. Translate the user sentence.",
),
(
    "human",
    "I love Meditron."
),
]

# Retrieves the top 3 documents using an hybrid approach (e.g. dense + sparse embeddings)
llm.invoke(messages)
```

## Vision-enabled RAG

Standard RAG passes retrieved **text** to the LLM. Vision-enabled RAG also loads **images** linked to retrieved chunks (`image_paths` in chunk metadata) and sends text + images to a vision-capable model.

Two flags are independent:

| Setting | Config | Effect |
| ------- | ------ | ------ |
| Multimodal **retrieval** | `indexer.dense_model.is_multimodal` | Dense embeddings use image+text (re-index after changing). |
| Vision **generation** | `rag.llm.use_vision` | Answer step receives loaded images; `false` keeps text-only generation. |

### Quick setup

1. Process documents with images, then **index** (set `is_multimodal: true` on the dense model only if you want multimodal dense retrieval).
2. In the RAG config, set a vision-capable `llm_name` (for local use, e.g. `Qwen/Qwen2.5-VL-3B-Instruct`) and `use_vision: true`.
3. Optionally set `rag.max_images_per_request` (default 20) to cap images per query.
4. Run index, then RAG in batch or API mode as above.

**Indexer** (`examples/index/config.yaml`):

```yaml
indexer:
  dense_model:
    model_name: Qwen/Qwen2.5-VL-3B-Instruct
    is_multimodal: true   # optional: multimodal dense retrieval
  sparse_model:
    model_name: splade
    is_multimodal: false
  db:
    uri: ./proc_demo.db
    name: my_db
```

**RAG** (from [`examples/rag/config.yaml`](https://github.com/swiss-ai/mmore/blob/master/examples/rag/config.yaml)):

```yaml
rag:
  llm:
    llm_name: Qwen/Qwen2.5-VL-3B-Instruct
    max_new_tokens: 1200
    use_vision: true
    provider: HF
  retriever:
    db:
      uri: ./proc_demo.db
      name: my_db
    hybrid_search_weight: 0.5
    k: 5
  max_images_per_request: 20
  system_prompt: "Use the following context to answer the questions.\n\nContext:\n{context}"
```

Local Qwen-VL dependencies (install after indexing / after creating the DB):

```bash
python3 -m pip install -e ".[qwen,cpu]"
```

Recommended order:

1. Build the processed corpus and run indexing to create/populate the Milvus DB.
2. Only then install the Qwen extra and run vision-enabled RAG inference.

### How it works

- **Indexing** stores each chunk’s text and `image_paths` in Milvus. Dense vectors follow `indexer.dense_model`; sparse vectors are text-only. Hybrid search combines dense (optionally multimodal) with sparse; sparse-only search does not use images at retrieval time.
- **Inference** retrieves chunks and builds the text prompt. With `use_vision: true`, MMORE loads up to `max_images_per_request` images from metadata and calls the multimodal adapter. If no images load, the model still receives text context only.

Collections indexed before `image_paths` existed are supported: the retriever falls back to text-only Milvus fields automatically.

### Scope and roadmap

**This release (local generation):** Hugging Face **Qwen-VL** family (`Qwen2-VL`, `Qwen2.5-VL`, `Qwen3-VL`) via `HuggingFaceVisionAdapter` and [`qwen-vl-utils`](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-utils) for image preprocessing. Other local VLMs (e.g. LLaVA-NeXT) use different model classes and preprocessing; supporting them means adding small per-family adapters, not a single generic loader.

**Follow-up (API generation):** Retrieval and image loading are already provider-agnostic. With `use_vision: true` and a non-HF provider, the pipeline uses `OpenAIMultimodalAdapter` (LangChain chat models + multimodal messages). A later PR can document and test OpenAI/Anthropic vision APIs and align message formats per provider.

## 🔧 Customization

Our RAG pipeline is built to take full advantage of [LangChain](https://python.langchain.com/docs/introduction/) abstractions, providing compatibility with all components offered.

#### Retriever

Our retriever is a LangChain [`BaseRetriever`](https://python.langchain.com/api_reference/core/retrievers/langchain_core.retrievers.BaseRetriever.html). If you want to create a custom retriever (e.g. GraphRetriever,...) you can simply make it inherit from this class and use it as described in our examples.

#### WebRAG 
Within the `rag` pipeline, web search is currently configured through the retriever settings in local / file-based workflows.

It uses the [`DuckDuckGo Search API`](https://python.langchain.com/docs/integrations/tools/ddg/) to search the web using the input query, then adds its results to the context. 

#### CLI for RAG 
A CLI is also available for interactive querying.

Start it with:

```bash
python3 -m mmore ragcli --config-file /path/to/config.yaml
```

You can customize the CLI by defining [a RAG configuration file](https://github.com/swiss-ai/mmore/blob/master/examples/rag/config.yaml) or by setting preferences from within the CLI.



#### LLM
The LLM wrappers are based on LangChain's [`BaseChatModel`](https://python.langchain.com/api_reference/core/language_models/langchain_core.language_models.chat_models.BaseChatModel.html). 

If you want to create a custom retriever you can simply make it inherit from this class and use it as described in our examples. 

```{warning}
MMORE supports [Hugging Face Hub](https://huggingface.co/models) models.

In some cases, a simpler solution is to push a model to the Hub and use it through the existing class rather than implementing a new wrapper.
```

## Notes

The standalone `websearch` module and the `rag` pipeline do not expose web search in exactly the same way.

In particular:

- the standalone `websearch` module supports API usage, with optional RAG integration
- within the `rag` pipeline, web search is currently configured through the retriever settings in local / file-based workflows
- file-based inference may be slow when using local models

## See also

- [Indexing](indexing.md)
- [Process](process.md)
- [Architecture](architecture.md)



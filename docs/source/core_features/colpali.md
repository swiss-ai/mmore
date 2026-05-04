# 🖼️ ColPali Integration

## Overview

This module provides a complete pipeline for processing PDF documents with ColPali embeddings, storing them in a Milvus vector database, and performing semantic search.

It is designed for efficient document retrieval and RAG applications.

## 🧭 Architecture

The system consists of three main components:

1. **PDF Processor** - Extracts embeddings from PDF pages
2. **Milvus Indexer** - Stores and indexes embeddings
3. **Retriever** - Performs semantic search queries

## 📁 File Structure

```
src/mmore/colpali/
├── milvuscolpali.py      # Milvus database management
├── run_index.py          # Indexing pipeline
├── run_process.py        # PDF processing pipeline  
├── run_retriever.py      # Search and retrieval API
└── retriever.py          # ColPaliRetriever class for RAG integration
```

## 🚀 Quick Start

### 1. Process PDFs into embeddings

```bash
python3 -m mmore colpali process --config-file examples/colpali/config_process.yml
```

**Example config (`config_process.yml`):**
```yaml
data_path:
  - 'examples/sample_data/pdf'
output_path: "./output"
model_name: "vidore/colpali-v1.3"
skip_already_processed: true
num_workers: 5
batch_size: 8
```

### 2. Index embeddings into Milvus

```bash
python3 -m mmore colpali index --config-file examples/colpali/config_index.yml
```

**Example config (`config_index.yml`):**
```yaml
parquet_path: ./output/pdf_page_objects.parquet
milvus:
    db_path: ./output/milvus_data.db
    collection_name: pdf_pages
    create_collection: true
    dim: 128
    metric_type: IP
```

### 3. Run Retrieval

#### Retrieval Server Mode
```bash
# Start the retrieval API server
python3 -m mmore colpali retrieve --config-file examples/colpali/config_retrieval.yml
```

Or with a custom host and port:
```bash
python3 -m mmore colpali retrieve --config-file examples/colpali/config_retrieval.yml --host 0.0.0.0 --port 8001
```

**Example config (`config_retrieval.yml`):**
```yaml
db_path: "./milvus_data"
collection_name: "pdf_pages"
model_name: "vidore/colpali-v1.3"
top_k: 3
dim: 128
max_workers: 16
metric_type: "IP"
text_parquet_path: "./output/pdf_page_text.parquet"
```

#### Single Query Mode
```bash
# Run retrieval for a single query defined in the config file
python3 -m mmore colpali retrieve --config-file examples/colpali/config_retrieval_single.yml
```

**Example config (`config_retrieval_single.yml`):**
```yaml
mode: "single"
db_path: "./milvus_data"
collection_name: "pdf_pages"
model_name: "vidore/colpali-v1.3"
query: "What may lead to dysbiosis and inflammation?"
top_k: 5
```
Host and port are specified via CLI flags (`--host` and `--port`), not in the config file.

#### Batch Mode
```bash
# Process queries from file
python3 -m mmore colpali retrieve --config-file examples/colpali/config_retrieval.yml --input-file queries.jsonl --output-file results.json
```

**Example queries file (`queries.jsonl`):**
Each line should be a JSON-encoded string (one query per line):
```jsonl
"machine learning"
"neural networks"
"data processing"
```

Each line must be a valid JSON string, including quotes, since the file is parsed line by line with `json.loads()`.

**Example config (`config_retrieval.yml`):**
```yaml
db_path: "./milvus_data"
collection_name: "pdf_pages"
model_name: "vidore/colpali-v1.3"
top_k: 5
dim: 128
max_workers: 16
text_parquet_path: "./output/pdf_page_text.parquet"
```

## 🔧 Core Components

### MilvusColpaliManager
- manages local Milvus database operations
- handles collection creation and indexing
- provides efficient batch insertion
- implements hybrid search with reranking

**Key Features:**
- local Milvus instance with no external dependencies
- automatic collection management
- multi-vector support for pages
- efficient batch operations

### PDF Processor
- converts PDF pages to images
- generates ColPali embeddings
- handles parallel processing
- supports stop-and-resume workflows for large datasets

**Processing Flow:**
1. Crawl PDF files from specified directories
2. Convert each page to high-resolution PNG
3. Generate embeddings using ColPali model
4. Store results in Parquet format

### Retriever
- supports multiple usage modes: server mode by default, single-query mode via config, or batch mode with `--input-file` and `--output-file`
- performs fast semantic search with reranking
- exposes a REST API for integration
- supports configurable top-k results
- provides a LangChain-compatible `BaseRetriever` for RAG integration
- can retrieve page text through the `text_parquet_path` configuration

## 🎯 Use Cases

### Document Retrieval
```bash
# Example API call
curl -X POST "http://localhost:8001/v1/retrieve" \
     -H "Content-Type: application/json" \
     -d '{"query": "machine learning", "top_k": 3}'
```

**Response format:**
```json
{
  "query": "machine learning",
  "results": [
    {
      "pdf_name": "ml_book.pdf",
      "pdf_path": "/path/to/ml_book.pdf",
      "page_number": 42,
      "content": "Machine learning is a subset of artificial intelligence...",
      "similarity": 0.894,
      "rank": 1
    }
  ]
}
```

### RAG Pipeline Integration
```python
from mmore.colpali.retriever import ColPaliRetriever, ColPaliRetrieverConfig
from mmore.rag.pipeline import RAGPipeline, RAGConfig

# Create ColPali retriever with text support
colpali_config = ColPaliRetrieverConfig(
    db_path="./output/milvus_data.db",
    collection_name="pdf_pages",
    model_name="vidore/colpali-v1.3",
    text_parquet_path="./output/pdf_page_text.parquet",
    top_k=3,
    dim=128,
    max_workers=16,
    metric_type="IP",
)
colpali_retriever = ColPaliRetriever.from_config(colpali_config)

# Use with RAG pipeline (requires LLM config)
# rag_config = RAGConfig(retriever=colpali_retriever, ...)
# rag_pipeline = RAGPipeline.from_config(rag_config)
```

The `ColPaliRetriever` is a LangChain-compatible `BaseRetriever` that returns `Document` objects with:
- `page_content`: the text content from the PDF page, if `text_parquet_path` is provided
- `metadata`: contains `pdf_name`, `pdf_path`, `page_number`, `rank`, and `similarity` score

## 📦 Output Formats

### Process Output

**Embeddings Parquet (`pdf_page_objects.parquet`)**
```json
{
  "pdf_path": "/path/to/doc1.pdf",
  "page_number": 1,
  "embedding": [0.1, 0.2, "..."]
}
```
**Text Mapping Parquet (`pdf_page_text.parquet`)**
```json
{
  "pdf_path": "/path/to/doc1.pdf",
  "page_number": 1,
  "text": "Page content text here..."
}
```

### Search Results

**API Response:**
```json
{
  "query": "machine learning",
  "results": [
    {
      "pdf_name": "ml_book.pdf",
      "pdf_path": "/path/to/ml_book.pdf",
      "page_number": 42,
      "content": "Machine learning is a subset of artificial intelligence...",
      "similarity": 0.894,
      "rank": 1
    }
  ]
}
```

**Batch Mode Output:**
```json
{
  "query": "machine learning",
  "context": [
    {
      "page_content": "Machine learning is a subset of artificial intelligence...",
      "metadata": {
        "pdf_name": "ml_book.pdf",
        "pdf_path": "/path/to/ml_book.pdf",
        "page_number": 42,
        "rank": 1,
        "similarity": 0.894
      }
    }
  ]
}
```

## 🔁 Pipeline Example

### Complete Workflow
```bash
# 1. Process all PDFs in a directory
python3 -m mmore colpali process --config-file examples/colpali/config_process.yml

# 2. Index the embeddings
python3 -m mmore colpali index --config-file examples/colpali/config_index.yml

# 3. Start the API server
python3 -m mmore colpali retrieve --config-file examples/colpali/config_retrieval.yml

# 4. Query the system
curl -X POST "http://localhost:8001/v1/retrieve" \
     -H "Content-Type: application/json" \
     -d '{"query": "your search query", "top_k": 3}'
```

### Alternative: Batch processing
```bash
# 1. Process PDFs (same as above)
python3 -m mmore colpali process --config-file examples/colpali/config_process.yml

# 2. Index embeddings (same as above)
python3 -m mmore colpali index --config-file examples/colpali/config_index.yml

# 3. Run batch retrieval
python3 -m mmore colpali retrieve --config-file examples/colpali/config_retrieval.yml \
                       --input-file queries.jsonl \
                       --output-file results.json
```

## 💡 Configuration tips

### For large datasets
- increase `batch_size` and `num_workers` in process config
- use `skip_already_processed: true` for incremental processing

### For better accuracy
- use higher DPI in PDF conversion, default is 200
- increase `top_k` in retrieval to inspect more candidate pages
- consider using larger ColPali models if available

### For production
- run Milvus in distributed mode for larger datasets
- use the API mode for scalable serving
- implement caching for frequent queries

## See also

- [Process](../getting_started/process.md)
- [Indexing](../getting_started/indexing.md)
- [RAG](../getting_started/rag.md)

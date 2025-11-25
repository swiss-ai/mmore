# ColPali Integration for MMORE

## Overview

This module provides a complete pipeline for processing PDF documents using ColPali embeddings, storing them in a Milvus vector database, and performing semantic search. It is designed for efficient document retrieval and RAG applications.

## Architecture

The system consists of three main components:

1. **PDF Processor** - Extracts embeddings from PDF pages
2. **Milvus Indexer** - Stores and indexes embeddings
3. **Retriever** - Performs semantic search queries

## File Structure

```
src/mmore/colpali/
├── milvuscolpali.py      # Milvus database management
├── run_index.py          # Indexing pipeline
├── run_process.py        # PDF processing pipeline  
├── run_retriever.py      # Search and retrieval API
└── retriever.py          # ColPaliRetriever class for RAG integration
```

## Quick Start

### 1. Process PDFs into Embeddings

```bash
# Process PDFs and generate embeddings
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

### 2. Index Embeddings into Milvus

```bash
# Index embeddings into Milvus database
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

#### API Mode (Recommended)
```bash
# Start the retrieval API server
python3 -m mmore colpali retrieve --config-file examples/colpali/config_retrieval.yml
```

Or with custom host and port:
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

Note: Host and port are specified via CLI flags (`--host` and `--port`), not in the config file.

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

Note: Each line must be a valid JSON string (with quotes), as the file is parsed line-by-line using `json.loads()`.

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
- Manages local Milvus database operations
- Handles collection creation and indexing
- Provides efficient batch insertion
- Implements hybrid search with reranking

**Key Features:**
- Local Milvus instance (no external dependencies)
- Automatic collection management
- Multi-vector support for pages
- Efficient batch operations

### PDF Processor
- Converts PDF pages to images
- Generates ColPali embeddings
- Handles parallel processing
- Ability to stop and resume processing for large datasets

**Processing Flow:**
1. Crawl PDF files from specified directories
2. Convert each page to high-resolution PNG
3. Generate embeddings using ColPali model
4. Store results in Parquet format

### Retriever
- Multiple operation modes: API mode (default) or batch mode (with `--input-file` and `--output-file`)
- Fast semantic search with reranking
- REST API for integration
- Configurable top-k results
- LangChain-compatible `BaseRetriever` for RAG pipeline integration
- Text content retrieval via `text_parquet_path` configuration

## Use Cases

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
- `page_content`: The text content from the PDF page (if `text_parquet_path` is provided)
- `metadata`: Contains `pdf_name`, `pdf_path`, `page_number`, `rank`, and `similarity` score

## Output Formats

### Process Output

**Embeddings Parquet (`pdf_page_objects.parquet`):**
```parquet
pdf_path | page_number | embedding
---------|-------------|-----------
/path/to/doc1.pdf | 1 | [0.1, 0.2, ...]
```

**Text Mapping Parquet (`pdf_page_text.parquet`):**
```parquet
pdf_path | page_number | text
---------|-------------|-----------
/path/to/doc1.pdf | 1 | "Page content text here..."
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
[
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
]
```

## Pipeline Example

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

**Alternative: Batch Processing**
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

## Configuration Tips

### For Large Datasets
- Increase `batch_size` and `num_workers` in process config
- Use `skip_already_processed: true` for incremental processing

### For Better Accuracy
- Use higher DPI in PDF conversion (default: 200)
- Increase `top_k` in retrieval for more candidate pages
- Consider using larger ColPali models if available

### For Production
- Run Milvus in distributed mode for larger datasets
- Use the API mode for scalable serving
- Implement caching for frequent queries
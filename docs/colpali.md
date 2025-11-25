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
src/mmore/colopali/milvuscolpali.py      # Milvus database management
src/mmore/colopali/run_index.py          # Indexing pipeline
src/mmore/colopali/run_process.py        # PDF processing pipeline  
src/mmore/colopali/run_retriever.py      # Search and retrieval API
```

## Quick Start

### 1. Process PDFs into Embeddings

```bash
# Process PDFs and generate embeddings
python3 -m src.mmore.colpali.run_process --config_file examples/colpali/config_process.yml
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
python3 -m src.mmore.colpali.run_index --config_file examples/colpali/config_index.yml
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
python3 -m src.mmore.colpali.run_retriever --config_file examples/colpali/config_retrieval_api.yml
```

**Example config (`config_retrieval_api.yml`):**
```yaml
mode: "api"
db_path: "./milvus_data"
collection_name: "pdf_pages"
model_name: "vidore/colpali-v1.3"
host: "0.0.0.0"
port: 8001
top_k: 3
dim: 128
max_workers: 16
```

#### Single Query Mode
```bash
# Run a single query
python3 -m src.mmore.colpali.run_retriever --config_file examples/colpali/config_retrieval_single.yml
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

#### Batch Mode
```bash
# Process queries from file
python3 -m src.mmore.colpali.run_retriever --config_file examples/colpali/config_retrieval_batch.yml
```

**Example queries file (`queries.json`):**
```json
["machine learning", "neural networks", "data processing"]
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
- Resume ability for large datasets

**Processing Flow:**
1. Crawl PDF files from specified directories
2. Convert each page to high-resolution PNG
3. Generate embeddings using ColPali model
4. Store results in Parquet format

### Retriever
- Multiple operation modes: API, batch, single query
- Fast semantic search with reranking
- REST API for integration
- Configurable top-k results

## Use Cases

### Document Retrieval
```python
# Example API call
curl -X POST "http://localhost:8001/v1/retrieve" \
     -H "Content-Type: application/json" \
     -d '{"query": "machine learning", "top_k": 3}'
```

### RAG Pipeline Integration
```python
# Get relevant documents for RAG
results = retriever.search("neural network architectures")
context = "\n".join([f"Page {r['page_number']}: {get_page_content(r)}" for r in results])
```

### Batch Processing
```python
# Process multiple queries efficiently
queries = ["AI safety", "transformer models", "reinforcement learning"]
batch_results = process_queries_in_batch(queries)
```

## Output Formats

### Process Output
```parquet
pdf_name | page_number | pdf_path | embedding
---------|-------------|----------|-----------
doc1.pdf | 1           | /path/... | [0.1, 0.2, ...]
```

### Search Results
```json
{
  "query": "machine learning",
  "results": [
    {
      "pdf_name": "ml_book.pdf",
      "page_number": 42,
      "score": 0.894,
      "rank": 1
    }
  ]
}
```

## Pipeline Example

### Complete Workflow
```bash
# 1. Process all PDFs in a directory
python3 -m src.mmore.colpali.run_process --config_file examples/colpali/config_process.yml

# 2. Index the embeddings
python3 -m src.mmore.colpali.run_index --config_file examples/colpali/config_index.yml

# 3. Start the API server
python3 -m src.mmore.colpali.run_retriever --config_file examples/colpali/config_retrieval_api.yml

# 4. Query the system
curl -X POST "http://localhost:8001/v1/retrieve" \
     -H "Content-Type: application/json" \
     -d '{"query": "your search query", "top_k": 3}'
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
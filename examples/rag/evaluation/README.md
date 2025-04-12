# Unified RAGAS Evaluation for MMORE

This directory contains a simplified approach to running RAGAS evaluations for RAG systems using a single unified configuration file.

## Overview

The RAGAS evaluation process typically requires three separate configuration files:
1. **Evaluator Config**: Defines the dataset, metrics, and evaluation models
2. **Indexer Config**: Configures the vector database and embedding models
3. **RAG Pipeline Config**: Sets up the RAG system to be evaluated

Our unified approach combines these into a single YAML file for easier management and better user experience.

## Files

- `unified_rag_evaluation_config.yaml`: Single configuration file for the entire evaluation process
- `/src/mmore/run_evaluation.py`: Script to run the evaluation using the unified config (located in the main source directory)

## Usage

```bash
python -m src.mmore.run_evaluation --config-file examples/rag/evaluation/unified_rag_evaluation_config.yaml --output results.yaml
```

### Command Line Arguments

- `--config-file`: Path to the unified configuration file (required)
- `--output`: Optional path to save evaluation results as YAML

## Configuration Structure

The unified configuration file has the following sections:

### Dataset Configuration
```yaml
dataset:
  hf_dataset_name: "Mallard74/eval_medical_benchmark"  # Hugging Face dataset name
  split: "train"                                       # Dataset split to use
  feature_map:                                         # Map dataset columns to expected format
    user_input: "user_input"                           # Query/question column
    reference: "reference"                             # Ground truth answer column
    corpus: "corpus"                                   # Document corpus column
    query_id: "query_ids"                              # Unique identifier column
```

### Evaluation Metrics
```yaml
metrics:
  - LLMContextRecall       # Measures if retrieved context contains info needed to answer
  - Faithfulness           # Measures if generated answer is supported by context
  - FactualCorrectness     # Measures factual accuracy of generated answer
  - SemanticSimilarity     # Measures semantic similarity between generated and reference answers
```

### Embedding Models
```yaml
embeddings:
  name: "all-MiniLM-L6-v2"  # Embedding model for evaluation
```

### Evaluator LLM Configuration
```yaml
evaluator_llm:
  llm_name: "gpt-4o"        # LLM used for evaluation
  max_new_tokens: 150       # Maximum tokens for evaluation responses
```

### Indexer Configuration
```yaml
indexer:
  dense_model:
    model_name: "sentence-transformers/all-MiniLM-L6-v2"  # Dense retrieval model
    is_multimodal: false                                 # Whether the model handles multimodal data
  sparse_model:
    model_name: "splade"                                # Sparse retrieval model
    is_multimodal: false                               # Whether the model handles multimodal data
  db:
    uri: "./examples/rag/unified_eval_benchmark.db"     # Vector database location
    name: "unified_eval_benchmark"                      # Database name
  chunker:
    chunking_strategy: "sentence"                       # Text chunking strategy
```

### RAG Pipeline Configuration
```yaml
rag_pipeline:
  llm:
    llm_name: "gpt-4o-mini"  # LLM to evaluate in the RAG pipeline
    max_new_tokens: 150      # Maximum tokens for RAG responses
  retriever:
    db:
      uri: "./examples/rag/unified_eval_benchmark.db"  # Same DB as indexer
    hybrid_search_weight: 0.5  # Weight between dense and sparse retrieval (0=dense only, 1=sparse only)
    k: 3                       # Number of documents to retrieve
```

## Requirements

- OpenAI API key (for evaluation LLM)
- Hugging Face dataset access
- Python packages: ragas, langchain, pymilvus, etc.

## Notes

- The script will create temporary component-specific configuration files during execution
- Collection names are automatically sanitized to be compatible with Milvus
- Results are displayed in the console and can optionally be saved to a file

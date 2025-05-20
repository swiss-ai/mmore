# 🤖 MMORE RAG Evaluation Pipeline

## 💡 TL;DR

The `RAG` module comes with an Evaluator that allows you to evaluate your full RAG pipeline—from the context retrieval to the LLM's output:

1. **Prepare your benchmark evaluation dataset** in the required format.
2. **Choose your list of metrics** to evaluate [Available Metrics](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/).
3. **Set up your configuration files** for the evaluator, indexer, and RAG pipeline.
4. **Run the evaluation** for your RAG setup (retriever + LLM).

🚀 **Powered by RAGAS**: Ragas is a library designed to supercharge the evaluation of Large Language Model (LLM) applications.

## 💻 Minimal Example

Here's a step-by-step guide to set up the evaluation pipeline:

### 1. **Create the Evaluator Config File**:

This file defines the evaluation settings for your pipeline.

```yaml
hf_dataset_name: "Mallard74/eval_medical_benchmark"  # Hugging Face Eval dataset name (Example dataset)
split: "train"  # Dataset split
hf_feature_map: {'user_input': 'user_input', 'reference': 'reference', 'corpus': 'corpus', 'query_id': 'query_ids'}  # Column mapping
metrics:  # List of metrics to evaluate
  - LLMContextRecall
  - Faithfulness
  - FactualCorrectness
  - SemanticSimilarity
embeddings_name: "all-MiniLM-L6-v2"  # Evaluator Embedding model name
llm:  # Evaluator LLM config
  llm_name: "gpt-4o"
  max_new_tokens: 150
```
### 2. **Create the Indexer Config File**:

This file configures the indexer for your evaluation.

```yaml
dense_model_name: sentence-transformers/all-MiniLM-L6-v2
sparse_model_name: splade
db:
  uri: "./examples/rag/milvus_mock_eval_medical_benchmark.db"  # Dataset's Vectorstore URI
  name: "mock_eval_medical_benchmark"
chunker:
  chunking_strategy: sentence  # Your chunking strategy
```

### 3. **Create the RAG Pipeline Config File**:

This file sets up the RAG pipeline for evaluation.

```yaml
llm:
  llm_name: "gpt-4o-mini"  # RAG LLM model to evaluate
  max_new_tokens: 150
retriever:
  db:
    uri: "./examples/rag/milvus_mock_eval_medical_benchmark.db"  # Dataset's Vectorstore URI
  hybrid_search_weight: 0.5
  k: 3
```
    
### 4. **Run the Evaluation**:

Once the configuration files are in place, you can run the evaluation pipeline with the following Python script:

```python
from mmore.rag.evaluator import RAGEvaluator

# Instantiate RAGEvaluator
evaluator = RAGEvaluator.from_config(args.eval_config)

# Run the evaluation
result = evaluator(
    indexer_config=args.indexer_config,
    rag_config=args.rag_config
)
```

- See [`examples/rag/evaluation`](../examples/rag/evaluation) for a simple example.
> :warning: Note that you should create a separate database file for each dataset. The pipeline will create partitions per dense model for convenience.
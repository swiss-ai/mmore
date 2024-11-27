import os
import sys

from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, SemanticSimilarity
from langchain_huggingface import HuggingFacePipeline
import argparse
import pandas as pd
from datasets import load_dataset  # Load datasets from the HF Hub
from src.mmore.rag.evaluator import EvalConfig, RAGEvaluator
from src.mmore.rag.llm import LLMConfig, LLM
from src.mmore.index.indexer import DBConfig

from dotenv import load_dotenv
load_dotenv()

EXAMPLE_LLM = 'gpt-4o-mini'
EXAMPLE_DENSE = 'all-MiniLM-L6-v2'
EXAMPLE_SPARSE = 'splade'

MOCK_EVALUATOR_CONFIG = './examples/rag/evaluation/rag_eval_example.yaml'

def get_args():
    parser = argparse.ArgumentParser(description='Run RAG Evaluation pipeline with specified parameters or use default mock data')
    parser.add_argument('--eval-config', type=str, default=MOCK_EVALUATOR_CONFIG, help='Path to a rag eval config file.')
    parser.add_argument('--llm', type=str, help='Model to evaluate')
    parser.add_argument('--dense', type=str, help='Dense retrieval model')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    # Instantiate RAGEvaluator
    evaluator = RAGEvaluator.from_config(args.eval_config)

    # Run the evaluation
    result = evaluator(
        llm=args.llm if args.llm else EXAMPLE_LLM,
        dense=args.dense if args.dense else EXAMPLE_DENSE,
        sparse='splade',
        k=3
    )

    print(result)
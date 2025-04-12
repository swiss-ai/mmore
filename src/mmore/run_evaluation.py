#!/usr/bin/env python
"""
RAGAS Evaluation Runner

This script uses a unified YAML configuration file to run the complete
RAGAS evaluation pipeline for RAG systems.
"""

import os
import sys
import argparse
import yaml
import re
import time
from typing import Dict, Any
import logging
from dataclasses import dataclass, field

from .rag.evaluator import EvalConfig, RAGEvaluator
from .rag.llm import LLMConfig
from .index.indexer import IndexerConfig, DBConfig, Indexer
from .type import MultimodalSample
from .utils import load_config

# Setup logging
EVAL_EMOJI = "📊"
logging.basicConfig(
    format=f'[RAGAS Eval {EVAL_EMOJI} -- %(asctime)s] %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Custom wrapper for Indexer to make it compatible with RAGEvaluator
class CompatibleIndexer(Indexer):
    @property
    def dense_model_name(self):
        return self.dense_model_config.model_name

@dataclass
class UnifiedEvalConfig:
    """Unified configuration for RAGAS evaluation."""
    dataset: Dict[str, Any]
    metrics: list
    embeddings: Dict[str, Any]
    evaluator_llm: Dict[str, Any]
    indexer: Dict[str, Any]
    rag_pipeline: Dict[str, Any]

def load_unified_config(config_path: str) -> UnifiedEvalConfig:
    """Load the unified configuration file."""
    logger.info(f"Loading unified configuration from: {config_path}")
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return config_dict

def create_evaluator_config(unified_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract evaluator configuration from unified config."""
    return {
        "hf_dataset_name": unified_config["dataset"]["hf_dataset_name"],
        "split": unified_config["dataset"]["split"],
        "hf_feature_map": unified_config["dataset"]["feature_map"],
        "metrics": unified_config["metrics"],
        "embeddings_name": unified_config["embeddings"]["name"],
        "llm": unified_config["evaluator_llm"]
    }

def create_indexer_config(unified_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract indexer configuration from unified config."""
    return {
        "dense_model": unified_config["indexer"]["dense_model"],
        "sparse_model": unified_config["indexer"]["sparse_model"],
        "db": unified_config["indexer"]["db"],
        "chunker": unified_config["indexer"]["chunker"]
    }

def create_rag_config(unified_config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract RAG pipeline configuration from unified config."""
    return {
        "llm": unified_config["rag_pipeline"]["llm"],
        "retriever": unified_config["rag_pipeline"]["retriever"]
    }

class TempConfigFile:
    """Context manager for temporary configuration files."""
    def __init__(self, config_dict: Dict[str, Any], filename: str):
        self.config_dict = config_dict
        self.filename = filename
        self.path = f"./temp_{filename}"
        
    def __enter__(self):
        with open(self.path, 'w') as f:
            yaml.dump(self.config_dict, f)
        return self.path
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if os.path.exists(self.path):
            os.remove(self.path)
            
def save_temp_configs(config_dict: Dict[str, Any], filename: str) -> str:
    """Save a configuration dictionary to a temporary YAML file."""
    temp_path = f"./temp_{filename}"
    with open(temp_path, 'w') as f:
        yaml.dump(config_dict, f)
    return temp_path

def run_evaluation(unified_config_path: str, output_path: str = None, max_retries: int = 3, retry_delay: int = 5, use_fallback_model: bool = False):
    """Run the complete evaluation pipeline using the unified configuration."""
    try:
        # Load the unified configuration
        unified_config = load_unified_config(unified_config_path)
        
        # Create temporary configuration files for each component
        evaluator_config = create_evaluator_config(unified_config)
        indexer_config = create_indexer_config(unified_config)
        rag_config = create_rag_config(unified_config)
        
        # Use context managers for temporary files
        with TempConfigFile(evaluator_config, "evaluator_config.yaml") as evaluator_config_path, \
             TempConfigFile(indexer_config, "indexer_config.yaml") as indexer_config_path, \
             TempConfigFile(rag_config, "rag_config.yaml") as rag_config_path:
            
            logger.info("Initializing RAG Evaluator...")
            evaluator = RAGEvaluator.from_config(evaluator_config_path)
            
            # Define a patched version of the call method
            def patched_call(self, indexer_config, rag_config):
                # Create a compatible indexer instead of the standard one
                if isinstance(indexer_config, str):
                    config = load_config(indexer_config, IndexerConfig)
                else:
                    config = indexer_config
                
                # Create the milvus client
                from pymilvus import MilvusClient
                milvus_client = MilvusClient(
                    config.db.uri,
                    db_name=config.db.name,
                    enable_sparse=True,
                )
                
                # Create our compatible indexer
                indexer = CompatibleIndexer(
                    dense_model_config=config.dense_model,
                    sparse_model_config=config.sparse_model,
                    client=milvus_client,
                )
                
                # Continue with the original method but skip the indexer creation
                queries = self.dataset["user_input"]
                query_ids = self.dataset["query_ids"]
                rag_outputs = []
                
                # Sanitize the collection name - replace all non-alphanumeric chars with underscores
                import re
                collection_name = re.sub(r'[^a-zA-Z0-9_]', '_', indexer.dense_model_name)
                if not indexer.client.has_collection(collection_name):
                    for i, documents in enumerate(self.dataset["corpus"]):
                        print('Creating the indexer...')
                        indexer.index_documents([MultimodalSample(doc, modalities=[]) for doc in documents],
                                              collection_name=collection_name,
                                              partition_name=str(query_ids[i]))
                
                # Create RAG pipeline
                from src.mmore.rag.pipeline import RAGPipeline
                rag = RAGPipeline.from_config(rag_config)
                
                # Generate RAG outputs
                for i, query in enumerate(queries):
                    query_id = query_ids[i]
                    rag_outputs.append(rag(queries={'input': query, 'collection_name': collection_name, 'partition_name': str(query_id)},
                                       return_dict=True)[0])
               
                # Update the dataset with RAG outputs
                eval_dataset = self._get_eval_dataset(rag_outputs)
                
                from ragas import evaluate, EvaluationDataset
                return evaluate(
                    dataset=EvaluationDataset.from_hf_dataset(eval_dataset),
                    metrics=self.metrics,
                    llm=self.evaluator_llm,
                    embeddings=self.embeddings,
                    batch_size=4,
                ).to_pandas()
        
        # Apply the patch
        RAGEvaluator.__call__ = patched_call
        
        logger.info("Running evaluation...")
        result = None
        retry_count = 0
        last_error = None
        
        while retry_count <= max_retries:
            try:
                if retry_count > 0:
                    logger.info(f"Retry attempt {retry_count}/{max_retries}...")
                    time.sleep(retry_delay)
                
                result = evaluator(
                    indexer_config=indexer_config_path,
                    rag_config=rag_config_path
                )
                # If we get here, the evaluation was successful
                break
            except Exception as e:
                last_error = e
                error_str = str(e)
                logger.warning(f"Error during evaluation attempt {retry_count + 1}/{max_retries + 1}: {error_str}")
                
                if 'openai.APIConnectionError' in error_str or 'RateLimitError' in error_str or 'Connection error' in error_str:
                    logger.warning("OpenAI API connection error or rate limit exceeded.")
                    if use_fallback_model and retry_count >= max_retries - 1:
                        logger.info("Switching to local fallback model...")
                        # Modify the evaluator configuration to use a local model
                        try:
                            from langchain_community.llms import HuggingFacePipeline
                            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
                            
                            logger.info("Loading local model: google/flan-t5-large")
                            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
                            model = AutoModelForCausalLM.from_pretrained("google/flan-t5-large", device_map="auto")
                            pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
                            local_llm = HuggingFacePipeline(pipeline=pipe)
                            
                            # Replace the OpenAI LLM with our local model
                            evaluator.evaluator_llm = local_llm
                            logger.info("Successfully switched to local model")
                            retry_count += 1
                        except Exception as fallback_error:
                            logger.error(f"Failed to load fallback model: {fallback_error}")
                            retry_count += 1
                    else:
                        logger.warning("Retrying with OpenAI API...")
                        retry_count += 1
                else:
                    # For other errors, don't retry
                    logger.error(f"Non-retryable error: {error_str}")
                    break
        
        if result is None:
            logger.error(f"Failed after {retry_count} retries: {str(last_error)}")
            if last_error:
                raise last_error
        
        # Restore the original method
        RAGEvaluator.__call__ = original_call
        
        # Format results for better readability in YAML
        formatted_results = {}
        
        # Extract metrics and their values
        if hasattr(result, 'mean') and callable(result.mean):
            means = result.mean(numeric_only=True)
            formatted_results['metrics'] = {}
            for metric, score in means.items():
                formatted_results['metrics'][metric] = float(f"{score:.4f}")
            
            # Add detailed results by query
            formatted_results['detailed_results'] = []
            for idx, row in result.iterrows():
                query_result = {
                    'query': row.get('user_input', ''),
                    'response': row.get('response', ''),
                    'reference': row.get('reference', ''),
                    'scores': {}
                }
                
                # Add all numeric scores for this query
                for col in result.columns:
                    if col not in ['user_input', 'response', 'reference', 'retrieved_contexts'] and isinstance(row[col], (int, float)):
                        query_result['scores'][col] = float(f"{row[col]:.4f}")
                
                formatted_results['detailed_results'].append(query_result)
        else:
            # If it's not a DataFrame, just use the raw result
            formatted_results = result
        
        # Set default output path if not provided
        if not output_path:
            config_dir = os.path.dirname(unified_config_path)
            config_name = os.path.splitext(os.path.basename(unified_config_path))[0]
            output_path = os.path.join(config_dir, f"{config_name}_results.yaml")
        
        # Save results to file
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(formatted_results, f, default_flow_style=False, sort_keys=False)
        logger.info(f"Evaluation results saved to: {output_path}")
        
        logger.info("Evaluation Results:")
        # Handle different result types (pandas DataFrame, Series, or dict)
        if hasattr(result, 'to_dict'):
            # If it's a pandas DataFrame
            if hasattr(result, 'mean') and callable(result.mean):
                try:
                    # Calculate mean for each metric across all examples
                    means = result.mean(numeric_only=True)
                    for metric, score in means.items():
                        logger.info(f"{metric}: {score:.4f} (mean)")
                    
                    # Also show the full DataFrame in a readable format
                    logger.info("\nDetailed results by query:")
                    logger.info(result.to_string())
                except Exception as e:
                    logger.warning(f"Could not calculate means: {e}")
                    logger.info(str(result))
            elif hasattr(result, 'items'):
                # It's likely a Series or dict-like object
                try:
                    for metric, score in result.items():
                        if isinstance(score, (int, float)):
                            logger.info(f"{metric}: {score:.4f}")
                        elif hasattr(score, 'shape') and score.shape and score.shape[0] > 1:
                            # It's an array with multiple values
                            avg_score = score.mean() if hasattr(score, 'mean') else sum(score) / len(score)
                            logger.info(f"{metric}: {avg_score:.4f} (average of {score.shape[0]} values)")
                        elif hasattr(score, 'item'):
                            # It's a single-value array or scalar
                            try:
                                score_value = score.item()
                                logger.info(f"{metric}: {score_value:.4f}")
                            except ValueError:
                                # If item() fails, just use the score as is
                                logger.info(f"{metric}: {score}")
                        else:
                            # Just a regular value
                            logger.info(f"{metric}: {score}")
                except Exception as e:
                    logger.warning(f"Could not format result: {e}")
                    logger.info(str(result))

            else:
                # If result is not a dict-like object, just print it
                logger.info(str(result))
                
            return result
    
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if max_retries > 0:
            logger.info(f"Retrying in {retry_delay} seconds... ({max_retries} retries left)")
            time.sleep(retry_delay)
            return run_evaluation(unified_config_path, output_path, max_retries - 1, retry_delay, use_fallback_model)
        elif use_fallback_model and "OpenAI API" in str(e):
            logger.info("Attempting to use local fallback model...")
            # Modify config to use a local model
            unified_config = load_unified_config(unified_config_path)
            unified_config["evaluator_llm"]["llm_name"] = "llama3"
            return run_evaluation(unified_config_path, output_path, 0, retry_delay, False)
        else:
            raise

def evaluation(config_file: str, output_path: str = None, max_retries: int = 3, retry_delay: int = 5, use_fallback_model: bool = False):
    """Main entry point for the evaluation process."""
    return run_evaluation(config_file, output_path, max_retries, retry_delay, use_fallback_model)

def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation with a unified configuration file")
    parser.add_argument(
        "--config-file", 
        type=str, 
        required=True,
        help="Path to the unified evaluation configuration file"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        help="Path to save evaluation results (optional)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for API calls (default: 3)"
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=5,
        help="Delay between retries in seconds (default: 5)"
    )
    parser.add_argument(
        "--use-fallback-model",
        action="store_true",
        help="Use a local fallback model if OpenAI API fails"
    )
    
    args = parser.parse_args()
    evaluation(args.config_file, args.output, args.max_retries, args.retry_delay, args.use_fallback_model)

if __name__ == "__main__":
    main()

from datasets import Dataset, load_dataset, load_from_disk
from ragas import evaluate, EvaluationDataset
from ragas.embeddings import BaseRagasEmbeddings
from ragas.llms import BaseRagasLLM
from ragas.metrics.base import Metric
from langchain_huggingface import HuggingFaceEmbeddings
from src.mmore.rag.pipeline import RAGPipeline, RAGConfig
from src.mmore.index.indexer import IndexerConfig, Indexer, DBConfig
from src.mmore.rag.retriever import RetrieverConfig
from src.mmore.rag.llm import LLM, LLMConfig
from src.mmore.type import MultimodalSample
from src.mmore.utils import load_config
from typing import Union, List
from dataclasses import dataclass


@dataclass
class EvalConfig:
    """RAG Eval Configuration"""
    metrics: Union[str, List[str]]
    evaluator_name: str
    embeddings_name: str
    uri: str
    hf_dataset_name: str
    hf_feature_map: dict
    split: str


class RAGEvaluator:
    dataset: Dataset
    metrics: Metric | List[Metric]
    evaluator_llm: BaseRagasLLM
    embeddings: BaseRagasEmbeddings
    db_config: DBConfig

    def __init__(self, dataset, metrics, evaluator_llm, embeddings, db_config):
        self.dataset = dataset
        self.metrics = metrics
        self.evaluator_llm = evaluator_llm
        self.embeddings = embeddings
        self.db_config = db_config

    @classmethod
    def from_config(cls, config: str | EvalConfig):
        if isinstance(config, str):
            config = load_config(config, EvalConfig)
        # Load and prepare the dataset
        hf_dataset = load_dataset(config.hf_dataset_name, split=config.split)
        dataset = hf_dataset.rename_columns(config.hf_feature_map)

        # Add 'retrieved_contexts' and 'response' as empty fields
        dataset = dataset.map(lambda x: {"retrieved_contexts": [], "response": []})

        # Define the Milvus configuration
        db_config = DBConfig(uri=config.uri)

        # Parse and store metrics
        metrics = RAGEvaluator._parse_metrics(config.metrics)

        # Define evaluator LLM and embeddings
        evaluator_llm_config = LLMConfig(
            llm_name=config.evaluator_name,
            max_new_tokens=150,
            temperature=0.8
        )
        evaluator_llm = LLM.from_config(evaluator_llm_config)
        embeddings = HuggingFaceEmbeddings(model_name=config.embeddings_name)

        return cls(dataset, metrics, evaluator_llm, embeddings, db_config)

    @staticmethod
    def _parse_metrics(metrics: List):
        if not isinstance(metrics, list):
            raise TypeError("The 'metrics' parameter must be a list of metric instances.")

        parsed_metrics = []
        for metric in metrics:
            if callable(metric):
                parsed_metrics.append(metric())
            else:
                raise ValueError(f"Invalid metric provided: {metric}. Each metric must be callable.")

        return parsed_metrics

    def _get_eval_dataset(self, outputs: List[dict]) -> Dataset:
        """
        Update the dataset with 'response' and 'retrieved_contexts'.
        """

        def add_outputs_to_record(record, output):
            record["response"] = output["answer"]
            record["retrieved_contexts"] = [output["context"]]
            return record

        updated_dataset = self.dataset.map(
            lambda record, idx: add_outputs_to_record(record, outputs[idx]),
            with_indices=True
        )

        return updated_dataset

    def __call__(self, llm: str, dense: str, sparse: str, k: int):
        queries = self.dataset["user_input"]
        query_ids = self.dataset["query_ids"]
        rag_outputs = []

        # Indexing logic
        indexer_config = IndexerConfig(dense_model_name=dense, sparse_model_name=sparse, db=self.db_config)
        indexer = Indexer.from_config(indexer_config)

        collection_name = dense.replace("-", "_")
        if not indexer.client.has_collection(collection_name):
            for i, documents in enumerate(self.dataset["corpus"]):
                print('Creating the indexer...')
                indexer.index_documents([MultimodalSample(doc, modalities=[]) for doc in documents],
                                        collection_name=collection_name, partition_name=str(query_ids[i]))
                print("Indexer created.")

        # RAG initialization
        retriever_config = RetrieverConfig(db=self.db_config, k=k)
        llm_config = LLMConfig(llm_name=llm, max_new_tokens=150, temperature=0.8)
        rag = RAGPipeline.from_config(RAGConfig(retriever_config, llm_config))

        # Generate RAG outputs
        for i, query in enumerate(queries):
            query_id = query_ids[i]
            rag_outputs.append(rag(queries={'input': query, 'collection_name': collection_name, 'partition_name': str(query_id)},
                                   return_dict=True)[0])

        # Update the dataset with RAG outputs
        eval_dataset = self._get_eval_dataset(rag_outputs)

        return evaluate(
            dataset=EvaluationDataset.from_hf_dataset(eval_dataset),
            metrics=self.metrics,
            llm=self.evaluator_llm,
            embeddings=self.embeddings,
            batch_size=4,
        ).to_pandas()

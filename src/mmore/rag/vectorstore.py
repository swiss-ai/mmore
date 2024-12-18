"""
NOT USING THIS! Maybe a future update.
-----
Simple vector database indexer using Milvus for document storage.
Supports multimodal documents with chunking capabilities.
"""

from abc import ABC, abstractmethod

from typing import List, Any, Optional, Literal
from dataclasses import dataclass
from collections import deque
import numpy as np
import uuid

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores.base import VectorStore, VectorStoreRetriever
# from .models import get_model_wrapper, MultimodalModelWrapper
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from .model.dense.multimodal import MultimodalEmbeddings
from .model.sparse.splade import SpladeSparseEmbedding
from langchain_milvus.utils.sparse import BaseSparseEmbedding, BM25SparseEmbedding

from langchain_milvus import Milvus

from type import MultimodalSample
import collections
import sys
import nltk
from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema, Collection, AnnSearchRequest, model
import torch
from tqdm import tqdm


@dataclass
class VectorStoreConfig:
    dense_model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'
    sparse_model_name: str = "splade"
    collection_name: str = 'rag'
    milvus_uri: str = 'milvus_demo.db'  # "http://localhost:19530" Milvus standalone docker service


class End2EndVectorStore(ABC):
    vector_store: VectorStore

    def __init__(self, vector_store):
        self.vector_store = vector_store

    @abstractmethod
    @classmethod
    @abstractmethod
    def add_documents(self, documents: list[MultimodalSample], **kwargs: Any) -> list[str]:
        pass


class End2EndVectorStoreMilvus:
    milvus: Milvus

    def __init__(self, milvus) -> None:
        self.milvus = milvus

    @classmethod
    def from_config(cls, config: VectorStoreConfig):
        # Get models       
        dense_model = End2EndVectorStore._init_dense_model(config.dense_model_name)
        sparse_model = End2EndVectorStore._init_sparse_model(config.sparse_model_name)

        # Instatiate the VectorStore
        milvus = Milvus(
            embedding_function=dense_model,
            # vector_field=['dense', 'sparse'],
            collection_name=config.collection_name,
            connection_args={"uri": config.milvus_uri},
            auto_id=True
        )

        return cls(milvus=milvus)

    @classmethod
    def from_documents(cls, documents: List[MultimodalSample], config: VectorStoreConfig = VectorStoreConfig()):
        # Get models       
        dense_model = End2EndVectorStore._init_dense_model(config.dense_model_name)
        sparse_model = End2EndVectorStore._init_sparse_model(config.sparse_model_name)

        # Translate to multimodal embedder input
        texts = [MultimodalEmbeddings._multimodal_to_text(doc) for doc in documents]
        # metadatas = [doc.metadata for doc in documents]
        metadatas = [{'type': i} for i, doc in enumerate(documents)]

        milvus = Milvus.from_texts(
            texts,
            metadatas=metadatas,
            embedding=dense_model,
            # vector_field=['dense', 'sparse'],
            collection_name=config.collection_name,
            connection_args={"uri": config.milvus_uri}
        )

        return cls(milvus=milvus)

    def add_documents(self, documents: list[MultimodalSample], **kwargs: Any) -> list[str]:
        docs = [MultimodalEmbeddings._multimodal_to_doc(sample) for sample in documents]
        return self.milvus.add_documents(docs, **kwargs)

    def as_retriever(self, **kwargs: Any) -> VectorStoreRetriever:
        print(kwargs)
        return self.milvus.as_retriever(**kwargs)

    @staticmethod
    def _init_dense_model(dense_model_name: str) -> Embeddings:
        if dense_model_name == 'meta-llama/Llama-3.2-11B-Vision':
            return MultimodalEmbeddings(model_name=dense_model_name)
        else:
            return HuggingFaceEmbeddings(model_name=dense_model_name)

    @staticmethod
    def _init_sparse_model(sparse_model_name: str, corpus: List[str] = None) -> BaseSparseEmbedding:
        if sparse_model_name.lower() == 'bm25':
            return NotImplementedError()
            # return BM25SparseEmbedding(corpus)
        else:
            sparse_model_name = "naver/splade-cocondenser-selfdistil" if sparse_model_name == 'splade' else sparse_model_name
            return SpladeSparseEmbedding(sparse_model_name)

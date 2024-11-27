import os
# TODO: REMOVE WHEN .toml IS BUILT
import sys
sys.path.insert(1, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import argparse

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from src.mmore.type import MultimodalSample

from src.rag.retriever import Retriever, RetrieverConfig
from src.rag.indexer import Indexer, IndexerConfig, MilvusConfig
#from src.rag.vectorstore import End2EndVectorStore, VectorStoreConfig

from langchain_core.documents import Document
import random
import sys 

from dotenv import load_dotenv
load_dotenv()

EXAMPLE_MED_DOCS = [
    "Patient was discharged in stable condition following a three-day hospitalization for acute bronchitis, with instructions to continue antibiotics and follow up in one week.",
    "Chest X-ray reveals no acute abnormalities but shows mild chronic interstitial lung changes consistent with prior reports.",
    "The patient has been informed about the potential risks and benefits of laparoscopic cholecystectomy and has provided verbal and written consent.",
    "Biopsy of the gastric mucosa confirms the presence of Helicobacter pylori with associated chronic gastritis.",
    "Prescribe amoxicillin 500 mg orally three times a day for 7 days for treatment of bacterial sinusitis.",
    "Patient is referred to endocrinology for further evaluation of persistent hyperthyroidism noted on recent lab results.",
    "Complete blood count (CBC) shows mild anemia with a hemoglobin level of 10.5 g/dL.",
    "The procedure was completed successfully with minimal blood loss and no immediate complications.",
    "Recommend physical therapy twice weekly for six weeks to improve range of motion and strength in the right shoulder.",
    "Patient received the updated influenza vaccine on September 12, 2024, without any adverse reactions."
]

# Test queries as questions
EXAMPLE_QUERIES = [
    "Was the patient discharged?",
    "Are there any chest X-ray findings?",
    "Is there a surgery consent mentioned?",
    "What are the biopsy results?",
    "Is there a prescription for antibiotics?",
    "Is there a referral to endocrinology?",
    "What are the blood count results?",
    "What was the outcome of the procedure?",
    "Is there a plan for physical therapy?",
    "Is there any immunization record?"
]

COLLECTION_NAME = "medical_test"
URI = './db/medical_test.db'
MILVUS_CONFIG = MilvusConfig(uri=URI, collection_name=COLLECTION_NAME)

def get_args():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Run RAG pipeline with specified LLM, dense, and sparse models.')
    parser.add_argument('--llm', type=str, default='HuggingFaceTB/SmolLM2-1.7B', help='Name of the language model (LLM)')
    parser.add_argument('--dense', type=str, default='sentence-transformers/all-MiniLM-L6-v2', help='Name of the dense retrieval model')
    parser.add_argument('--sparse', type=str, default='splade', help='Name of the sparse retrieval model')

    # Parse the arguments
    return parser.parse_args()

if __name__ == "__main__":
    # Parse args
    args = get_args()

    # Assign arguments to variables
    dense_model = args.dense
    sparse_model = args.sparse
    llm = args.llm
    
    # Create an indexer
    if not os.path.exists(URI):
        print('Creating the indexer...')
        documents=[MultimodalSample(text=d, modalities=[]) for d in EXAMPLE_MED_DOCS]
        indexer_config = IndexerConfig(dense_model_name=dense_model, sparse_model_name=sparse_model, milvus_config=MILVUS_CONFIG)
        indexer = Indexer.from_documents(indexer_config, documents)
        print("Indexer created.")

    config = RetrieverConfig(
        model_name=dense_model, 
        sparse_model_name=sparse_model,
        milvus_config=MILVUS_CONFIG,
        k=3
    )

    retriever = Retriever.from_config(config)

    query = random.choice(EXAMPLE_QUERIES)

    print('Query:')
    print(query)
    print()

    dense_results = retriever.retrieve(
        query=query,
        k=1,
        search_type="dense"
    )

    sparse_results = retriever.retrieve(
        query=query,
        k=1,
        search_type="sparse"
    )

    hybrid_results = retriever.retrieve(
        query=query,
        k=1,
        search_type="hybrid"
    )

    print("Dense Results:")
    print(dense_results)
    print("\nSparse Results:")
    print(sparse_results)
    print("\nHybrid Results:")
    print(hybrid_results)
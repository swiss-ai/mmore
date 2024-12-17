import os

import random
import argparse

import pandas as pd

from src.mmore.rag.pipeline import RAGPipeline
from src.mmore.type import MultimodalSample, MultimodalRawInput, FileDescriptor
from src.mmore.index.indexer import Indexer

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

MOCK_FD = FileDescriptor(file_path='/path/to/doc', file_name='meddoc', file_size=1, created_at='EPFL', modified_at='ETHZ', file_extension='.txt')
MOCK_MODALITIES = [MultimodalRawInput('image', './image.png')]
EXAMPLE_MED_SAMPLES = [MultimodalSample(text=d, modalities=MOCK_MODALITIES, metadata=MOCK_FD) for d in EXAMPLE_MED_DOCS]

# Sample queries as questions
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

MOCK_INDEXER_CONFIG = './examples/index/indexer_example.yaml'
MOCK_RAG_CONFIG = './examples/rag/rag_example.yaml'

def get_args():
    # Create argument parser
    parser = argparse.ArgumentParser(description='Run RAG pipeline with specified LLM, dense, and sparse models.')
    parser.add_argument('--query', type=str, default=None)
    parser.add_argument('--indexer-config', type=str, default=MOCK_INDEXER_CONFIG, help='Path to an indexer config file.')
    parser.add_argument('--rag-config', type=str, default=MOCK_RAG_CONFIG, help='Path to a rag config file.')

    # Parse the arguments
    return parser.parse_args()

if __name__ == "__main__":
    # get script args
    args = get_args()

    # # Create an indexer
    # print('Indexing documents...')
    # indexer = Indexer.from_documents(
    #     args.indexer_config, 
    #     documents=EXAMPLE_MED_SAMPLES,
    #     collection_name='med_docs',
    # )
    # print("Indexer created.")

    # Initialize pipeline
    print('Creating the RAG Pipeline...')
    rag = RAGPipeline.from_config(args.rag_config)
    print('RAG pipeline initialized!')

    #print(f'> RAG Config:\n{config}')
    
    # Run query
    # result = rag({
    #     'input': args.query if args.query else random.choice(EXAMPLE_QUERIES), 
    #     'collection_name': 'med_docs',
    #     }, return_dict=True)[0]

    result = rag({
            'input': "Using the image, What is the indicator with the highest average weight in the “Projet de Kaduna : Durabilité au niveau LGA” chart, and what is its average weight?",
            'image_path': './examples/outputs/images/tmpo6ykxmno.png',  # Provide the path to the image
            'collection_name': 'med_docs',
        }, return_dict=True)[0]

    print('-'*50, '\n')
    print(f"> Query:\n{result['input']}\n")
    print(f"> Retrieved:\n{result['docs']}\n")
    print(f"> Chat:\n{result['answer']}")
    print('\n', '-'*50)
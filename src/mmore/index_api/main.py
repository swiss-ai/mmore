from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import List
import logging
import os
import tempfile
from pathlib import Path
import pdb
import shutil
from pymilvus import MilvusClient

from src.mmore.process.crawler import Crawler, CrawlerConfig
from src.mmore.process.dispatcher import Dispatcher, DispatcherConfig
from src.mmore.index.indexer import Indexer, IndexerConfig, DBConfig, get_model_from_index
from src.mmore.rag.model import DenseModelConfig, SparseModelConfig
from src.mmore.type import MultimodalSample
from src.mmore.index_api.models import CreateIndexRequest, IndexerResponse 

app = FastAPI(title = "Indexer API")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache indexers in memory
indexers = {}

@app.get("/")
async def root():
    return {"message": "Indexer API is running (what are you doing here...go catch it!)"}

#TODO: Find a way to deal with already existing collection files (delete? error message?)
#TODO: Add better function definition
#TODO: The add document function should take in the db and uri

@app.post("/indexers", response_model = IndexerResponse)
def create_indexer(request: CreateIndexRequest):
    try:
        logging.info(f"Received request to create new indexer with name: {request.collection_name}")

        # Creating the configuration object from the request body
        dense_config = DenseModelConfig(
            model_name = request.config.dense_model_name,
            is_multimodal= request.config.is_multimodal)

        sparse_config = SparseModelConfig(
            model_name = request.config.sparse_model_name,
            is_multimodal = request.config.is_multimodal
        )

        db_config = DBConfig(
            uri = request.config.db_uri,
            name = request.config.db_name
        )

        # Now let's create the indexer
        logging.info(f"Creating IndexerConfig")
        config = IndexerConfig(
            dense_model = dense_config,
            sparse_model = sparse_config,
            db = db_config
        )

        # Convert the document_path into MultimodalSample objects

        logging.info(f"Converting document_paths to MultimodalSample")
        documents = MultimodalSample.from_jsonl(request.document_paths)
        document_ids = [doc.id for doc in documents]

        # Now we can create the Indexer
        indexer = Indexer.from_documents(
            config = config, 
            documents = documents,
            collection_name = request.collection_name,
            partition_name = request.partition_name,
            batch_size = request.batch_size
            )

        # Store the indexer
        indexers[request.collection_name] = indexer

        return IndexerResponse(
            status='success',
            message= f"Indexer created and {len(documents)} documents indexed",
            documents_indexed = len(documents),
            collection_name = request.collection_name,
            id_list = document_ids
        )
    except Exception as e:
        raise HTTPException(status_code = 500, detail=str(e))


@app.post("/indexers/{collection_name}", response_model = IndexerResponse)
def add_documents(collection_name: str, uri: str, db_name: str, files: List[UploadFile]):
    """ Add documents to an existing collection """
    try: 
        
        logging.info("Recieved additional documents")

        # Create a temporary directory to store the additional documents to process
        with tempfile.TemporaryDirectory() as temp_dir:
            for file in files:
                file_name = Path(temp_dir) / file.filename
                with file_name.open('wb') as buffer:
                    shutil.copyfileobj(file.file, buffer)
        
            logging.info("Processing the documents")
            documents  = process_files(temp_dir, collection_name)[0] 
            
            document_ids  = [docs.id for docs in documents]# for doc in docs]
            logging.info(f"Retreive indexer {collection_name}")
            indexer = get_indexer(collection_name, uri, db_name)

            print("Indexing documents")
            # pdb.set_trace()
            indexer.index_documents(
                documents = documents, # Could this be the wrong format
                collection_name=collection_name
            # partition_name=request.partition_name
            )

        return IndexerResponse(
            status='success',
            message= f"Indexer created and {len(documents)} documents indexed",
            documents_indexed = len(documents),
            collection_name = collection_name,
            id_list = document_ids
        )
    except Exception as e:
        raise HTTPException(status_code = 500, detail=str(e))

# Helper functions

def get_indexer(collection_name: str, uri: str, db_name: str) -> Indexer:
    """ Get an existing indexer in cached Dict or load from the collection """
    if collection_name in indexers:
        return indexers[collection_name]
    
    try:
        # pdb.set_trace()
        client = MilvusClient(
            uri = uri, # endpoint for file based storage 
            db_name = db_name, # database name
            enable_sparse=True
        )

        # Get model configs from the collection
        dense_config = get_model_from_index(client, "dense_embedding", collection_name)
        sparse_config = get_model_from_index(client, "sparse_embedding", collection_name)
        
        # Create and store the indexer
        indexer = Indexer(
            dense_model_config=dense_config,
            sparse_model_config=sparse_config,
            client=client
        )
        indexers[collection_name] = indexer
        
        return indexer
    except Exception as e:
        raise HTTPException(
            status_code=404, 
            detail=f"Collection {collection_name} not found or could not be loaded: {str(e)}"
        )

# Note: I am setting aside all the config options for both the crawler and dispatcher and assuming 
#       that when we upload additional files there are very few (thus no distributed or fast options)
def process_files(temp_dir: str, collection_name: str) -> List[MultimodalSample]:
    output_path = f"./tmp/{collection_name}"    
    crawler_config = CrawlerConfig(
    root_dirs=[temp_dir],
    supported_extensions=[
            ".pdf", ".docx", ".pptx", ".md", ".txt",  # Document files
            ".xlsx", ".xls", ".csv",  # Spreadsheet files
            ".mp4", ".avi", ".mov", ".mkv",  # Video files
            ".mp3", ".wav", ".aac",  # Audio files
            ".eml", # Emails 
        ],
        output_path= output_path
    )
    crawler = Crawler(config=crawler_config)
    crawl_result = crawler.crawl()
    dispatcher_config = DispatcherConfig(
        output_path = output_path,
        use_fast_processors = False,
        extract_images = True
    )
    # pdb.set_trace()
    dispatcher = Dispatcher(result=crawl_result, config=dispatcher_config)
    return list(dispatcher())







# Debugging and testing  

@app.get("/test-milvus")
def test_milvus_connection():
    """ Tests that the Milvus database works and prints the collections created """
    try:
        from pymilvus import MilvusClient
        client = MilvusClient(uri="demo.db", db_name="my_db")
        collections = client.list_collections()
        return {"status": "success", "collections": collections}
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
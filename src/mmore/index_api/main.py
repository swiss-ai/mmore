from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import os

from mmore.index.indexer import Indexer, IndexerConfig, DBConfig, get_model_from_index
from mmore.rag.model import DenseModelConfig, SparseModelConfig
from mmore.type import MultimodalSample
from pymilvus import MilvusClient
from models import CreateIndexRequest, IndexerResponse, AddDocumentRequest

app = FastAPI(title = "Indexer API")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache indexers in memory
indexers = {}

@app.get("/")
async def root():
    return {"message": "Indexer API is running (what are you doing here...go catch it!)"}

#TODO: maybe a way to delete the vector_db files
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
        logging.info(f"Converting document_paths to MultimodalSample")

        # Convert the document_path into MultimodalSample objects
        documents = MultimodalSample.from_jsonl(request.document_paths)

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
            collection_name = request.collection_name
        )
    except Exception as e:
        raise HTTPException(status_code = 500, detail=str(e))

@app.post("/indexers/{collection_name}/documents", response_model = IndexerResponse)
def add_documents(collection_name: str, document_paths: str):
    """ Add documents to an existing collection """
    try: 
        logging.info(f"Retreive indexer {collection_name}")
        indexer = _get_indexer(collection_name)

        logging.info(f"Converting document_paths to MultimodalSample")
        documents = MultimodalSample.from_jsonl(document_paths)
        
        logging.info(f"Indexing documents")
        indexer.index_documents(
            documents,
            collection_name=collection_name
            # partition_name=request.partition_name
        )

        return IndexerResponse(
            status='success',
            message= f"Indexer created and {len(documents)} documents indexed",
            documents_indexed = len(documents),
            collection_name = collection_name
        )
    except Exception as e:
        raise HTTPException(status_code = 500, detail=str(e))


def _get_indexer(collection_name: str) -> Indexer:
    """ Get an existing indexer in cached Dict or load from the collection """
    if collection_name in indexers:
        return indexers[collection_name]
    
    try:
        # TODO: Maybe pass the names as parameters
        client = MilvusClient(
            uri=os.environ.get("MILVUS_URI", "./proc_demo.db"),
            db_name=os.environ.get("MILVUS_DB", "my_db"),
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
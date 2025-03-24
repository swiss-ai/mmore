from fastapi import FastAPI
from pydantic import BaseModel

from mmore.index.indexer import Indexer, IndexerConfig, DBConfig
from mmore.rag.model import DenseModelConfig, SparseModelConfig
from mmore.type import MultimodalSample

app = FastAPI("Indexer API")

@app.get("/")
async def root():
    return {"message": "Indexer API is running (hope you can catch it)"}

@app.post("/indexer")
def create_indexer():
    # Create config objects from request
    dense_model_config = DenseModelConfig(model)


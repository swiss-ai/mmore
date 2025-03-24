from typing import List, Dict, Optional
from pydantic import BaseModel 

#TODO: both embeddings should have a is multimodal?
class IndexerConfigSchema(BaseModel):
    dense_model_name: str
    sparse_model_name: str
    is_multimodal: bool = False
    db_uri: str = "demo.db"
    db_name: str = "my_db"
    
class CreateIndexRequest(BaseModel):
    """ Request body for creating a new index """
    config: IndexerConfigSchema
    collection_name: str = "my_docs"
    partition_name: str = None
    batch_size: int = 64
    document_paths: str 

class IndexerResponse(BaseModel):
    status: str
    message: str
    documents_indexed: int
    collection_name: str

# class AddDocumentRequest(BaseModel):
#     document_paths: str
#     partition_name: Optional[str] = None

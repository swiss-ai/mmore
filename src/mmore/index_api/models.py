from typing import List, Dict, Any, Optional
from pydantic import BaseModel

class IndexerConfigSchema(BaseModel):
    dense_model_name: str
    sparse_model_name: str
    is_mutlimodal: bool
    


class CreateIndexRequest(BaseModel):
    """Request body for creating an index"""
    config: IndexerConfigSchema
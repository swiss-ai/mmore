from typing import Dict, List

import torch
from langchain_milvus.utils.sparse import BaseSparseEmbedding


class SpladeSparseEmbedding(BaseSparseEmbedding):
    """Sparse embedding model based on Splade.

    This class uses the Splade embedding model in Milvus model to implement sparse vector embedding.
    This model requires pymilvus[model] to be installed.
    `pip install pymilvus[model]`
    For more information please refer to:
    https://milvus.io/docs/embed-with-splade.md
    """

    def __init__(self, model_name: str = "naver/splade-cocondenser-selfdistil"):
        from pymilvus.model.sparse import SpladeEmbeddingFunction  # type: ignore

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.splade = SpladeEmbeddingFunction(model_name=model_name, device=self.device)

    def embed_query(self, query: str) -> Dict[int, float]:
        res = self.splade.encode_queries([query])
        # res[0] because res has one row per query and there is only one query
        res_as_dict: Dict[int, float] = {
            k: v for k, v in zip(res[0].indices.tolist(), res[0].data.tolist())
        }
        return res_as_dict

    def embed_documents(self, texts: List[str]) -> List[Dict[int, float]]:
        res = self.splade.encode_documents(texts)
        res_as_dicts: List[Dict[int, float]] = [
            {k: v for k, v in zip(row.indices.tolist(), row.data.tolist())}
            for row in res
        ]
        return res_as_dicts

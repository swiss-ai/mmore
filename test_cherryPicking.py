# test_get_docs_by_ids.py

from mmore.index.indexer import DBConfig
from mmore.rag.retriever import Retriever, RetrieverConfig
import logging

from mmore.rag.model import DenseModel, SparseModel
import mmore.rag.retriever as retriever_mod

# Patch the retriever module's namespace so that DenseModel and SparseModel are defined.
retriever_mod.__dict__['DenseModel'] = DenseModel
retriever_mod.__dict__['SparseModel'] = SparseModel

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

def main():
    # Set up RetrieverConfig using your local Milvus DB configuration.
    retriever_config = RetrieverConfig(
        db=DBConfig(uri="demo.db", name="my_db"),
        hybrid_search_weight=0.5,
        k=1
    )
    
    # Initialize the retriever.
    retriever = Retriever.from_config(retriever_config)
    logger.info("Retriever loaded successfully.")
    
    # Specify the document IDs you wish to retrieve.
    # Replace these with valid IDs from your indexed documents.
    doc_ids = ["4992005992976948060", "-7269877291812006592"]
    logger.info(f"Retrieving documents with IDs: {doc_ids}")
    
    # Retrieve the documents by their IDs.
    docs = retriever.get_documents_by_ids(doc_ids, collection_name="my_docs")
    
    if docs:
        logger.info("Documents retrieved successfully:")
        for doc in docs:
            print("Document ID:", doc.metadata.get("id", "N/A"))
            print("Content:")
            print(doc.page_content)
            print("-" * 40)
    else:
        logger.warning("No documents were retrieved with the given IDs.")
    
if __name__ == "__main__":
    main()

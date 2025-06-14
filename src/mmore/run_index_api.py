import argparse
import logging
import os
import shutil
import tempfile
from pathlib import Path as FilePath
from typing import List, cast

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Path, UploadFile
from fastapi.responses import FileResponse
from pymilvus import MilvusClient

from mmore.index.indexer import DBConfig, Indexer, IndexerConfig, get_model_from_index
from mmore.process.crawler import Crawler, CrawlerConfig
from mmore.process.dispatcher import Dispatcher, DispatcherConfig
from mmore.rag.model import DenseModelConfig, SparseModelConfig
from mmore.rag.retriever import Retriever, RetrieverConfig
from mmore.run_retriever import RetrieverQuery
from mmore.type import MultimodalSample
from mmore.utils import load_config

MILVUS_URI: str = os.getenv("MILVUS_URI", "demo.db")
MILVUS_DB: str = os.getenv("MILVUS_DB", "my_db")
COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "my_documents")

UPLOAD_DIR: str = "./uploads"

# Create the upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="Indexer API")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache indexers in memory
indexers = {}
retrievers = {}


@app.get("/")
async def root():
    return {
        "message": "Indexer API is running (what are you doing here...go catch it!)"
    }


# SINGLE FILE UPLOAD ENDPOINT
@app.post("/v1/files", status_code=201, tags=["File Operations"])
async def upload_file(
    fileId: str = Form(..., description="Unique identifier for the file"),
    file: UploadFile = File(..., description="The file content"),
):
    """
    Upload a new file with a unique identifier.

    Requirements:
    - Unique fileId
    """
    try:
        # Check if file with this ID already exists
        # pdb.set_trace()
        file_storage_path = FilePath(UPLOAD_DIR) / fileId
        if file_storage_path.exists():
            raise HTTPException(
                status_code=400, detail=f"File with ID {fileId} already exists"
            )

        if file.filename is None:
            raise HTTPException(
                status_code=422, detail="Provided file should have a filename"
            )

        # Use a temporary directory for processing so that we only process the incoming docs
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_file_path = FilePath(temp_dir) / file.filename
            with temp_file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            await file.close()

            # Save a permanent copy for later retrieval
            os.makedirs(os.path.dirname(file_storage_path), exist_ok=True)
            shutil.copy2(temp_file_path, file_storage_path)

            # Process and index the file
            file_extension = FilePath(file.filename).suffix.lower()
            documents = process_files(temp_dir, COLLECTION_NAME, [file_extension])

            # Set the custom ID
            for doc in documents:
                doc.id = fileId

            # Get indexer and index the document
            indexer = get_indexer(COLLECTION_NAME, MILVUS_URI, MILVUS_DB)
            indexer.index_documents(
                documents=documents, collection_name=COLLECTION_NAME
            )
            indexer.client.flush(COLLECTION_NAME)

            return {
                "status": "success",
                "message": f"File successfully indexed in {COLLECTION_NAME} collection",
                "fileId": fileId,
                "filename": file.filename,
            }

    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/files/bulk", status_code=201, tags=["File Operations"])
async def upload_files(
    listIds: List[str] = Form(..., description="List of IDs for the files"),
    files: List[UploadFile] = File(..., description="Files to upload"),
):
    """
    Upload multiple files with custom IDs and index them.
    """
    try:
        listIds = listIds[0].split(",")
        # Check if IDs and files match in number
        if len(listIds) != len(files):
            raise HTTPException(
                status_code=400,
                detail=f"Number of IDs ({len(listIds)}) doesn't match number of files ({len(files)})",
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            logging.info(f"Starting to process {len(files)} files with custom IDs")

            for file, file_id in zip(files, listIds):
                if file.filename is None:
                    raise HTTPException(
                        status_code=422,
                        detail=f"File {file_id} does not have a filename",
                    )

                # Check if file with this ID already exists
                file_storage_path = FilePath(UPLOAD_DIR) / file_id
                if file_storage_path.exists():
                    raise HTTPException(
                        status_code=400, detail=f"File with ID {file_id} already exists"
                    )

                # Save to temp directory
                file_name = FilePath(temp_dir) / file.filename
                with file_name.open("wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)

                # Save a permanent copy
                shutil.copy2(file_name, file_storage_path)

                # Close the file
                await file.close()

            logging.info(f"Files saved to temporary directory: {temp_dir}")

            # Process the documents
            file_extensions = [
                FilePath(cast(str, file.filename)).suffix.lower() for file in files
            ]
            documents = process_files(temp_dir, COLLECTION_NAME, file_extensions)

            # Change the IDs to match the ones from the client
            modified_documents = []
            for doc, docId in zip(documents, listIds):
                doc.id = docId
                modified_documents.append(doc)

            logging.info("Indexing the files")

            indexer = get_indexer(COLLECTION_NAME, MILVUS_URI, MILVUS_DB)
            indexer.index_documents(
                documents=modified_documents, collection_name=COLLECTION_NAME
            )
            indexer.client.flush(COLLECTION_NAME)

            return {
                "status": "success",
                "message": f"Successfully processed and indexed {len(modified_documents)} documents",
                "documents": [
                    {"id": doc.id, "text": doc.text[:50] + "..."}
                    for doc in modified_documents
                ],
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/v1/files/{id}", tags=["File Operations"])
async def update_file(
    id: str = Path(..., description="ID of the file to update"),
    file: UploadFile = File(..., description="The new file content"),
):
    """
    Replace an existing file with a new version.
    """
    try:
        # Check if file exists
        file_storage_path = FilePath(UPLOAD_DIR) / id
        if not file_storage_path.exists():
            raise HTTPException(status_code=404, detail=f"File with ID {id} not found")

        if file.filename is None:
            raise HTTPException(
                status_code=422, detail="Provided file should have a filename"
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the file temporarily for processing
            temp_file_path = FilePath(temp_dir) / file.filename
            with temp_file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            await file.close()

            # Replace the existing file
            os.remove(file_storage_path)
            shutil.copy2(temp_file_path, file_storage_path)

            # Process and index the file
            file_extension = FilePath(file.filename).suffix.lower()
            documents = process_files(temp_dir, COLLECTION_NAME, [file_extension])

            # Set the custom ID
            for doc in documents:
                doc.id = id

            # Get indexer and reindex the document
            indexer = get_indexer(COLLECTION_NAME, MILVUS_URI, MILVUS_DB)

            # First delete the existing document
            try:
                # Delete existing document with this ID from collection
                client = MilvusClient(
                    uri=MILVUS_URI, db_name=MILVUS_DB, enable_sparse=True
                )
                client.delete(collection_name=COLLECTION_NAME, filter=f"id == '{id}'")
            except Exception as delete_error:
                logger.warning(
                    f"Error deleting existing document (may not exist): {str(delete_error)}"
                )

            # Index the new document
            indexer.index_documents(
                documents=documents, collection_name=COLLECTION_NAME
            )
            indexer.client.flush(COLLECTION_NAME)

            return {
                "status": "success",
                "message": "File successfully updated",
                "fileId": id,
                "filename": file.filename,
            }

    except Exception as e:
        logger.error(f"Error updating file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/v1/files/{id}", tags=["File Operations"])
async def delete_file(id: str = Path(..., description="ID of the file to delete")):
    """
    Delete a file from the system.

    """
    try:
        # Check if file exists
        file_storage_path = FilePath(UPLOAD_DIR) / id
        if not file_storage_path.exists():
            raise HTTPException(status_code=404, detail=f"File with ID {id} not found")

        # Delete the physical file
        os.remove(file_storage_path)

        # Delete from vector database
        try:
            client = MilvusClient(uri=MILVUS_URI, db_name=MILVUS_DB, enable_sparse=True)
            delete_result = client.delete(
                collection_name=COLLECTION_NAME, filter=f"id == '{id}'"
            )
            logger.info(f"Deleted document from vector DB: {delete_result}")
        except Exception as db_error:
            logger.warning(
                f"Error deleting from vector DB (continuing): {str(db_error)}"
            )

        return {
            "status": "success",
            "message": "File successfully deleted",
            "fileId": id,
        }

    except Exception as e:
        logger.error(f"Error deleting file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/files/{id}", tags=["File Operations"])
async def download_file(id: str = Path(..., description="ID of the file to download")):
    """
    Download a file from the system.
    """
    try:
        # Check if file exists
        file_storage_path = FilePath(UPLOAD_DIR) / id
        if not file_storage_path.exists():
            raise HTTPException(status_code=404, detail=f"File with ID {id} not found")

        # Determine the filename from metadata or use the ID
        filename = id  # Default to ID if we can't determine original name

        # Return the file
        return FileResponse(
            path=file_storage_path,
            filename=filename,
            media_type="application/octet-stream",
        )

    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/retriever", status_code=201, tags=["Context Retrieval"])
def retriever(query: RetrieverQuery) -> list[dict]:
    """Query the retriever"""

    retriever = get_retriever(MILVUS_URI, MILVUS_DB)

    try:
        docs_for_query = retriever.invoke(
            query.query,
            document_ids=query.fileIds,
            k=query.maxMatches,
            min_score=query.minSimilarity,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    docs_info = []
    for doc in docs_for_query:
        meta = doc.metadata
        docs_info.append(
            {
                "fileId": meta["id"],
                "content": doc.page_content,
                "similarity": meta["similarity"],
            }
        )

    return docs_info


# Helper functions


def create_new_indexer(collection_name: str, uri: str, db_name: str) -> Indexer:
    """Create a new indexer with default configuration"""
    try:
        # Default model configurations
        dense_config = DenseModelConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2", is_multimodal=False
        )

        sparse_config = SparseModelConfig(model_name="splade", is_multimodal=False)

        db_config = DBConfig(uri=uri, name=db_name)

        # Create indexer config
        config = IndexerConfig(
            dense_model=dense_config, sparse_model=sparse_config, db=db_config
        )

        # Create an empty list of documents for initialization
        empty_docs = []

        # Create indexer from documents (this will create the collection)
        indexer = Indexer.from_documents(
            config=config, documents=empty_docs, collection_name=collection_name
        )

        # Store in cache
        indexers[collection_name] = indexer

        logging.info(
            f"Successfully created new indexer for collection: {collection_name}"
        )
        return indexer
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to create new indexer: {str(e)}"
        )


def get_indexer(collection_name: str, uri: str, db_name: str) -> Indexer:
    """Get an existing indexer in cached Dict or load from the collection"""
    if collection_name in indexers:
        return indexers[collection_name]

    try:
        client = MilvusClient(uri=uri, db_name=db_name, enable_sparse=True)

        collections = client.list_collections()

        if collection_name not in collections:
            return create_new_indexer(collection_name, uri, db_name)

        # Get model configs from the collection
        dense_config = cast(
            DenseModelConfig,
            get_model_from_index(client, "dense_embedding", collection_name),
        )
        sparse_config = cast(
            SparseModelConfig,
            get_model_from_index(client, "sparse_embedding", collection_name),
        )

        # Create and store the indexer
        indexer = Indexer(
            dense_model_config=dense_config,
            sparse_model_config=sparse_config,
            client=client,
        )

        indexers[collection_name] = indexer

        return indexer
    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Collection {collection_name} not found or could not be loaded: {str(e)}",
        )


def get_retriever(uri: str, db_name: str) -> Retriever:
    """
    Get an existing retriever in cached Dict or load from the DB uri.

    Args:
      uri, the uri of the database file
      db_name, the name of the database (ignored if the uri is already associated with the retriever)

    Returns the corresponding Retriever object.
    """

    if uri not in retrievers:
        config = load_config({"db": {"uri": uri, "name": db_name}}, RetrieverConfig)
        retrievers[uri] = Retriever.from_config(config)

    return retrievers[uri]


def process_files(
    temp_dir: str,
    collection_name: str,
    extensions: List[str] = [
        ".pdf",
        ".docx",
        ".pptx",
        ".md",
        ".txt",
        ".xlsx",
        ".xls",
        ".csv",
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".mp3",
        ".wav",
        ".aac",
        ".eml",
    ],
) -> List[MultimodalSample]:
    output_path = f"./tmp/{collection_name}"
    crawler_config = CrawlerConfig(
        root_dirs=[temp_dir],
        # For more effecient processing give only the extensions needed
        supported_extensions=extensions,
        output_path=output_path,
    )
    crawler = Crawler(config=crawler_config)
    crawl_result = crawler.crawl()
    dispatcher_config = DispatcherConfig(
        output_path=output_path, use_fast_processors=False, extract_images=True
    )
    # pdb.set_trace()
    dispatcher = Dispatcher(result=crawl_result, config=dispatcher_config)
    return sum(list(dispatcher()), [])


def run_api(host: str, port: int):
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host on which the API should be run."
    )
    parser.add_argument(
        "--port", default=8000, help="Port on which the API should be run."
    )
    args = parser.parse_args()

    run_api(args.host, args.port)

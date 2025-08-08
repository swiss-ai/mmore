import argparse
import logging
import os
import shutil
import tempfile
from pathlib import Path as FilePath
from typing import List, cast

import uvicorn
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Path, UploadFile
from fastapi.responses import FileResponse
from pymilvus import MilvusClient

logger = logging.getLogger(__name__)
RETRIVER_EMOJI = "🗂️"
logging.basicConfig(
    format=f"[INDEX API {RETRIVER_EMOJI} -- %(asctime)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

from .process.processors import register_all_processors
from .rag.retriever import RetrieverConfig
from .utils import get_indexer, load_config, process_files_default

UPLOAD_DIR: str = "./uploads"

# Create the upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_router(config_path: str) -> APIRouter:
    router = APIRouter()

    config = load_config(config_path, RetrieverConfig)

    MILVUS_URI = config.db.uri or "./proc_demo.db"
    MILVUS_DB = config.db.name or "my_db"
    COLLECTION_NAME = config.collection_name or "my_docs"

    # Initialize the index database and the processors
    get_indexer(COLLECTION_NAME, MILVUS_URI, MILVUS_DB)
    register_all_processors(preload=True)

    @router.get("/")
    async def root():
        return {
            "message": "Indexer API is running (what are you doing here...go catch it!)"
        }

    # SINGLE FILE UPLOAD ENDPOINT
    @router.post("/v1/files", status_code=201, tags=["File Operations"])
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
                documents = process_files_default(
                    temp_dir, COLLECTION_NAME, [file_extension]
                )

                for doc in documents:
                    defDocId = doc.document_id
                    doc.document_id = fileId
                    doc.id = doc.id.replace(defDocId, fileId)

                # Get indexer and index the document
                try:
                    indexer = get_indexer(COLLECTION_NAME, MILVUS_URI, MILVUS_DB)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))

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

    @router.post("/v1/files/bulk", status_code=201, tags=["File Operations"])
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
                            status_code=400,
                            detail=f"File with ID {file_id} already exists",
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
                documents = process_files_default(
                    temp_dir, COLLECTION_NAME, file_extensions
                )

                # Change the IDs to match the ones from the client
                modified_documents = []
                for doc, docId in zip(documents, listIds):
                    defDocId = doc.document_id
                    doc.document_id = docId
                    doc.id = doc.id.replace(defDocId, docId)
                    modified_documents.append(doc)

                logging.info("Indexing the files")

                try:
                    indexer = get_indexer(COLLECTION_NAME, MILVUS_URI, MILVUS_DB)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))

                indexer.index_documents(
                    documents=modified_documents, collection_name=COLLECTION_NAME
                )
                indexer.client.flush(COLLECTION_NAME)

                return {
                    "status": "success",
                    "message": f"Successfully processed and indexed {len(modified_documents)} documents",
                    "documents": [
                        {"fileId": doc.document_id, "text": doc.text[:50] + "..."}
                        for doc in modified_documents
                    ],
                }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.put("/v1/files/{fileId}", tags=["File Operations"])
    async def update_file(
        fileId: str = Path(..., description="ID of the file to update"),
        file: UploadFile = File(..., description="The new file content"),
    ):
        """
        Replace an existing file with a new version.
        """
        try:
            # Check if file exists
            file_storage_path = FilePath(UPLOAD_DIR) / fileId
            if not file_storage_path.exists():
                raise HTTPException(
                    status_code=404, detail=f"File with ID {fileId} not found"
                )

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
                documents = process_files_default(
                    temp_dir, COLLECTION_NAME, [file_extension]
                )

                # Set the custom ID
                for doc in documents:
                    doc.id = fileId

                # Get indexer and reindex the document
                try:
                    indexer = get_indexer(COLLECTION_NAME, MILVUS_URI, MILVUS_DB)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))

                # First delete the existing document
                try:
                    # Delete existing document with this ID from collection
                    client = MilvusClient(
                        uri=MILVUS_URI, db_name=MILVUS_DB, enable_sparse=True
                    )
                    client.delete(
                        collection_name=COLLECTION_NAME,
                        filter=f"document_id == '{fileId}'",
                    )
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
                    "fileId": fileId,
                    "filename": file.filename,
                }

        except HTTPException as e:
            raise e

        except Exception as e:
            logger.error(f"Error updating file: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.delete("/v1/files/{fileId}", tags=["File Operations"])
    async def delete_file(
        fileId: str = Path(..., description="ID of the file to delete"),
    ):
        """
        Delete a file from the system.

        """
        try:
            # Check if file exists
            file_storage_path = FilePath(UPLOAD_DIR) / fileId
            if not file_storage_path.exists():
                raise HTTPException(
                    status_code=404, detail=f"File with ID {fileId} not found"
                )

            # Delete the physical file
            os.remove(file_storage_path)

            # Delete from vector database
            try:
                client = MilvusClient(
                    uri=MILVUS_URI, db_name=MILVUS_DB, enable_sparse=True
                )
                delete_result = client.delete(
                    collection_name=COLLECTION_NAME, filter=f"document_id == {fileId}"
                )
                logger.info(f"Deleted document from vector DB: {delete_result}")
            except Exception as db_error:
                logger.warning(
                    f"Error deleting from vector DB (continuing): {str(db_error)}"
                )

            return {
                "status": "success",
                "message": "File successfully deleted",
                "fileId": fileId,
            }

        except HTTPException as e:
            raise e

        except Exception as e:
            logger.error(f"Error deleting file: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/v1/files/{fileId}", tags=["File Operations"])
    async def download_file(
        fileId: str = Path(..., description="ID of the file to download"),
    ):
        """
        Download a file from the system.
        """
        try:
            # Check if file exists
            file_storage_path = FilePath(UPLOAD_DIR) / fileId
            if not file_storage_path.exists():
                raise HTTPException(
                    status_code=404, detail=f"File with ID {fileId} not found"
                )

            # Retrieve the filename from metadata
            try:
                client = MilvusClient(
                    uri=MILVUS_URI, db_name=MILVUS_DB, enable_sparse=True
                )
                file_paths = client.query(
                    collection_name=COLLECTION_NAME,
                    filter=f"document_id == '{fileId}'",
                    output_fields=["file_path"],
                )

                if len(file_paths) == 0:
                    raise ValueError(
                        f"Document of id {fileId} not found in the database"
                    )

                # all the elements with the same id refer to the same file so they have the same path
                file_path: str = file_paths[0]["file_path"]
                filename = file_path.split("/")[-1]
            except Exception as db_error:
                logger.warning(
                    f"Error deleting from vector DB (continuing): {str(db_error)}"
                )
                raise db_error

            # Return the file
            return FileResponse(
                path=file_storage_path,
                filename=filename,
                media_type="application/octet-stream",
            )

        except HTTPException as e:
            raise e

        except Exception as e:
            logger.error(f"Error downloading file: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    return router


def run_api(config_file: str, host: str, port: int):
    router = make_router(config_file)

    app = FastAPI(title="Indexer API")
    app.include_router(router)

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file", required=True, help="Path to the retriever configuration file."
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host on which the API should be run."
    )
    parser.add_argument(
        "--port", default=8000, help="Port on which the API should be run."
    )
    args = parser.parse_args()

    run_api(args.config_file, args.host, args.port)

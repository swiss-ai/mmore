import argparse
import asyncio
import json
import logging
import os
import shutil
import tempfile
import threading
from pathlib import Path as FilePath
from typing import Callable, List, Optional

import uvicorn
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, Path, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pymilvus import MilvusClient

logger = logging.getLogger(__name__)
RETRIVER_EMOJI = "🗂️"
logging.basicConfig(
    format=f"[INDEX API {RETRIVER_EMOJI} -- %(asctime)s] %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

from mmore.profiler import enable_profiling_from_env

from .job_queue import DuplicateJobError, Job, JobQueue, QueueFullError
from .process.processors import register_all_processors
from .rag.retriever import RetrieverConfig
from .type import MultimodalSample
from .utils import get_indexer, load_config, process_files_default

UPLOAD_DIR: str = "./uploads"

# Create the upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


POLL_INTERVAL = 2.0
HEARTBEAT_SECONDS = 15


def _job_payload(job: Job) -> dict:
    return {
        "jobId": job.id,
        "fileId": job.file_id,
        "filename": job.filename,
        "status": job.status.value,
        "device": job.device,
        "result": job.result,
        "error": job.error,
    }


def _apply_uploaded_file_metadata(
    documents: List[MultimodalSample], file_id: str, filename: str
) -> None:
    """Bind processed chunks to the API file ID and persist the original filename."""
    for doc in documents:
        chunk_id = doc.id.rsplit("+")[1] if "+" in doc.id else None
        doc.document_id = file_id
        doc.id = f"{file_id}+{chunk_id}" if chunk_id else file_id

        doc.metadata.extra["filename"] = filename


def make_router(config_path: str) -> APIRouter:
    router = APIRouter()

    config = load_config(config_path, RetrieverConfig)

    MILVUS_URI = config.db.uri or "./proc_demo.db"
    MILVUS_DB = config.db.name or "my_db"
    COLLECTION_NAME = config.collection_name or "my_docs"

    # Initialize the index database and the processors
    get_indexer(COLLECTION_NAME, MILVUS_URI, MILVUS_DB)
    register_all_processors(preload=True)

    jobs = JobQueue(
        jobs_per_gpu=getattr(config, "jobs_per_gpu", 1),
        max_queue_size=getattr(config, "max_queue_size", None),
    )

    db_lock = threading.Lock()

    def _stage_upload(file: UploadFile, filename: str) -> tuple[str, str]:
        """Save the uploaded bytes now, while the request is alive.

        Returns (job_dir, input_dir). The worker removes job_dir when done.
        """
        job_dir = tempfile.mkdtemp(prefix="index_job_")
        input_dir = os.path.join(job_dir, "input")
        os.makedirs(input_dir)
        with open(os.path.join(input_dir, filename), "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return job_dir, input_dir

    def _make_ingest_job(
        job_dir: str,
        input_dir: str,
        file_id: str,
        filename: str,
        replace: bool,
    ) -> Callable[[str], dict]:
        extension = FilePath(filename).suffix.lower()

        def ingest(device: str) -> dict:
            try:
                documents = process_files_default(
                    input_dir,
                    COLLECTION_NAME,
                    [extension],
                    device=device,
                    output_path=os.path.join(job_dir, "out"),
                )
                _apply_uploaded_file_metadata(documents, file_id, filename)

                indexer = get_indexer(COLLECTION_NAME, MILVUS_URI, MILVUS_DB)
                with db_lock:
                    if replace:
                        indexer.client.delete(
                            collection_name=COLLECTION_NAME,
                            filter=f"document_id == '{file_id}'",
                        )
                    indexer.index_documents(
                        documents=documents, collection_name=COLLECTION_NAME
                    )
                    indexer.client.flush(COLLECTION_NAME)

                # Persist the permanent copy only on success
                dest = FilePath(UPLOAD_DIR) / file_id
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                shutil.copy2(os.path.join(input_dir, filename), dest)
                return {"chunks": len(documents)}
            finally:
                shutil.rmtree(job_dir, ignore_errors=True)

        return ingest

    @router.get("/")
    async def root():
        return {
            "message": "Indexer API is running (what are you doing here...go catch it!)"
        }

    # SINGLE FILE UPLOAD ENDPOINT
    @router.post("/v1/files", status_code=202, tags=["File Operations"])
    async def upload_file(
        fileId: str = Form(..., description="Unique identifier for the file"),
        file: UploadFile = File(..., description="The file content"),
    ):
        """
        Queue a new file for processing and indexing.

        Returns a jobId immediately, the work runs in the background.
        """
        if file.filename is None:
            raise HTTPException(
                status_code=422, detail="Provided file should have a filename"
            )
        if (FilePath(UPLOAD_DIR) / fileId).exists():
            raise HTTPException(
                status_code=409, detail=f"File with ID {fileId} already exists"
            )

        job_dir, input_dir = _stage_upload(file, file.filename)
        await file.close()
        ingest = _make_ingest_job(
            job_dir, input_dir, fileId, file.filename, replace=False
        )
        try:
            job_id = jobs.submit(fileId, file.filename, ingest)
        except DuplicateJobError:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(
                status_code=409,
                detail=f"File with ID {fileId} is already being processed",
            )
        except QueueFullError:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(status_code=503, detail="Server busy, retry later")

        return {"jobId": job_id, "fileId": fileId}

    @router.post("/v1/files/bulk", status_code=202, tags=["File Operations"])
    async def upload_files(
        listIds: List[str] = Form(..., description="List of IDs for the files"),
        files: List[UploadFile] = File(..., description="Files to upload"),
    ):
        """
        Queue multiple files, one independent job per file.

        Returns a per-file outcome (jobId or error), so one bad file does not
        fail the whole batch.
        """
        listIds = [
            file_id.strip()
            for ids in listIds
            for file_id in ids.split(",")
            if file_id.strip()
        ]
        if len(listIds) != len(files):
            raise HTTPException(
                status_code=400,
                detail=f"Number of IDs ({len(listIds)}) doesn't match number of files ({len(files)})",
            )

        results = []
        for file, file_id in zip(files, listIds):
            if file.filename is None:
                results.append({"fileId": file_id, "error": "missing filename"})
                continue
            if (FilePath(UPLOAD_DIR) / file_id).exists():
                results.append({"fileId": file_id, "error": "already exists"})
                continue

            job_dir, input_dir = _stage_upload(file, file.filename)
            await file.close()
            ingest = _make_ingest_job(
                job_dir, input_dir, file_id, file.filename, replace=False
            )
            try:
                job_id = jobs.submit(file_id, file.filename, ingest)
                results.append({"fileId": file_id, "jobId": job_id})
            except DuplicateJobError:
                shutil.rmtree(job_dir, ignore_errors=True)
                results.append({"fileId": file_id, "error": "already being processed"})
            except QueueFullError:
                shutil.rmtree(job_dir, ignore_errors=True)
                results.append({"fileId": file_id, "error": "queue full"})

        return {"jobs": results}

    @router.put("/v1/files/{fileId}", status_code=202, tags=["File Operations"])
    async def update_file(
        fileId: str = Path(..., description="ID of the file to update"),
        file: UploadFile = File(..., description="The new file content"),
    ):
        """
        Queue a replacement for an existing file and re-index it.

        The old vectors are deleted and the new ones inserted only after
        processing succeeds, so a failure leaves the existing document intact.
        """
        if not (FilePath(UPLOAD_DIR) / fileId).exists():
            raise HTTPException(
                status_code=404, detail=f"File with ID {fileId} not found"
            )
        if file.filename is None:
            raise HTTPException(
                status_code=422, detail="Provided file should have a filename"
            )

        job_dir, input_dir = _stage_upload(file, file.filename)
        await file.close()
        ingest = _make_ingest_job(
            job_dir, input_dir, fileId, file.filename, replace=True
        )
        try:
            job_id = jobs.submit(fileId, file.filename, ingest)
        except DuplicateJobError:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(
                status_code=409,
                detail=f"File with ID {fileId} is already being processed",
            )
        except QueueFullError:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(status_code=503, detail="Server busy, retry later")

        return {"jobId": job_id, "fileId": fileId}

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
                    collection_name=COLLECTION_NAME, filter=f"document_id == '{fileId}'"
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

    @router.get("/v1/jobs/{jobId}", tags=["Jobs"])
    async def get_job(jobId: str = Path(..., description="ID of the job")):
        """One-shot job status snapshot, fallback for when the SSE stream drops."""
        job = jobs.get(jobId)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unknown job {jobId}")
        return _job_payload(job)

    @router.get("/v1/jobs/{jobId}/events", tags=["Jobs"])
    async def stream_job_events(jobId: str = Path(..., description="ID of the job")):
        """Push job status updates to the client over SSE until the job ends."""

        async def event_stream():
            last: Optional[str] = None
            idle = 0.0
            while True:
                job = jobs.get(jobId)
                status = job.status.value if job else "unknown"
                if status != last:
                    last = status
                    idle = 0.0
                    payload = (
                        _job_payload(job) if job else {"jobId": jobId, "status": status}
                    )
                    yield f"data: {json.dumps(payload)}\n\n"
                    if job is None or job.status.is_terminal:
                        return
                else:
                    idle += POLL_INTERVAL
                    if idle >= HEARTBEAT_SECONDS:
                        idle = 0.0
                        yield ": keepalive\n\n"
                await asyncio.sleep(POLL_INTERVAL)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return router


def run_api(config_file: str, host: str, port: int):
    router = make_router(config_file)

    app = FastAPI(title="Indexer API")
    app.include_router(router)

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    enable_profiling_from_env()
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

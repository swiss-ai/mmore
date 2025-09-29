import argparse
import os
from datetime import datetime
from typing import Any, Optional

import motor.motor_asyncio
import uvicorn
from fastapi import BackgroundTasks, FastAPI, Query
from pymongo import DESCENDING
from starlette.middleware.cors import CORSMiddleware

from mmore.dashboard.backend.model import (
    BatchedReports,
    DashboardMetadata,
    Progress,
    Report,
    WorkerLatest,
)

app = FastAPI()
# allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = motor.motor_asyncio.AsyncIOMotorClient(os.environ.get("MONGODB_URL"))
db = client.fastdbyeah
reports_collection = db.get_collection("reports")
dashboardmetadata_collection = db.get_collection("dashboardmetadata")


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str) -> Report:
    report = await reports_collection.find_one({"name": name})
    if not report:
        raise ValueError("Report not found")

    if "worker_id" not in report or "finished_file_paths" not in report:
        raise ValueError("Incomplete report data")

    return Report(**report)


async def insert_report_into_db(data: dict):
    """
    Inserts a student document into the database.
    """
    try:
        result = await reports_collection.insert_one(data)
        print(f"Inserted document with ID: {result.inserted_id}")
    except Exception as e:
        print(f"Error inserting document: {e}")


@app.post("/reports")
async def submit_report(report: Report, background_tasks: BackgroundTasks) -> Any:
    report.timestamp = datetime.now()
    dump: dict[str, Any] = report.model_dump(by_alias=True, exclude={"id"})
    print(dump)

    # background_tasks.add_task(insert_student_into_db, dump)
    #
    # # Return an immediate response
    # return {"message": "6759bdd58ab193d692682582"}

    s = await reports_collection.insert_one(dump)
    print(s)
    print(str(s.inserted_id))
    return await get_stop_status()


@app.get("/reports/latest")
async def get_latest_reports(
    page_size: int = Query(100, ge=1), page_idx: int = Query(0, ge=0)
) -> BatchedReports:
    """
    Get the latest reports in a paginated way.
    @param page_size: page size
    @param page_idx: page index
    @return: BatchedReports json object
    """
    skip = page_idx * page_size
    cursor = (
        reports_collection.find()
        .sort("timestamp", DESCENDING)
        .skip(skip)
        .limit(page_size)
    )
    reports = await cursor.to_list(length=page_size)
    total_docs = await reports_collection.count_documents({})
    return BatchedReports(reports=reports, total_records=total_docs)


def human_readable_time_ago(dt: datetime) -> str:
    """
    Convert a datetime object to a human-readable time ago string.
    @param dt: time
    @return: string
    """
    diff = datetime.now() - dt
    minutes = int(diff.total_seconds() // 60)
    if minutes < 60:
        return f"{minutes} min ago"
    hours = minutes // 60
    if hours < 24:
        return f"{hours} h ago"
    days = hours // 24
    return f"{days} d ago"


@app.get("/reports/workers/latest")
async def get_workers_latest() -> list[WorkerLatest]:
    """
    Get the latest 1000 reports for each worker.
    @return: List of WorkerLatest objects.
    """

    # group by worker_id, sort descending by timestamp, keep the latest timestamp,
    # and slice the last 1000

    # max nbr of reports per worker -> to avoid retrieving too much data
    max_nbr_reports_by_worker = 500

    pipeline = [
        {"$sort": {"timestamp": -1}},
        {
            "$group": {
                "_id": "$worker_id",
                "reports": {"$push": "$$ROOT"},
                "latest_timestamp": {"$first": "$timestamp"},
            }
        },
        {
            "$project": {
                "_id": 0,
                "worker_id": "$_id",
                "latest_timestamp": 1,
                "latest_reports": {"$slice": ["$reports", max_nbr_reports_by_worker]},
            }
        },
    ]

    docs = await reports_collection.aggregate(pipeline).to_list(None)

    # post-processing: add last_active (human-readable time)
    # and latest_reports (count of finished_file_paths)
    for doc in docs:
        doc["last_active"] = human_readable_time_ago(doc["latest_timestamp"])
        doc["latest_reports"] = [
            {
                "timestamp": r["timestamp"],
                "count": len(r.get("finished_file_paths", [])),
                # "file_paths": r.get("finished_file_paths", [])
            }
            for r in doc["latest_reports"]
        ]

    return docs


@app.post("/init-db")
async def init_db(metadata: DashboardMetadata) -> dict:
    """
    Reset the database and initialize everything
    """
    metadata.start_time = datetime.now()
    try:
        await reset_reports_collection()
        await reset_dashboardmetadata_collection()
        await dashboardmetadata_collection.insert_one(metadata.dict())
        return {"message": "Database reset and initialized."}
    except Exception as e:
        return {"error": str(e)}


async def reset_reports_collection():
    await db.drop_collection("reports")
    await db.create_collection("reports")
    await db["reports"].create_index("timestamp")


async def reset_dashboardmetadata_collection():
    await db.drop_collection("dashboardmetadata")
    await db.create_collection("dashboardmetadata")


async def latest_activity():
    """Get the latest activity timestamp in human-readable format."""
    latest_doc = await reports_collection.find_one({}, sort=[("timestamp", DESCENDING)])
    if latest_doc:
        last_eta = human_readable_time_ago(latest_doc["timestamp"])
    else:
        last_eta = "No reports available"
    return last_eta


async def count_nbr_finished_files():
    cursor = reports_collection.find({}, {"finished_file_paths": 1})
    finished_files_count = 0
    async for doc in cursor:
        finished_files_count += len(doc.get("finished_file_paths", []))
    return finished_files_count


@app.get("/progress")
async def get_progress() -> Progress:
    m: Optional[dict[str, Any]] = await dashboardmetadata_collection.find_one()
    if m is None:
        raise ValueError("No dashboard metadata found")

    metadata: DashboardMetadata = DashboardMetadata(**m)

    nbr_finished_files = await count_nbr_finished_files()
    total = metadata.total_files
    progress_porcentage = (nbr_finished_files / (total or 1)) * 100
    last_eta = await latest_activity()

    return Progress(
        total_files=total,
        start_time=metadata.start_time,
        finished_files=nbr_finished_files,
        progress=progress_porcentage,
        last_activity=last_eta,
        ask_to_stop=metadata.ask_to_stop,
    )


@app.post("/stop")
async def stop_processing():
    await dashboardmetadata_collection.update_one({}, {"$set": {"ask_to_stop": True}})
    return {"message": "Asking to processors to stop."}


async def get_stop_status():
    m = await dashboardmetadata_collection.find_one()
    if m is None:
        return False

    return m.get("ask_to_stop", False)


def run_api(host: str, port: int):
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host on which the dashboard API should be run.",
    )
    parser.add_argument(
        "--port", default=8000, help="Port on which the dashboard API should be run."
    )
    args = parser.parse_args()

    run_api(args.host, args.port)

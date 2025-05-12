from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, BeforeValidator, Field
from typing_extensions import Annotated

# ObjectID will be casted into a string before being validated at being a str
PyObjectId = Annotated[str, BeforeValidator(str)]


class Report(BaseModel):
    """Report model saved in the database."""

    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    worker_id: str = Field(alias="worker_id")
    finished_file_paths: List[str]
    timestamp: Optional[datetime] = Field(default=None)

    class Config:
        populate_by_name = True
        extra = "allow"


class DashboardMetadata(BaseModel):
    """Progress metadata saved in the database."""

    total_files: int
    start_time: Optional[datetime] = Field(default=None)
    ask_to_stop: bool = False


##########


class LatestReport(BaseModel):
    """report info sent to frontend"""

    timestamp: datetime
    count: int
    # file_paths: List[str]


class WorkerLatest(BaseModel):
    """worker info sent to frontend"""

    worker_id: str
    latest_timestamp: datetime
    last_active: str
    latest_reports: List[LatestReport]


class Progress(BaseModel):
    total_files: int
    start_time: Optional[datetime]
    finished_files: int
    progress: float
    last_activity: str
    ask_to_stop: bool


class BatchedReports(BaseModel):
    """Batch report model  sent to frontend"""

    reports: List[Report]
    total_records: int

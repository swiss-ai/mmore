from dataclasses import dataclass
from datetime import datetime
import os
from typing import Any, Dict, List
import logging
import validators

logger = logging.getLogger(__name__)


@dataclass
class MultimodalRawInput:
    type: str
    value: str


class FileDescriptor:
    def __init__(
            self,
            file_path: str,
            file_name: str,
            file_size: int,
            created_at: str,
            modified_at: str,
            file_extension: str,
    ):
        self.file_path = file_path
        self.file_name = file_name
        self.file_size = file_size
        self.created_at = created_at
        self.modified_at = modified_at
        self.file_extension = file_extension

    @staticmethod
    def from_filename(file_path: str):
        try:
            stat = os.stat(file_path)
            return FileDescriptor(
                file_path=file_path,
                file_name=os.path.basename(file_path),
                file_size=stat.st_size,
                created_at=datetime.fromtimestamp(stat.st_ctime).isoformat(),
                modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
                file_extension=os.path.splitext(file_path)[1].lower().strip(),
            )
        except (FileNotFoundError, PermissionError) as e:
            logging.error(f"Error accessing file {file_path}: {e}")
            return None

    def to_dict(self):
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "file_size": self.file_size,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "file_extension": self.file_extension,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]):
        return cls(
            file_path=data["file_path"],
            file_name=data["file_name"],
            file_size=data["file_size"],
            created_at=data["created_at"],
            modified_at=data["modified_at"],
            file_extension=data["file_extension"],
        )


class URLDescriptor:
    def __init__(
        self,
        url: str,
        file_path: str = None,
        file_name: str = None,
        file_size: int = 0,
        created_at: str = None,
        modified_at: str = None,
        file_extension: str = ".html",
    ):
        if not validators.url(url):
            raise ValueError(f"Invalid URL: {url}")

        self.url = url
        self.file_path = file_path or url
        self.file_name = file_name or os.path.basename(url.rstrip('/'))
        self.file_size = file_size
        self.created_at = created_at or datetime.now().isoformat()
        self.modified_at = modified_at or self.created_at
        self.file_extension = file_extension

    @staticmethod
    def from_filename(file_path: str):
        raise NotImplementedError("URLDescriptor does not support from_filename.")

    def to_dict(self) -> Dict[str, str]:
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "file_size": self.file_size,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "file_extension": self.file_extension,
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            url=data["file_path"],  # URL stored in `file_path` for compatibility
            file_path=data["file_path"],
            file_name=data["file_name"],
            file_size=data["file_size"],
            created_at=data["created_at"],
            modified_at=data["modified_at"],
            file_extension=data["file_extension"],
        )

@dataclass
class MultimodalSample:
    text: str | List[Dict[str, str]]
    modalities: List[MultimodalRawInput]
    metadata: Dict[str, str] | None = None

    def to_dict(self):
        if isinstance(self.text, list):
            return {
                "conversations": self.text,
                "modalities": [m.__dict__ for m in self.modalities],
                "metadata": self.metadata.to_dict() if self.metadata else None,
            }
        return {
            "text": self.text,
            "modalities": [m.__dict__ for m in self.modalities],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        if isinstance(data, MultimodalSample):
            return data

        key = "conversations" if "conversations" in data else "text"
        # Take care of quotes in the text (jsonl serialization)
        if key == "text":
            data[key] = data[key]
        else:
            for conv in data[key]:
                for k, v in conv.items():
                    conv[k] = v
        return cls(
            text=data[key],
            modalities=[MultimodalRawInput(**m) for m in data["modalities"]],
            metadata=data.get("metadata", None),
        )
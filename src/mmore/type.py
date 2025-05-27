"""
This module defines data structures and utilities for managing multimodal inputs,
file metadata, and URL descriptors.

Classes:
    MultimodalRawInput: Represents a single modality (e.g. image) with a type and value.
    MultimodalSample: Encapsulates a multimodal input sample with text, modalities, and metadata.
    FileDescriptor: Captures metadata for files, including path, size, timestamps, and extension.
    URLDescriptor: Represents a URL with validation and computational weight.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import validators

logger = logging.getLogger(__name__)


@dataclass
class MultimodalRawInput:
    """
    Represents a single modality input.

    Attributes:
        type (str): The type of the modality (e.g., "image").
        value (str): The value of the modality, such as a file path or text content.
    """

    type: str
    value: str


@dataclass
class MultimodalSample:
    """
    Encapsulates a multimodal input sample, including text, modalities, and optional metadata.

    Attributes:
        text (str | List[Dict[str, str]]): The textual content or structured conversation data.
        modalities (List[MultimodalRawInput]): List of modalities (e.g., images, audio).
        metadata (Dict[str, str] | None): Additional metadata associated with the sample.
    """

    text: str
    modalities: List[MultimodalRawInput]
    metadata: Dict[str, Union[str, Dict, List]] = field(default_factory=dict)
    id: str = ""

    def __post_init__(self):
        if self.id == "":
            self.id = str(hash(self.text))
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self):
        if isinstance(self.text, list):
            return {
                "conversations": self.text,
                "modalities": [m.__dict__ for m in self.modalities],
                "metadata": self.metadata if self.metadata else None,
            }
        return {
            "text": self.text,
            "modalities": [m.__dict__ for m in self.modalities],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        # key = "conversations" if "conversations" in data else "text"
        # # Take care of quotes in the text (jsonl serialization)
        # if key == "text":
        #     data[key] = data[key]
        # else:
        #     for conv in data[key]:
        #         for k, v in conv.items():
        #             conv[k] = v
        return cls(
            text=data["text"],
            modalities=[MultimodalRawInput(**m) for m in data.get("modalities", [])],
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_jsonl(cls, file_path: str) -> List["MultimodalSample"]:
        samples = []
        with open(file_path, "r") as f:
            for line in f:
                samples.append(cls.from_dict(json.loads(line)))
        return samples

    @staticmethod
    def to_jsonl(file_path: str, samples: List["MultimodalSample"]) -> None:
        with open(file_path, "a") as f:
            for sample in samples:
                f.write(json.dumps(sample.to_dict()) + "\n")


class FileDescriptor:
    """
    Captures metadata for files, including file path, size, creation and modification times, and file extension.

    Attributes:
        file_path (str): The full path to the file.
        file_name (str): The name of the file.
        file_size (int): The size of the file in bytes.
        created_at (str): ISO format timestamp of when the file was created.
        modified_at (str): ISO format timestamp of when the file was last modified.
        file_extension (str): The file's extension (e.g., ".txt").
    """

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
            "file_size": str(self.file_size),
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "file_extension": self.file_extension,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, str]):
        return cls(
            file_path=data["file_path"],
            file_name=data["file_name"],
            file_size=int(data["file_size"]),
            created_at=data["created_at"],
            modified_at=data["modified_at"],
            file_extension=data["file_extension"],
        )


class URLDescriptor:
    """
    Represents a URL with optional metadata attributes for processing, including file-like properties.
    It was done to be symmetrical with the FileDescriptor class for easy integration with existing code.

    Attributes:
        url (str): The URL to be described.
        file_path (str): defaults to the URL itself.
        file_name (str): derived from the URL if not provided.
        file_size (int): defaults to 0.
        created_at (str): defaults to the current time.
        modified_at (str): defaults to `created_at`.
        file_extension (str): defaults to ".html".
    """

    def __init__(
        self,
        url: str,
        file_path: Optional[str] = None,
        file_name: Optional[str] = None,
        file_size: int = 0,
        created_at: Optional[str] = None,
        modified_at: Optional[str] = None,
        file_extension: str = ".html",
    ):
        if not validators.url(url):
            raise ValueError(f"Invalid URL: {url}")

        self.url = url
        self.file_path = file_path or url
        self.file_name = file_name or os.path.basename(url.rstrip("/"))
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
            "file_size": str(self.file_size),
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
            file_size=int(data["file_size"]),
            created_at=data["created_at"],
            modified_at=data["modified_at"],
            file_extension=data["file_extension"],
        )

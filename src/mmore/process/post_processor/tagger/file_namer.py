from langdetect import detect

from .base import BaseTagger, BaseTaggerConfig
import os

class FileNamer(BaseTagger):
    def __init__(self, name: str = "🔤 File Namer", metadata_key: str = "file_name"):
        super().__init__(name, metadata_key)

    def tag(self, sample):
        if "file_path" not in sample.metadata:
            return "unknown"

        return os.path.basename(str(sample.metadata["file_path"]))

    @classmethod
    def from_config(cls, config: BaseTaggerConfig):
        file_namer = FileNamer()
        return file_namer



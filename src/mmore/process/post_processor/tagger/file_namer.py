import os

from .base import BaseTagger, BaseTaggerConfig


class FileNamer(BaseTagger):
    """
    A tagger that extracts the file name from the sample's metadata.
    This tagger is useful for identifying the source file of a sample, especially when dealing with multiple files.
    It retrieves the file name from the `file_path` metadata key and uses it as a tag.
    Attributes:
        name (str): The name of the tagger.
        metadata_key (str): The key in the sample's metadata from which to extract the file name.
    """

    def __init__(self, name: str = "🔤 File Namer", metadata_key: str = "file_name"):
        """
        Initializes the FileNamer tagger.
        Args:
            name (str): The name of the tagger.
            metadata_key (str): The key in the sample's metadata from which to extract the file name.
        """
        super().__init__(name, metadata_key)

    def tag(self, sample):
        if "file_path" not in sample.metadata:
            return "unknown"

        return os.path.basename(str(sample.metadata["file_path"]))

    @classmethod
    def from_config(cls, config: BaseTaggerConfig):
        file_namer = FileNamer()
        return file_namer

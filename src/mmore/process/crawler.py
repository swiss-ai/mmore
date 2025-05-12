import json
import logging
import os
from typing import Dict, List, Optional

import validators

from ..type import FileDescriptor, URLDescriptor

logger = logging.getLogger(__name__)


class DispatcherReadyResult:
    def __init__(
        self, urls: List[URLDescriptor], file_paths: Dict[str, List[FileDescriptor]]
    ):
        """
        Initialize the DispatcherReadyResult object.

        Args:
            urls (List[URLDescriptor]): List of URLs to process.
            file_paths (Dict[str, List[FileDescriptor]]): Dictionary of file paths to process.
        """
        self.urls = urls
        self.file_paths = file_paths

        self.common_root = (
            os.path.commonpath(self.file_paths.keys()) if len(file_paths) > 1 else None
        )

        # All keys should be shortened to be relative to the common root
        if len(urls) > 1:
            keys_to_change = list(self.file_paths.keys())
            for key in keys_to_change:
                if self.common_root:
                    key = key.replace(self.common_root, "")

                self.file_paths[key] = self.file_paths[key]
                del self.file_paths[key]

    def __call__(self):
        """
        Flatten the file paths into a single list.

        Returns:
            List[FileDescriptor]: A flattened list of file descriptors.
        """
        return [
            file_path
            for file_list in self.file_paths.values()
            for file_path in file_list
        ]

    def __len__(self):
        return len(self.urls) + sum(
            len(file_list) for file_list in self.file_paths.values()
        )

    def __repr__(self):
        """
        Returns a string representation of the DispatcherReadyResult object.
        """
        return f"DispatcherReadyResult(urls={self.urls}, file_paths={self.file_paths}, common_root={self.common_root})"

    def to_dict(self):
        """
        Convert the result to a dictionary format.

        Returns:
            dict: A dictionary representation of the result.
        """
        return {
            "urls": [url.to_dict() for url in self.urls],
            "file_paths": {
                key: [file.to_dict() for file in file_list]
                for key, file_list in self.file_paths.items()
            },
            "common_root": self.common_root,
        }

    @staticmethod
    def from_dict(data: dict):
        """
        Create a DispatcherReadyResult object from a dictionary.

        Args:
            data (dict): A dictionary containing the result data.

        Returns:
            DispatcherReadyResult: The reconstructed object.
        """
        urls = [URLDescriptor(url=url["url"]) for url in data["urls"]]
        file_paths = {
            key: [FileDescriptor(**file) for file in file_list]
            for key, file_list in data["file_paths"].items()
        }
        return DispatcherReadyResult(urls=urls, file_paths=file_paths)


class FindAlreadyComputedFiles:
    """
    This class is used to get the list of all files that have already been processed.
    It will traverse the output_path directory and get all the results.jsonl files
    where in each line (representing a sample) we have the metadata of the file_path that was used to create that sample.
    > See create_sample in utils.py (file_path is in metadata of the sample).

    Reminder here is the structure of the output_path directory (see DispatcherConfig):
    output_path
    ├── processors
    | ├── Processor_type_1
    | | └── results.jsonl
    | ├── Processor_type_2
    | | └── results.jsonl
    | ├── ...
    |
    └── merged
        └── merged_results.jsonl
    """

    def __init__(self, output_path: str):
        """
        output_path: the path where the output of the Process is stored.
        """
        if output_path is None:
            raise ValueError("output_path must be provided.")
        self.output_path = output_path

    def _get_all_samples_jsonl_paths(self, output_path):
        # Get all the results.jsonl files in the output_path directory.
        samples_files = []
        for root, _, files in os.walk(output_path):
            for file in files:
                if file.endswith("results.jsonl"):
                    samples_files.append(os.path.join(root, file))
        return samples_files

    def _get_metadata_jsonl_path(self, results_jsonl_path):
        # read jsonl file and for each item in the file, get the metadata's file_path
        # return the list of all file_path in this jsonl file
        file_paths = []
        with open(results_jsonl_path, "r") as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                if "metadata" in data and "file_path" in data["metadata"]:
                    file_paths.append(data["metadata"]["file_path"])
                else:
                    print(
                        f"Warning file_path not found in metadate (line{i} of {results_jsonl_path})"
                    )
        return file_paths

    def get_all_files_already_processed(self) -> set[str]:
        """
        This function returns the set of all files path's that have already been processed.
        Returns:
             set of all files path's that have already been processed.
        """
        samples = self._get_all_samples_jsonl_paths(self.output_path)
        files_already_processed = set()
        for f in samples:
            line = self._get_metadata_jsonl_path(f)
            files_already_processed.update(line)
        return files_already_processed


class CrawlerConfig:
    """
    Configuration for the Crawler.

    Attributes:
        root_dirs (List[str]): List of root directories to crawl.
        supported_extensions (List[str]): List of file extensions to include.
    """

    def __init__(
        self,
        root_dirs: List[str],
        supported_extensions: Optional[List[str]] = None,
        output_path: Optional[str] = None,
    ):
        """
        Initialize a CrawlerConfig object.

        Args:
            root_dirs (List[str]): List of root directories to crawl.
            supported_extensions (Optional[List[str]]): List of file extensions to include.
            output_path (Optional[str]): Path to the output directory, useful to take a look at the already processed files and discard them.
        """
        self.root_dirs = root_dirs
        self.supported_extensions = supported_extensions or []
        self.output_path = output_path

    @staticmethod
    def from_dict(config: Dict):
        """
        Create a CrawlerConfig object from a dictionary.

        Args:
            config (dict): Configuration dictionary.

        Returns:
            CrawlerConfig: The reconstructed configuration object.
        """
        return CrawlerConfig(
            root_dirs=config.get("root_dirs", []),
            supported_extensions=config.get("supported_extensions", None),
            output_path=config.get("output_path", None),
        )

    @staticmethod
    def from_yaml(yaml_path: str):
        """
        Load a CrawlerConfig object from a YAML file.

        Args:
            yaml_path (str): Path to the YAML configuration file.

        Returns:
            CrawlerConfig: The reconstructed configuration object.
        """
        import yaml

        try:
            with open(yaml_path, "r") as file:
                config = yaml.safe_load(file)
            return CrawlerConfig.from_dict(config)
        except (FileNotFoundError, yaml.YAMLError):
            logger.error(f"[Crawler] Error processing {yaml_path}.")
            raise

    def to_dict(self):
        """
        Convert the configuration to a dictionary.

        Returns:
            dict: A dictionary representation of the configuration.
        """
        return {
            "root_dirs": self.root_dirs,
            "supported_extensions": self.supported_extensions,
            "output_path": self.output_path,
        }


class Crawler:
    """
    A utility class to crawl directories and URLs for files.

    Attributes:
        config (CrawlerConfig): The crawler configuration.
        files (Dict): Dictionary containing local files and URLs.
        dirs (List[str]): List of directories to crawl.
        lax_mode (bool): Whether to skip unrecognized paths or raise an error.
    """

    def __init__(
        self,
        config: Optional[CrawlerConfig] = None,
        root_dirs: List[str] = [],
        output_path: str = "",
        lax_mode: bool = False,
    ):
        """
        Initialize a Crawler object.

        Args:
            config (CrawlerConfig): The crawler configuration.
            root_dirs (List[str]): List of root directories to crawl.
            output_path (str): Path to the output directory, useful to take a look at the already processed files and discard them.
            lax_mode (bool): Whether to skip unrecognized paths or raise an error.

        Raises:
            ValueError: If neither config nor root_dirs is provided.
        """

        if not config:
            if not root_dirs:
                raise ValueError("Either config or root_dirs must be provided.")
            if not output_path:
                raise ValueError("Either config or output_path must be provided.")
            config = CrawlerConfig(root_dirs=root_dirs, output_path=output_path)

        self.config = config
        self.files = {"local": {}, "url": []}
        self.dirs = []
        self.lax_mode = lax_mode

    def reset(self):
        """
        Reset the internal state of the crawler.
        """
        self.files = {"local": {}, "url": []}
        self.dirs = []

    def _traverse_directories(self) -> None:
        """
        Traverse the directories and collect file descriptors for supported files.
        """
        for root_dir in self.dirs:
            for dirpath, _, filenames in os.walk(root_dir):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in self.config.supported_extensions:
                        if not self.files["local"].get(root_dir):
                            self.files["local"][root_dir] = []
                        self.files["local"][root_dir].append(
                            FileDescriptor.from_filename(filepath)
                        )

    def _filter_out_already_processed_files(
        self, files: Dict[str, List[FileDescriptor]], output_path: str
    ) -> Dict[str, List[FileDescriptor]]:
        """
        Avoid processing files that have already been processed.
        Immutable function.
        Args:
            files: the crawled files that want to be processed. We want to remove the files that have already been processed.
            output_path: the path where the outputs of the "process" is stored.
        Returns:
            filtered out 'files' to process.
        """
        all_files_done: set[str] = FindAlreadyComputedFiles(
            output_path
        ).get_all_files_already_processed()
        logger.info(f"Found {len(all_files_done)} files already processed.")

        for root_dir, files_in_dir in files.items():
            files[root_dir] = [
                f for f in files_in_dir if f.file_path not in all_files_done
            ]

        if len(all_files_done) > 0:
            logger.info(f"Removed {len(all_files_done)} files already processed.")
            logger.info(
                f"New total files to process: {sum(len(files) for files in files.values())}"
            )
        return files

    def crawl(self, skip_already_processed: bool = False) -> DispatcherReadyResult:
        """
        Crawl the configured directories and URLs.
        Args:
            skip_already_processed (bool): if set to True, the crawler will scan the outputs folder and detect files that correspond to them, and skip them.
        Returns:
            DispatcherReadyResult: The result of the crawl operation, ready to be dispatched to the processors.
        """
        self.reset()

        for root_dir in self.config.root_dirs:
            if validators.url(root_dir):
                self.files["url"].append(URLDescriptor(url=root_dir))
            elif os.path.isdir(root_dir):
                self.files["local"][root_dir] = []
                self.dirs.append(root_dir)
            elif self.lax_mode:
                logger.warning(f"Skipping unrecognized path or URL: {root_dir}")
            else:
                raise ValueError(f"Invalid path or URL: {root_dir}")

        self._traverse_directories()  # Only local directories are traversed

        total_files = sum(len(files) for files in self.files["local"].values()) + len(
            self.files["url"]
        )
        logger.info(f"Found {total_files} files/URLs to process.")

        urls: List[URLDescriptor] = self.files["url"]
        file_paths: Dict[str, List[FileDescriptor]] = self.files["local"]

        if self.config.output_path and skip_already_processed:
            logger.info(
                "Checking if some of those files to process have already been processed."
            )
            file_paths = self._filter_out_already_processed_files(
                files=file_paths, output_path=self.config.output_path
            )

        return DispatcherReadyResult(urls=urls, file_paths=file_paths)

import os
import logging
from typing import List, Dict, Optional
import validators
from ..type import FileDescriptor, URLDescriptor

logger = logging.getLogger(__name__)


class DispatcherReadyResult:
    def __init__(
            self, urls: List[URLDescriptor], file_paths: Dict[str, List[FileDescriptor]]
    ):
        """
        Initialize the DispatcherReadyResult object.
            :param urls: List of URLs to process.
            :param file_paths: Dictionary of file paths to process.
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
                self.file_paths[key.replace(self.common_root, "")] = self.file_paths[
                    key
                ]
                del self.file_paths[key]

    def __call__(self):
        """
        Flatten the file paths into a single list.
        :return: List of file paths.
        """
        return [
            file_path
            for file_list in self.file_paths.values()
            for file_path in file_list
        ]

    def __repr__(self):
        return f"DispatcherReadyResult(urls={self.urls}, file_paths={self.file_paths}, common_root={self.common_root})"

    def to_dict(self):
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
        urls = [URLDescriptor(url=url["url"]) for url in data["urls"]]
        file_paths = {
            key: [FileDescriptor(**file) for file in file_list]
            for key, file_list in data["file_paths"].items()
        }
        return DispatcherReadyResult(urls=urls, file_paths=file_paths)


class CrawlerConfig:
    def __init__(
            self,
            root_dirs: List[str],
            supported_extensions: Optional[List[str]] = None,
    ):
        self.root_dirs = root_dirs
        self.supported_extensions = supported_extensions or []

    @staticmethod
    def from_dict(config: Dict):
        return CrawlerConfig(
            root_dirs=config.get("root_dirs", []),
            supported_extensions=config.get("supported_extensions", None),
        )

    @staticmethod
    def from_yaml(yaml_path: str):
        import yaml

        try:
            with open(yaml_path, "r") as file:
                config = yaml.safe_load(file)
            return CrawlerConfig.from_dict(config)
        except (FileNotFoundError, yaml.YAMLError):
            logger.error(f"[Crawler] Error processing {yaml_path}.")
            raise

    def to_dict(self):
        return {
            "root_dirs": self.root_dirs,
            "supported_extensions": self.supported_extensions,
        }


class Crawler:
    def __init__(
            self,
            config: CrawlerConfig = None,
            root_dirs: List[str] = None,
            lax_mode: bool = False,
    ):
        if not config:
            if not root_dirs:
                raise ValueError("Either config or root_dirs must be provided.")
            config = CrawlerConfig(root_dirs=root_dirs)

        self.config = config
        self.files = {"local": {}, "url": []}
        self.dirs = []
        self.lax_mode = lax_mode

    def reset(self):
        self.files = {"local": {}, "url": []}
        self.dirs = []

    def _traverse_directories(self) -> None:
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

    def crawl(self) -> DispatcherReadyResult:
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

        return DispatcherReadyResult(
            urls=self.files["url"], file_paths=self.files["local"]
        )

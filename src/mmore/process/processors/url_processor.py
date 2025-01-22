import logging
import os
import io
import requests
import torch
from urllib.parse import urljoin
from typing import List
from PIL import Image
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import trafilatura
from multiprocessing import Pool, cpu_count
import re
from urllib.parse import urljoin
from src.mmore.type import URLDescriptor
from .processor import Processor, ProcessorResult, ProcessorConfig

logger = logging.getLogger(__name__)


class URLProcessor(Processor):
    def __init__(self, urls: List[URLDescriptor], config=None):
        super().__init__(files=[], config=config or ProcessorConfig())

    def accepts(self, input_obj) -> bool:
        return isinstance(input_obj, URLDescriptor)

    def require_gpu(self) -> bool:
        return True

    def process_one_file(self, file_path: str, fast: bool = False) -> ProcessorResult:
        return NotImplementedError("URLProcessor is not implemented")

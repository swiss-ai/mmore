"""
Utility functions for processing files, images, PDFs, and text. 
These functions can be used across various processors for data extraction, 
cleaning, splitting, and aggregation.
"""


from io import BytesIO
import logging
import tempfile
import os
import requests
import validators
from cleantext import clean
from PIL import Image
from typing import List
import fitz
from src.mmore.type import FileDescriptor
from datetime import datetime
from typing import Tuple, Dict
from pathlib import Path
from uuid import uuid4
import json
import numpy as np 

logger = logging.getLogger(__name__)


def download_image(url) -> Image:
    """
    Download an image from a URL and return it as a PIL Image object.

    Args:
        url (str): URL of the image.

    Returns:
        Image.Image: PIL Image object of the downloaded image, or None if the download fails.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img
    except Exception as e:
        logger.error(f"Failed to download image from {url}: {e}")
        return None


def open_image(path) -> Image:
    """
    Open an image from a file path and return it as a PIL Image object.

    Args:
        path (str): Path to the image file.

    Returns:
        Image.Image: PIL Image object of the opened image, or None if opening fails.
    """
    try:
        img = Image.open(path).convert("RGB")
        return img
    except Exception as e:
        logger.error(f"Failed to open image at {path}: {e}")
        return None


def load_image(file: str) -> Image:
    """
    Load an image from a file path or URL and return it as a PIL Image object.

    Args:
        file (str): Path or URL to the image.

    Returns:
        Image.Image: PIL Image object, or None if loading fails.
    """
    if os.path.exists(file):
        return open_image(file)
    elif validators.url(file):
        return download_image(file)
    else:
        logger.error(f"Invalid image file or URL: {file}")
        return None

def clean_text(text: str) -> str:
    """
    Clean a given text using `cleantext` library. https://pypi.org/project/clean-text/

    Args:
        text (str): Input text to be cleaned.

    Returns:
        str: Cleaned text.
    """
    return clean(
        text=text,
        fix_unicode=True,
        to_ascii=True,
        lower=False,
        no_line_breaks=False,
        no_urls=False,
        no_emails=False,
        no_phone_numbers=False,
        no_numbers=False,
        no_digits=False,
        no_currency_symbols=False,
        no_punct=False,
        replace_with_punct="",
        replace_with_url="This is a URL",
        replace_with_email="Email",
        replace_with_phone_number="",
        replace_with_number="123",
        replace_with_digit="0",
        replace_with_currency_symbol="$",
        lang="en",
    )


def clean_image(image: Image.Image, min_width=512, min_height=512, variance_threshold=100) -> bool:
    """
    Validates an image based on size and variance (whether its one-colored).

    Args:
        image (PIL.Image.Image): The image to validate.
        min_width (int, optional): The minimum width an image must have to be considered valid. Defaults to 512.
        min_height (int, optional): The minimum height an image must have to be considered valid. Defaults to 512.
        variance_threshold (int, optional): The minimum variance in pixel intensity required. Images with lower variance are considered "empty". Defaults to 100.

    Returns:
        bool: True if the image meets all criteria, False otherwise.
    """
    w, h = image.size

    # Check size criteria
    if w < min_width or h < min_height:
        return False

    # Check variance threshold
    gray = image.convert("L")
    arr = np.array(gray)
    variance = arr.var()
    if variance < variance_threshold:
        return False

    return True

def _save_temp_image(image: Image.Image, base_path=None) -> str:
    """
    Save an image as a temporary file.

    Args:
        image (Image.Image): Image to save.
        base_path (str, optional): Base directory for saving the file.

    Returns:
        str: Path to the saved image.
    """
    try:
        # use systems temp dir if no path is provided 
        temp_dir = base_path or tempfile.gettempdir()
        pid = os.getpid() # add pid to avoid conflicts
        unique_prefix = f"temp_{pid}_"
        temp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=".png", prefix=unique_prefix, dir=temp_dir
        )
        temp_file_path = temp_file.name
        image.save(temp_file_path, format="PNG")
        temp_file.close()
        if base_path:
            return Path(temp_file_path).relative_to(base_path)
        return temp_file_path
    except Exception as e:
        logger.error(f"Failed to save temporary image: {e}")


def create_sample(texts: List[str], images: List[Image.Image], path=None) -> dict:
    """
    Create a sample dictionary containing text, images, and optional metadata.
    This function is called within all processors.

    Args:
        texts (List[str]): List of text strings.
        images (List[Image.Image]): List of images.
        path (str, optional): Path for metadata. Defaults to None.

    Returns:
        dict: Sample dictionary with text, image modalities, and metadata.
    """
    base_path = os.environ.get("MMORE_RESULTS_PATH", None)
    if base_path is not None:
        base_path = Path(base_path) / str(uuid4())
        base_path.mkdir(exist_ok=True)

    sample = {
        "text": "\n".join(texts),
        "modalities": [
            {"type": "image", "value": _save_temp_image(img, base_path=base_path)}
            for img in images
        ],
        "metadata": {"file_path": path},
    }

    if path is None:
        sample.pop("metadata")

    if base_path:  # TODO: ???
        try:
            with open(base_path / "descriptor.json", "w") as f:
                json.dump(sample, f)

            return sample
        except:
            pass

    return sample


def create_sample_alread_saved_images(texts: List[str], images: List[str], file_path: str) -> dict:
    return {
        "text": "\n".join(texts),
        "modalities": [{"type": "image", "value": img} for img in images],
        "metadata": {"file_path": file_path},
    }


def evenly_split_across_gpus(x_list, num_gpus):
    """
    Evenly split a list of things across multiple GPUs.

    :param x_list: List of things to split.
    :param num_gpus: Number of GPUs to split across.
    :return: List of things split across GPUs.
    """
    x_per_gpu = len(x_list) // num_gpus
    x_split = [x_list[i * x_per_gpu: (i + 1) * x_per_gpu] for i in range(num_gpus)]
    # NOTE : If the number of things is not divisible by the number of GPUs, the last GPU will get the remainder, and can get 0 to x_per_gpu - 1 things.
    # Nevertheless, if we consider that processing each thing takes the same amount of time, this will not be a problem.
    return x_split


def clean_pdf_list(pdf_list: List[FileDescriptor]) -> List[FileDescriptor]:
    """
    Filter and return a list of valid PDF files.

    Args:
        pdf_list (List[FileDescriptor]): List of file descriptors representing PDFs.

    Returns:
        List[FileDescriptor]: List of valid PDF file descriptors.
    """
    clean_list = []
    for files in pdf_list:
        try:
            pdf = fitz.open(files.file_path)
            clean_list.append(files)
            pdf.close()
        except Exception as e:
            logger.error(f"Failed to open pdf at {files.file_path}: {e}")
    return clean_list


def evenly_split_accross_gpus_num_pages(x_list, num_gpus):
    """
    Evenly split a list of files across multiple GPUs based on the number of pages in each file.
    The goal is to have a similar number of pages on each GPU.

    Args:
        x_list (List[FileDescriptor]): List of file descriptors to split.
        num_gpus (int): Number of GPUs to distribute files across.

    Returns:
        List[List[FileDescriptor]]: List of file groups assigned to each GPU.
    """

    def get_num_pages(file: FileDescriptor) -> int:
        pdf_doc = fitz.open(file.file_path)
        return len(pdf_doc)

        # Sort files by number of pages in descending order

    files_sorted = sorted(x_list, key=get_num_pages, reverse=True)

    # Initialize lists for each GPU
    gpu_lists = [[] for _ in range(num_gpus)]
    page_totals = [0] * num_gpus  # Track total pages for each GPU

    # Distribute files
    for file in files_sorted:
        # Find the GPU with the smallest number of pages
        min_gpu = page_totals.index(min(page_totals))

        # Add file to this GPU list and update page count
        gpu_lists[min_gpu].append(file)
        page_totals[min_gpu] += get_num_pages(file)
    
    logger.info(
        f"Chunks size: {[sum([get_num_pages(file) for file in chunk]) for chunk in gpu_lists]}"
    )
    return gpu_lists


def merge_multiple_files_in_one(
        files: List["FileDescriptor"],
) -> Tuple[FileDescriptor, Dict[str, int]]:
    """
    Merge multiple PDF files into a single file while keeping an index of page boundaries.

    Args:
        files (List[FileDescriptor]): List of file descriptors to merge.

    Returns:
        Tuple[FileDescriptor, Dict[str, int]]: Merged file descriptor and index of page boundaries.
    """
    if len(files) == 0:
        logger.info("No files to merge")
        return None, None
    merged_document = fitz.open()  # Create a new PDF document
    page_indices = {}  # Dictionary to store the starting page of each file

    current_page = 0
    for file in files:
        pdf = fitz.open(file.file_path)
        try:
            merged_document.insert_pdf(pdf)  # Append all pages of the current file
        except Exception as e:
            logger.error(f"Failed to merge file {file.file_name}: {e}")
            continue
        page_indices[file.file_name] = current_page  # Store the starting page
        current_page += pdf.page_count  # Update the current page count
        pdf.close()

    temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    if current_page == 0:
        logger.info("No pages to merge")
        return None, None
    merged_document.save(temp_file.name)
    merged_document.close()

    merged_file_descriptor = FileDescriptor(
        file_path=temp_file.name,
        file_name="merged_file",
        file_size=sum(file.file_size for file in files),
        created_at=datetime.now().isoformat(),
        modified_at=datetime.now().isoformat(),
        file_extension="pdf",
    )

    return merged_file_descriptor, page_indices


def aggregate_files_accross_gpu(
        files: List[FileDescriptor], num_gpus: int
) -> List[List[FileDescriptor]]:
    """
    First split the files accross GPUs, then aggregate the file groups in a single so that each GPU can process a single file.
    """
    pdfs_list = clean_pdf_list(files)
    chunks = evenly_split_accross_gpus_num_pages(pdfs_list, num_gpus)

    page_indices = []
    new_chunks = []
    for chunk in chunks:
        if len(chunk) == 1:
            new_chunks.append(chunk)
            page_indices.append({chunk[0].file_name: 0})
        else:
            file, index_dict = merge_multiple_files_in_one(chunk)
            if file is not None and index_dict is not None:
                new_chunks.append([file])
                page_indices.append(index_dict)
    return new_chunks, page_indices


def create_sample_list(texts: List[str], images: List[Image.Image], files_path: List[str]) -> List[dict]:
    return [create_sample([text], image, path) for text, image, path in zip(texts, images, files_path)]


def create_sample_list_already_saved_images(texts: List[str], images: List[List[str]], paths: List[str]) -> List[dict]:
    return [create_sample_alread_saved_images([text], image, path) for text, image, path in zip(texts, images, paths)]


def merge_split_with_full_page_indexer(files: List[FileDescriptor], num_gpus: int) -> Tuple[
    List[List[FileDescriptor]], List[List[Tuple[int, str]]]]:
    """
    Merge and split files while maintaining a full page index.

    Args:
        files (List[FileDescriptor]): List of file descriptors to process.
        num_gpus (int): Number of GPUs to distribute files across.

    Returns:
        Tuple[List[List[FileDescriptor]], List[List[Tuple[int, str]]]]:
        File descriptors per GPU and their corresponding page indices.
    """
    def build_pdf(pdf_pages: List[Tuple[str, Tuple[int, int]]]):
        """Create a merged PDF from specific page ranges."""
        output_pdf = fitz.open()
        for file_path, (lower, upper) in pdf_pages:
            pdf = fitz.open(file_path)
            output_pdf.insert_pdf(pdf, lower, upper)

        temp_file = tempfile.NamedTemporaryFile(
            delete=False, suffix=".pdf"
        )

        output_pdf.save(temp_file)

        return [FileDescriptor.from_filename(file_path=temp_file.name)]

    def get_total_pages():
        total = 0
        for descriptor in files:
            total += len(fitz.open(descriptor.file_path))
        return total

    total_pages = get_total_pages()

    full_page_indexer = [list() for _ in range(num_gpus)]
    pages_per_gpu = (total_pages // num_gpus) + 1

    gpu_file_descriptors = []

    current_gpu_pages = []
    current_gpu = 0
    current_num_pages = 0

    for file in files:
        pdf_file = fitz.open(file.file_path)
        current_gpu_pages.append((file.file_path, [0, 0]))

        for i, _ in enumerate(pdf_file):
            if current_num_pages >= pages_per_gpu:
                current_gpu_pages[-1][1][1] = i
                gpu_file_descriptors.append(build_pdf(current_gpu_pages))
                current_gpu += 1
                current_num_pages = 0
                current_gpu_pages.clear()
                current_gpu_pages.append((file.file_path, [i + 1, i + 1]))

            full_page_indexer[current_gpu].append((i, file.file_path))
            current_num_pages += 1

        current_gpu_pages[-1][1][1] = len(pdf_file)

    if len(current_gpu_pages) > 0:
        current_gpu_pages[-1][1][1] = i
        gpu_file_descriptors.append(build_pdf(current_gpu_pages))

    return gpu_file_descriptors, full_page_indexer

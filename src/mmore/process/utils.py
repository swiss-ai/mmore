"""
Utility functions for processing files, images, PDFs, and text.
These functions can be used across various processors for data extraction,
cleaning, splitting, and aggregation.
"""

import logging

import numpy as np
from cleantext import clean
from PIL import Image

logger = logging.getLogger(__name__)


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
        to_ascii=False,
        lower=False,
        no_line_breaks=False,
        no_urls=False,
        no_emails=True,
        no_phone_numbers=False,
        no_numbers=False,
        no_digits=False,
        no_currency_symbols=False,
        no_punct=False,
        replace_with_punct="",
        replace_with_url="This is a URL",
        replace_with_email="email@email.com",
        replace_with_phone_number="",
        replace_with_number="123",
        replace_with_digit="0",
        replace_with_currency_symbol="$",
        lang="en",
    )


def clean_image(
    image: Image.Image, min_width=512, min_height=512, variance_threshold=100
) -> bool:
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
    if image is None:
        return False

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

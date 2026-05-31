"""PDF download + text extraction. Generic by design — never raises on remote errors."""

import logging
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def download_pdf(
    url: str,
    save_dir: str,
    user_agent: str = "Mozilla/5.0",
    timeout: int = 30,
) -> Optional[str]:
    """Returns local path or None.

    1. GET the URL.
    2. If response looks like a PDF, save it.
    3. Otherwise parse the HTML for the first PDF-looking <a href>, follow it.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": user_agent}

    try:
        r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    except requests.RequestException as e:
        logger.warning("download_pdf failed for %s: %s", url, e)
        return None

    if r.status_code != 200:
        logger.info("download_pdf got %s for %s", r.status_code, url)
        return None

    if _looks_like_pdf(r):
        return _save_pdf(r.content, url, save_dir)

    pdf_url = _find_pdf_link(r.text, base=url)
    if not pdf_url:
        return None

    try:
        r2 = requests.get(
            pdf_url, headers=headers, timeout=timeout, allow_redirects=True
        )
    except requests.RequestException as e:
        logger.warning("download_pdf follow-link failed for %s: %s", pdf_url, e)
        return None

    if r2.status_code == 200 and _looks_like_pdf(r2):
        return _save_pdf(r2.content, pdf_url, save_dir)
    return None


def _looks_like_pdf(response: requests.Response) -> bool:
    ctype = response.headers.get("Content-Type", "").lower()
    if "pdf" in ctype:
        return True
    if response.url.lower().endswith(".pdf"):
        return True
    return response.content[:5] == b"%PDF-"


def _save_pdf(content: bytes, url: str, save_dir: str) -> str:
    name = Path(url.split("?", 1)[0]).name or "paper"
    if not name.lower().endswith(".pdf"):
        name += ".pdf"
    path = str(Path(save_dir) / name)
    Path(path).write_bytes(content)
    return path


def _find_pdf_link(html: str, base: str) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")
    for a in soup.find_all("a", href=True):
        href = a["href"].lower()
        if href.endswith(".pdf") or "/pdf" in href or "/epdf" in href:
            return urljoin(base, a["href"])
    return None


def extract_text(pdf_path: str) -> str:
    """Extract text from a PDF using PyMuPDF (already in the `process` extra)."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.error("PyMuPDF not installed — install `mmore[paper_discovery]`")
        return ""

    try:
        with fitz.open(pdf_path) as doc:
            return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        logger.warning("extract_text failed for %s: %s", pdf_path, e)
        return ""

"""
Utilities for multimodal RAG context.
Collects image paths from retrieved document metadata and loads image files for vision-capable LLMs.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, TypeVar

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# Type alias for loaded image objects (typically PIL.Image)
ImageT = TypeVar("ImageT")


@dataclass
class MultimodalContext:
    """Container for multimodal context: text, image paths, and optional loaded images."""

    text: str
    image_paths: List[str] = field(default_factory=list)
    images: Optional[List[Any]] = None  # List[PIL.Image] when loaded

    def has_images(self) -> bool:
        return bool(self.image_paths) or bool(self.images)


def format_docs_multimodal(docs: List[Document]) -> MultimodalContext:
    """Build a multimodal context object from retrieved documents."""
    text_parts = [
        f"[{doc.metadata.get('rank', i + 1)}] {doc.page_content}"
        for i, doc in enumerate(docs)
    ]
    text = "\n\n".join(text_parts)
    image_paths = aggregate_image_paths(docs)
    return MultimodalContext(text=text, image_paths=image_paths)


def aggregate_image_paths(docs: List[Document]) -> List[str]:
    """Collect unique, non-empty image paths from `doc.metadata["image_paths"]`."""
    image_paths: List[str] = []
    seen: set[str] = set()
    for doc in docs:
        for path in doc.metadata.get("image_paths") or []:
            path_str = str(path).strip()
            if path_str and path_str not in seen:
                seen.add(path_str)
                image_paths.append(path_str)
    return image_paths


def load_images_from_paths(
    paths: List[str],
    max_images: int = 20,
) -> List[Any]:
    """Load images from paths, limit count, and return copies.
    Uses a context manager per file to avoid leaving file handles open.
    Missing or invalid files are skipped.
    """
    try:
        from PIL import Image
    except ImportError:
        logger.warning("PIL not available; cannot load images from paths.")
        return []

    loaded: List[Any] = []
    for path in paths[:max_images]:
        try:
            p = Path(path)
            if not p.exists():
                logger.debug("Image path does not exist: %s", path)
                continue
            with Image.open(path) as img:
                loaded.append(img.convert("RGB").copy())
        except Exception as e:
            logger.debug("Failed to load image %s: %s", path, e)
    return loaded

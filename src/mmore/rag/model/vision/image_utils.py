import base64
import io
from pathlib import Path
from typing import Any, List

from langchain_core.documents import Document
from PIL import Image


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


_MIN_IMAGE_SIDE = 32


def load_images_from_paths(
    paths: List[str], max_images: int = 20, max_side: int | None = None
) -> List[Any]:
    """Load images from paths, limit count, and return RGB copies."""
    loaded: List[Any] = []
    for path in paths[:max_images]:
        try:
            p = Path(path)
            if not p.is_file():
                continue
            with Image.open(p) as img:
                rgb = img.convert("RGB")
                width, height = rgb.size
                if width < _MIN_IMAGE_SIDE or height < _MIN_IMAGE_SIDE:
                    continue
                if max_side is not None and max(width, height) > max_side:
                    scale = max_side / max(width, height)
                    new_size = (
                        max(_MIN_IMAGE_SIDE, int(width * scale)),
                        max(_MIN_IMAGE_SIDE, int(height * scale)),
                    )
                    rgb = rgb.resize(new_size, Image.Resampling.LANCZOS)
                loaded.append(rgb.copy())
        except Exception:
            continue
    return loaded


def images_to_base64_data_urls(images: List[Any]) -> List[str]:
    """Convert PIL Images (or objects with save()) to data URLs."""
    urls = []
    for img in images:
        try:
            buf = io.BytesIO()
            if hasattr(img, "save"):
                img.save(buf, format="PNG")
            else:
                raise TypeError("Image object must have save() (e.g. PIL.Image)")
            b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
            urls.append(f"data:image/png;base64,{b64}")
        except Exception:
            pass
    return urls


def build_vision_content(text: str, image_data_urls: List[str]) -> List[dict]:
    """Build content list for chat multimodal messages."""
    content: List[dict] = [{"type": "text", "text": text}]
    for url in image_data_urls:
        content.append(
            {"type": "image_url", "image_url": {"url": url, "detail": "auto"}}
        )
    return content

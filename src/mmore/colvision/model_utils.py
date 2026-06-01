"""
Utilities for loading ColVision models (ColPali, ColQwen2, ColQwen2.5, ColQwen3, ColGemma3).

Provides automatic model/processor class resolution from a HuggingFace model name.
"""

import logging
import re
from typing import Tuple, Type

import torch

logger = logging.getLogger(__name__)

# Patterns checked in order — more specific patterns first.
_MODEL_REGISTRY = [
    (r"colqwen3", "ColQwen3", "ColQwen3Processor"),
    (r"colqwen2\.5|colqwen2_5", "ColQwen2_5", "ColQwen2_5_Processor"),
    (r"colqwen2", "ColQwen2", "ColQwen2Processor"),
    (r"colgemma|colnetra", "ColGemma3", "ColGemmaProcessor3"),
    (r"colsmol|colidefics3", "ColIdefics3", "ColIdefics3Processor"),
    (r"colpali", "ColPali", "ColPaliProcessor"),
]

SUPPORTED_MODELS = {
    "ColPali": ["vidore/colpali-v1.2", "vidore/colpali-v1.3"],
    "ColQwen2": ["vidore/colqwen2-v0.1", "vidore/colqwen2-v1.0"],
    "ColQwen2.5": ["vidore/colqwen2.5-v0.1", "vidore/colqwen2.5-v0.2"],
    "ColQwen3": ["vidore/colqwen3-v0.1"],
    "ColGemma3": ["Cognitive-Lab/ColNetraEmbed"],
    "ColSmol": ["vidore/colSmol-256M", "vidore/colSmol-500M"],
}


def get_device() -> str:
    """Select the available device for model inference."""
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def empty_device_cache(device: str) -> None:
    """Free cached memory for the active accelerator, if any."""
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


def resolve_model_classes(model_name: str) -> Tuple[Type, Type]:
    """
    Resolve the model and processor classes from the HuggingFace model name.

    Returns:
        Tuple of (ModelClass, ProcessorClass)

    Raises:
        ValueError: if the model name does not match any known pattern.
    """
    import colpali_engine.models as models_module

    name_lower = model_name.lower()

    for pattern, model_cls_name, proc_cls_name in _MODEL_REGISTRY:
        if re.search(pattern, name_lower):
            model_cls = getattr(models_module, model_cls_name)
            proc_cls = getattr(models_module, proc_cls_name)
            logger.info(
                f"Resolved model '{model_name}' → {model_cls_name} + {proc_cls_name}"
            )
            return model_cls, proc_cls

    supported_examples = {
        fam: examples[0] for fam, examples in SUPPORTED_MODELS.items()
    }
    raise ValueError(
        f"Unknown model '{model_name}'. Supported families: "
        f"{list(SUPPORTED_MODELS.keys())}. Examples: {supported_examples}"
    )


def _patched_key_mapping(model_cls: Type) -> dict | None:
    """Fill known gaps in colpali-engine's checkpoint key mappings (see inline
    comments for each gap). Returns a patched mapping to pass to from_pretrained,
    or None when no patch is needed. Each patch self-disables once the upstream
    mapping covers the missing entry.
    """
    # All gaps below stem from the transformers >= 5.3 layout rename, and the
    # `key_mapping` from_pretrained argument only exists on transformers 5.x.
    # On the colvision-legacy stack (transformers 4.x, ColPali v1.3) passing it
    # would be unsupported, so skip patching entirely there.
    from transformers import __version__ as _tf_version

    if tuple(int(p) for p in _tf_version.split(".")[:2]) < (5, 3):
        return None

    mapping: dict[str, str] = {}
    for cls in reversed(model_cls.__mro__):
        m = getattr(cls, "_checkpoint_conversion_mapping", None)
        if m:
            mapping.update(m)

    extra: dict[str, str] = {}

    # Gap 1: ColQwen2 / ColQwen2.5 — transformers >= 5.3 renamed the Qwen2-VL text
    # sub-module model.* -> language_model.*; colpali-engine remaps model.layers but
    # omits model.embed_tokens / model.norm, which then get randomly re-initialised.
    targets = "".join(mapping.values())
    if "language_model.layers" in targets and "language_model.embed_tokens" not in targets:
        extra[r"^model\.embed_tokens"] = "language_model.embed_tokens"
        extra[r"^model\.norm"] = "language_model.norm"
        logger.info(
            "Patching %s: embed_tokens/norm remap added "
            "(colpali-engine gap — self-disables once fixed upstream).",
            model_cls.__name__,
        )

    # Gap 2: ColGemma3 — vision tower extra nesting level
    # Guard: detect Gemma-based ColVision models via MRO (robust across
    # colpali-engine versions where Gemma3Model may or may not define a mapping).
    # Self-disables if a future colpali-engine release adds a vision_tower entry.
    gemma_based = any("Gemma" in cls.__name__ for cls in model_cls.__mro__)
    if gemma_based and not any("vision_tower" in k for k in mapping):
        extra[r"^vision_tower\.vision_model\."] = "vision_tower."
        logger.info(
            "Patching %s: vision_tower.vision_model.* → vision_tower.* remap added "
            "(colpali-engine gap — self-disables once fixed upstream).",
            model_cls.__name__,
        )

    if not extra:
        return None

    return {**mapping, **extra}


def load_model_and_processor(
    model_name: str,
    device: str,
    dtype: torch.dtype = torch.bfloat16,
):
    """Load a ColVision model and its processor, resolved from the model name.

    Returns:
        Tuple of (model, processor) ready for inference.
    """
    model_cls, proc_cls = resolve_model_classes(model_name)
    logger.info(f"Loading model: {model_name} ({model_cls.__name__})")
    from_pretrained_kwargs: dict = dict(torch_dtype=dtype, device_map=device)
    key_mapping = _patched_key_mapping(model_cls)
    if key_mapping is not None:
        from_pretrained_kwargs["key_mapping"] = key_mapping
    model = model_cls.from_pretrained(model_name, **from_pretrained_kwargs).eval()
    processor = proc_cls.from_pretrained(model_name)
    return model, processor

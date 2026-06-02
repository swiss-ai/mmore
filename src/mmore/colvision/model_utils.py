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


def _colvision_key_mapping(model_cls: Type) -> dict[str, str]:
    """Build the checkpoint-key renaming for a ColVision model class.

    Merges colpali-engine's own `_checkpoint_conversion_mapping` (collected across
    the MRO) with the gaps colpali omits (see inline comments). Returns an empty
    dict when nothing applies. Pure construction — no transformers-version logic.
    """
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
    if (
        "language_model.layers" in targets
        and "language_model.embed_tokens" not in targets
    ):
        extra[r"^model\.embed_tokens"] = "language_model.embed_tokens"
        extra[r"^model\.norm"] = "language_model.norm"

    # Gap 2: ColGemma3 — vision tower extra nesting level (vision_tower.vision_model.*).
    # Detect Gemma-based models via MRO (robust across colpali-engine versions).
    # Self-disables if a future colpali-engine release adds a vision_tower entry.
    gemma_based = any("Gemma" in cls.__name__ for cls in model_cls.__mro__)
    if gemma_based and not any("vision_tower" in k for k in mapping):
        extra[r"^vision_tower\.vision_model\."] = "vision_tower."

    return {**mapping, **extra}


def _register_checkpoint_conversions(model_cls: Type) -> None:
    """Register the checkpoint-key renaming in transformers' conversion cache so it
    is applied on BOTH load paths — the base weights AND the PEFT/LoRA adapter weights.

    ColVision checkpoints (e.g. colqwen2.5-v0.2) are LoRA adapters. transformers >= 5.3
    loads adapter weights through `load_adapter`, which consults ONLY this registered
    conversion cache — not the runtime `key_mapping` argument colpali-engine passes.
    Without registering, the model.* -> language_model.* rename never reaches the LoRA
    weights, so they are silently re-initialised at random and the embeddings are wrong.

    No-op on transformers < 5.3 (the colvision-legacy stack): there the layout rename
    does not apply and the registration API does not exist.
    """
    from transformers import __version__ as _tf_version

    if tuple(int(p) for p in _tf_version.split(".")[:2]) < (5, 3):
        return

    mapping = _colvision_key_mapping(model_cls)
    if not mapping:
        return

    from transformers.conversion_mapping import register_checkpoint_conversion_mapping
    from transformers.core_model_loading import WeightRenaming

    register_checkpoint_conversion_mapping(
        model_cls.__name__,
        [
            WeightRenaming(source_patterns=k, target_patterns=v)
            for k, v in mapping.items()
        ],
        overwrite=True,  # idempotent: load_model_and_processor may run many times
    )
    logger.info(
        "Registered checkpoint key conversions for %s (covers base + LoRA adapter weights).",
        model_cls.__name__,
    )


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
    _register_checkpoint_conversions(model_cls)
    model = model_cls.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device
    ).eval()
    processor = proc_cls.from_pretrained(model_name)
    return model, processor

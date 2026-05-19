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
]

_DEFAULT_MODEL_CLASS = "ColPali"
_DEFAULT_PROCESSOR_CLASS = "ColPaliProcessor"

SUPPORTED_MODELS = {
    "ColPali": ["vidore/colpali-v1.2", "vidore/colpali-v1.3"],
    "ColQwen2": ["vidore/colqwen2-v0.1", "vidore/colqwen2-v1.0"],
    "ColQwen2.5": ["vidore/colqwen2.5-v0.1", "vidore/colqwen2.5-v0.2"],
    "ColQwen3": ["vidore/colqwen3-v0.1"],
    "ColGemma3": ["Cognitive-Lab/ColNetraEmbed"],
}


def resolve_model_classes(model_name: str) -> Tuple[Type, Type]:
    """
    Resolve the model and processor classes from the HuggingFace model name.

    Falls back to ColPali if the model name does not match any known pattern.

    Returns:
        Tuple of (ModelClass, ProcessorClass)
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

    # Default to ColPali
    model_cls = getattr(models_module, _DEFAULT_MODEL_CLASS)
    proc_cls = getattr(models_module, _DEFAULT_PROCESSOR_CLASS)
    logger.warning(
        f"Unknown model '{model_name}', falling back to ColPali. "
        f"Supported: {list(SUPPORTED_MODELS.keys())}"
    )
    return model_cls, proc_cls


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
    model = model_cls.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device,
    ).eval()
    processor = proc_cls.from_pretrained(model_name)
    return model, processor

from typing import Dict, Type

from .arxiv import ArxivAdapter
from .base import SourceAdapter
from .europepmc import EuropePmcAdapter
from .openalex import OpenAlexAdapter

# from .google_scholar import GoogleScholarAdapter

REGISTRY: Dict[str, Type[SourceAdapter]] = {
    "openalex": OpenAlexAdapter,
    "europepmc": EuropePmcAdapter,
    "arxiv": ArxivAdapter,
    # "google_scholar": GoogleScholarAdapter,
}


def get_adapter(name: str, **kwargs) -> SourceAdapter:
    if name not in REGISTRY:
        raise ValueError(f"Unknown source: {name!r}. Available: {list[str](REGISTRY)}")
    return REGISTRY[name](**kwargs)

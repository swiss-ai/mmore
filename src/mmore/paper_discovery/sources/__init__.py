"""Source adapter registry.

Each entry in `REGISTRY` maps a config-facing name (e.g. `"arxiv"`) to the
adapter class that implements `SourceAdapter`. `get_adapter()` instantiates
one with the constructor kwargs forwarded from `PaperDiscoveryConfig`.

To add a new source: implement the `SourceAdapter` protocol in a new module
and register it here.

Google Scholar is opt-in - the adapter itself defers importing `scholarly`
to its `search()` call, so registering the class unconditionally is safe.
If a user enables `google_scholar` in their config without installing
`scholarly`, the adapter logs a warning and returns `[]`.
"""

from typing import Dict, Type

from .arxiv import ArxivAdapter
from .base import SourceAdapter
from .europepmc import EuropePmcAdapter
from .google_scholar import GoogleScholarAdapter
from .openalex import OpenAlexAdapter

REGISTRY: Dict[str, Type[SourceAdapter]] = {
    "openalex": OpenAlexAdapter,
    "europepmc": EuropePmcAdapter,
    "arxiv": ArxivAdapter,
    "google_scholar": GoogleScholarAdapter,
}


def get_adapter(name: str, **kwargs) -> SourceAdapter:
    """Instantiate the adapter registered under `name`.

    Args:
      name:    Source key from `PaperDiscoveryConfig.sources`
               (e.g. `"openalex"`, `"arxiv"`).
      kwargs:  Forwarded to the adapter constructor — typically
               `user_agent`, `max_pages`, `max_results`, and source-specific
               extras (e.g. `category_map`, `enable_pair_query` for arXiv).

    Raises:
      ValueError: if `name` is not in `REGISTRY`.
    """
    if name not in REGISTRY:
        raise ValueError(f"Unknown source: {name!r}. Available: {list(REGISTRY)}")
    return REGISTRY[name](**kwargs)

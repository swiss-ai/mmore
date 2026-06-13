"""Source adapter protocol. EVery adapter follows the same shape:

    adapter = MySourceAdapter(user_agent=..., max_pages=..., max_results=...)
    papers = adapter.search(query, category_title)

Adapters MUST NOT raise on network errors. Return [] and log instead
"""

from typing import List, Protocol

from ..schema import Paper


class SourceAdapter(Protocol):
    name: str

    def search(self, query: str, category_title: str) -> List[Paper]: ...

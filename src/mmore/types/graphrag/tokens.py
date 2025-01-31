from typing import Protocol


class TokenCounter(Protocol):
    def count_tokens(self, text: str) -> int: ...

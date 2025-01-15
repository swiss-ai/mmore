"""Counter for Tiktoken based tokens."""

import tiktoken

from mmore.types.graphrag.tokens import TokenCounter


class TiktokenCounter(TokenCounter):
    def __init__(self, encoding_name: str = "cl100k_base"):
        self.tokenizer = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

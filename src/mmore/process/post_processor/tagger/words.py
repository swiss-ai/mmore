from .base import BaseTagger


class WordsCounter(BaseTagger):
    def __init__(
        self, name: str = "🔤 Words Counter", metadata_key: str = "word_count"
    ):
        super().__init__(name, metadata_key)

    def tag(self, sample):
        return len(sample.text.split())

from .base import BaseTagger
from langdetect import detect


class LangDetector(BaseTagger):
    def __init__(self, name: str = "🗣️ Lang Detector", metadata_key: str = "lang"):
        super().__init__(name, metadata_key)

    def tag(self, sample):
        text = sample.text.replace("<attachment>", "")

        try:
            lang = detect(text)
        except:
            lang = "unknown"

        return lang

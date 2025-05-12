from langdetect import detect

from .base import BaseTagger


class LangDetector(BaseTagger):
    def __init__(self, name: str = "🗣️ Lang Detector", metadata_key: str = "lang"):
        super().__init__(name, metadata_key)

    def tag(self, sample):
        text = sample.text.replace("<attachment>", "")

        try:
            lang = detect(text)
        except Exception:
            lang = "unknown"

        return lang

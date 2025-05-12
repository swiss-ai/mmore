import hashlib

from .base import BaseTagger


class TextHash(BaseTagger):
    def __init__(self, name: str = "#️⃣ Auto ID", metadata_key: str = "hash"):
        super().__init__(name, metadata_key)

    def tag(self, sample):
        return TextHash.hash(sample.text.replace("<attachment>", ""))

    @staticmethod
    def hash(text: str):
        return hashlib.md5(text.encode()).hexdigest()

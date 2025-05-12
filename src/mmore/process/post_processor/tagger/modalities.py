from .base import BaseTagger


class ModalitiesCounter(BaseTagger):
    def __init__(
        self, name: str = "📸 Modalities Counter", metadata_key="modalities_count"
    ):
        super().__init__(name, metadata_key)

    def tag(self, sample):
        return len(sample.modalities)

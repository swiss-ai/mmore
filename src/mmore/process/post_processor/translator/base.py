from typing import List
from mmore.type import MultimodalSample
from mmore.process.post_processor.base import BasePostProcessor
from langid import classify
import argostranslate.package
import argostranslate.translate

from dataclasses import dataclass


@dataclass
class TranslatorConfig:
    target_language: str
    attachment_tag: str


class TranslatorPostProcessor(BasePostProcessor):
    def __init__(self, target_language: str, attachment_tag: str):
        super().__init__(name="🌍 Translator")
        self.target_language = target_language
        self.attachment_tag = attachment_tag
        self.updated_packages = set()

    @classmethod
    def from_config(cls, config: TranslatorConfig):
        translator = TranslatorPostProcessor(
            target_language=config.target_language, attachment_tag=config.attachment_tag
        )
        return translator

    def process(
        self, sample: MultimodalSample, **kwargs
    ) -> MultimodalSample | List[MultimodalSample]:
        from_code, confidence = classify(sample.text)

        # Install package if needed
        self._update_package(from_code)

        # Split text to avoid attachment tag being translated
        splitted_texts = sample.text.split(self.attachment_tag)

        translatedTexts = []
        for text in splitted_texts:
            translatedTexts.append(
                argostranslate.translate.translate(
                    text, from_code, self.target_language
                )
            )

            translatedText = self.attachment_tag.join(translatedTexts)

        return MultimodalSample(translatedText, sample.modalities)

    def _update_package(self, from_code: str):
        if from_code in self.updated_packages:
            return

        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()
        package_to_install = next(
            filter(
                lambda x: x.from_code == from_code
                and x.to_code == self.target_language,
                available_packages,
            )
        )
        argostranslate.package.install_from_path(package_to_install.download())

        # Add source language to updated packages (lazy)
        self.updated_packages.add(from_code)

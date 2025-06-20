from typing import List, Optional
from mmore.type import MultimodalSample
from mmore.process.post_processor.base import BasePostProcessor
from langid import classify
from langid.langid import LanguageIdentifier, model
import argostranslate.package
import argostranslate.translate

from dataclasses import dataclass


@dataclass
class TranslatorConfig:
    target_language: str
    attachment_tag: str
    confidence_threshold: float
    constrained_languages: Optional[List[str]] = None


class TranslatorPostProcessor(BasePostProcessor):
    def __init__(self, target_language: str, attachment_tag: str, confidence_threshold: float,
                 constrained_languages: Optional[List[str]] = None):
        super().__init__(name="🌍 Translator")
        self.target_language = target_language
        self.attachment_tag = attachment_tag
        self.updated_packages = set()
        self.confidence_threshold = confidence_threshold
        self.classifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
        self.classifier.set_languages(constrained_languages)


    @classmethod
    def from_config(cls, config: TranslatorConfig):
        translator = TranslatorPostProcessor(
            target_language=config.target_language, attachment_tag=config.attachment_tag,
            confidence_threshold=config.confidence_threshold,
            constrained_languages=config.constrained_languages
        )
        return translator

    def process(
        self, sample: MultimodalSample, **kwargs
    ) -> List[MultimodalSample]:
        from_code, confidence = self.classifier.classify(sample.text)

        # If the sample is already in the right language, do nothing
        if from_code == self.target_language or confidence <= self.confidence_threshold:
            return [sample]

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
        return [MultimodalSample(
            text=translatedText, 
            modalities=sample.modalities,
            metadata={"original_text" : sample.text, **sample.metadata}
        )]


    def _update_package(self, from_code: str):
        if from_code in self.updated_packages:
            return

        argostranslate.package.update_package_index()
        available_packages = argostranslate.package.get_available_packages()

        print(from_code, self.target_language)

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

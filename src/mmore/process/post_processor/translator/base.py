from dataclasses import dataclass
from typing import List, Optional

import argostranslate.package
import argostranslate.translate
from langid.langid import LanguageIdentifier, model

from mmore.process.post_processor.base import BasePostProcessor
from mmore.type import MultimodalSample


@dataclass
class TranslatorConfig:
    """
    Configuration for the TranslatorPostProcessor.
    Attributes:
        target_language (str): The language code to translate text into.
        attachment_tag (str): A tag used to identify modalities placeholders in the text
        confidence_threshold (float): Minimum confidence level for translation to be applied.
        constrained_languages (Optional[List[str]]): List of languages to constrain the classifier to.
    """

    target_language: str
    attachment_tag: str
    confidence_threshold: float
    constrained_languages: Optional[List[str]] = None


class TranslatorPostProcessor(BasePostProcessor):
    """
    A post-processor that translates text in multimodal samples to a specified target language.
    This post-processor uses the Argos Translate library to perform translations and can handle
    text with specific attachment tags that should not be translated.
    It also includes a language classifier to determine the source language of the text.
    Attributes:
        target_language (str): The language code to translate text into.
        attachment_tag (str): A tag used to identify parts of the text that should not be translated.
        updated_packages (set): A set of updated language packages to avoid redundant updates.
        confidence_threshold (float): Minimum confidence level for translation to be applied.
        classifier (LanguageIdentifier): Language identifier for detecting the source language.
        constrained_languages (Optional[List[str]]): List of languages to constrain the classifier to.
    """

    def __init__(
        self,
        target_language: str,
        attachment_tag: str,
        confidence_threshold: float,
        constrained_languages: Optional[List[str]] = None,
    ):
        """
        Initializes the TranslatorPostProcessor.

        Args:
            target_language (str): The language code to translate text into.
            attachment_tag (str): A tag used to identify parts of the text that should not be translated.
            confidence_threshold (float): Minimum confidence level for translation to be applied.
            constrained_languages (Optional[List[str]]): List of languages to constrain the classifier to.
        """
        super().__init__(name="🌍 Translator")
        self.target_language = target_language
        self.attachment_tag = attachment_tag
        self.updated_packages = set()
        self.confidence_threshold = confidence_threshold
        self.classifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
        self.classifier.set_languages(constrained_languages)

    @classmethod
    def from_config(cls, config: TranslatorConfig):
        """
        Creates an instance of TranslatorPostProcessor from a configuration object.

        Args:
            config (TranslatorConfig): Configuration object containing parameters for the translator.

        Returns:
            TranslatorPostProcessor: An instance of the translator post-processor.
        """
        translator = TranslatorPostProcessor(
            target_language=config.target_language,
            attachment_tag=config.attachment_tag,
            confidence_threshold=config.confidence_threshold,
            constrained_languages=config.constrained_languages,
        )
        return translator

    def process(self, sample: MultimodalSample, **kwargs) -> List[MultimodalSample]:
        from_code, confidence = self.classifier.classify(sample.text)

        # If the sample is already in the right language, do nothing
        if from_code == self.target_language or confidence <= self.confidence_threshold:
            return [sample]

        # Install package if needed
        self._update_package(from_code)

        # Split text to avoid attachment tag being translated
        splitted_texts = sample.text.split(self.attachment_tag)

        translated_texts = []
        for text in splitted_texts:
            translated_texts.append(
                argostranslate.translate.translate(
                    text, from_code, self.target_language
                )
            )

        translated_text = self.attachment_tag.join(translated_texts)
        return [
            MultimodalSample(
                text=translated_text,
                modalities=sample.modalities,
                metadata={"original_text": sample.text, **sample.metadata},
            )
        ]

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

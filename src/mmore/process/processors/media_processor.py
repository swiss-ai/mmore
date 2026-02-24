import logging
import os
import tempfile
from typing import List, Optional

import torch
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image
from transformers.pipelines import pipeline as pipeline_t

from ...type import FileDescriptor, MultimodalSample
from .base import MediaProcessorConfig, Processor, ProcessorConfig

logger = logging.getLogger(__name__)


class MediaProcessor(Processor):
    """Processor for audio and video files.

    Transcribes audio/video using a Whisper model and extracts frames from
    video files.

    Supported extensions: ``.mp4``, ``.avi``, ``.mov``, ``.mkv``,
    ``.mp3``, ``.flac``, ``.wav``.

    Configuration (:class:`~.base.MediaProcessorConfig`):

    .. code-block:: yaml

        processors:
          MediaProcessor:
            normal_model: openai/whisper-large-v3-turbo
            fast_model: openai/whisper-tiny
            frame_sample_rate: 10
    """

    CONFIG_CLASS = MediaProcessorConfig

    @staticmethod
    def _get_available_devices():
        if torch.cuda.is_available():
            return [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
        return [torch.device("cpu")]

    devices = _get_available_devices()
    pipelines: list = []

    def __init__(self, config: Optional[ProcessorConfig] = None) -> None:
        super().__init__(config=config or MediaProcessorConfig())
        # Ensure config is a MediaProcessorConfig for typed access
        if not isinstance(self.config, MediaProcessorConfig):
            self.config = MediaProcessorConfig(
                output_path=self.config.output_path,
                extract_images=self.config.extract_images,
                attachment_tag=self.config.attachment_tag,
                dashboard_backend_url=self.config.dashboard_backend_url,
            )

    @classmethod
    def accepts(cls, file: FileDescriptor) -> bool:
        return file.file_extension.lower() in [
            ".mp4",
            ".avi",
            ".mov",
            ".mkv",
            ".mp3",
            ".flac",
            ".wav",
        ]

    def _load_pipelines(self, fast_mode: bool = False) -> None:
        """Load Whisper pipelines onto all available devices.

        Args:
            fast_mode: Use :attr:`~MediaProcessorConfig.fast_model` if
                ``True``, otherwise use
                :attr:`~MediaProcessorConfig.normal_model`.
        """
        assert isinstance(self.config, MediaProcessorConfig)
        model_name = self.config.fast_model if fast_mode else self.config.normal_model
        try:
            MediaProcessor.pipelines = []
            for device in MediaProcessor.devices:
                pipe = pipeline_t(
                    "automatic-speech-recognition",
                    model=model_name,
                    device=device,
                    return_timestamps=True,
                )
                MediaProcessor.pipelines.append(pipe)
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            MediaProcessor.pipelines = []

    @staticmethod
    def load_models(fast_mode: bool = False) -> None:
        """Pre-load Whisper pipelines with default model names.

        Called by :class:`~.base.ProcessorRegistry` when ``preload=True``.
        For instance-level model loading (with config-specified model names)
        the processor calls :meth:`_load_pipelines` instead.

        Args:
            fast_mode: Load the small/fast model if ``True``.
        """
        model_name = (
            "openai/whisper-tiny" if fast_mode else "openai/whisper-large-v3-turbo"
        )
        try:
            MediaProcessor.pipelines = []
            for device in MediaProcessor.devices:
                pipe = pipeline_t(
                    "automatic-speech-recognition",
                    model=model_name,
                    device=device,
                    return_timestamps=True,
                )
                MediaProcessor.pipelines.append(pipe)
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            MediaProcessor.pipelines = []

    def process_batch(
        self, files_paths: List[str], fast_mode: bool = False, num_workers: int = 1
    ) -> List[MultimodalSample]:
        if not self.pipelines:
            self._load_pipelines(fast_mode=fast_mode)
        if not self.pipelines:
            raise RuntimeError("Failed to load any processing pipelines.")

        file_chunks = self.evenly_split_across_gpus(files_paths, len(self.devices))

        results = []
        for pipeline, chunk in zip(self.pipelines, file_chunks):
            for file in chunk:
                try:
                    result = self._process_file(file, pipeline, fast_mode)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing {file}: {e}")
        return results

    def _process_file(self, file_path: str, pipeline, fast_mode: bool) -> MultimodalSample:
        all_text = self._extract_text(file_path, pipeline)
        images = self._extract_images(file_path) if self.config.extract_images else []
        return self.create_sample([all_text], images, {"file_path": file_path})

    def process(self, file_path: str) -> MultimodalSample:
        if not self.pipelines:
            self._load_pipelines(fast_mode=False)
        if not self.pipelines:
            raise RuntimeError("Failed to load any processing pipelines.")
        pipeline = self.pipelines[0]
        all_text = self._extract_text(file_path, pipeline)
        images = self._extract_images(file_path) if self.config.extract_images else []
        return self.create_sample([all_text], images, {"file_path": file_path})

    def _extract_text(self, file_path: str, pipeline) -> str:
        def _prepare_audio_file(file_path: str, ext: str, temp_audio):
            try:
                if ext in [".mp4", ".avi", ".mov", ".mkv"]:
                    with VideoFileClip(file_path) as clip:
                        if clip.audio is None:
                            raise ValueError("No audio track found in video.")
                        clip.audio.write_audiofile(temp_audio.name, codec="pcm_s16le")
                elif ext in [".mp3", ".flac", ".wav"]:
                    with AudioFileClip(file_path) as audio_clip:
                        audio_clip.write_audiofile(temp_audio.name, codec="pcm_s16le")
                temp_audio.flush()
            except Exception as e:
                logger.error(f"Error preparing audio file {file_path}: {e}")
                raise e

        ext = os.path.splitext(file_path)[1].lower()
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio:
                _prepare_audio_file(file_path, ext, temp_audio)
                result = pipeline(temp_audio.name)
                return result.get("text", "")
        except Exception as e:
            logger.error(f"Error transcribing {file_path}: {e}")
            return ""

    def _extract_images(self, file_path: str) -> List[Image.Image]:
        def _extract_video_frames(file_path: str) -> List[Image.Image]:
            images = []
            assert isinstance(self.config, MediaProcessorConfig)
            try:
                with VideoFileClip(file_path) as clip:
                    if clip.duration is None or clip.duration <= 0:
                        raise ValueError("Invalid video duration.")
                    duration = clip.duration
                    sample_rate = self.config.frame_sample_rate
                    num_thumbnails = max(1, int(duration / sample_rate))
                    for i in range(num_thumbnails):
                        t = min(i * sample_rate, duration - 0.1)
                        frame = clip.get_frame(t)
                        image = Image.fromarray(frame).convert("RGB")
                        images.append(image)
                logger.info(f"Extracted {len(images)} images from {file_path}.")
            except Exception as e:
                logger.error(f"Error extracting images from {file_path}: {e}")
            return images

        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".mp3", ".flac", ".wav"]:
            logger.info(f"No images to extract from {file_path}.")
            return []
        return _extract_video_frames(file_path)

    @staticmethod
    def evenly_split_across_gpus(x_list: list, num_gpus: int) -> list:
        """Split *x_list* as evenly as possible across *num_gpus* chunks."""
        x_per_gpu = len(x_list) // num_gpus
        remainder = len(x_list) % num_gpus
        chunks = []
        start = 0
        for i in range(num_gpus):
            end = start + x_per_gpu + (1 if i < remainder else 0)
            chunks.append(x_list[start:end])
            start = end
        return chunks

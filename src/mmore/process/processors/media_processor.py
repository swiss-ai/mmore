import os
import logging
import tempfile
import torch
from typing import List
from PIL import Image
from transformers import pipeline
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from src.mmore.type import FileDescriptor
from .processor import Processor
from src.mmore.process.utils import create_sample, evenly_split_across_gpus

logger = logging.getLogger(__name__)


class MediaProcessor(Processor):
    def __init__(self, files, config=None):
        super().__init__(files, config=config)
        self.device = None
        self.transcription_pipeline = None

    def load_models(self, device=None, fast_mode=False):
        print("Loading models", device, fast_mode)
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        model_name = self.config.custom_config.get("normal_model",
                                                   "openai/whisper-large-v3-turbo") if fast_mode else self.config.custom_config.get(
            "fast_model", "openai/whisper-tiny")

        try:
            self.transcription_pipeline = pipeline(
                self.config.custom_config.get("type", "automatic-speech-recognition"),
                model=model_name,
                device=self.device,
                return_timestamps=True,
            )
            logger.info(
                f"MediaProcessor: {model_name} model loaded successfully on device {device}."
            )
        except Exception as e:
            logger.error(
                f"MediaProcessor: Error loading Hugging Face Whisper model on device {device}: {e}"
            )
            self.transcription_pipeline = None

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

    def require_gpu(self) -> bool:
        return True, True

    def process_implementation(self, file_path):
        return self._process(file_path, fast_mode=False)

    def process_fast_implementation(self, file_path):
        return self._process(file_path, fast_mode=True)

    def _process(self, file_path: str, fast_mode=False):
        text = self._extract_text(file_path, fast_mode=fast_mode)
        images = self._extract_images(file_path)
        return create_sample([text], images, file_path)

    def split_files_across_gpus(self):
        """Split the files evenly across the available GPUs."""
        num_gpus = torch.cuda.device_count()
        return evenly_split_across_gpus(self.files, num_gpus)

    def _extract_text(self, file_path: str, fast_mode=False) -> str:
        def _prepare_audio_file(file_path: str, ext: str, temp_audio):
            """
            Extracts audio from a video or reads an audio file for transcription.
            """
            try:
                if ext in [".mp4", ".avi", ".mov", ".mkv"]:
                    with VideoFileClip(file_path) as clip:
                        if clip.audio is None:
                            raise ValueError("No audio track found in video.")
                        AudioFileClip(file_path).write_audiofile(temp_audio.name, codec="pcm_s16le")
                elif ext in [".mp3", ".flac", ".wav"]:
                    with open(file_path, "rb") as f_in:
                        temp_audio.write(f_in.read())
                    temp_audio.flush()
            except Exception as e:
                logger.error(f"Error preparing audio file {file_path}: {e}")
                raise

        if not self.transcription_pipeline:
            self.load_models(device=self.device, fast_mode=fast_mode)
        if not self.transcription_pipeline:
            logger.error("Transcription pipeline not initialized.")
            return ""

        ext = os.path.splitext(file_path)[1].lower()
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio:
                _prepare_audio_file(file_path, ext, temp_audio)
                result = self.transcription_pipeline(temp_audio.name)
                transcription = result.get("text", "")
                logger.debug(f"Transcription result for {file_path}: {transcription}")
                return transcription
        except Exception as e:
            logger.error(f"Error transcribing file {file_path}: {e}")
            return ""

    def _extract_images(self, file_path: str) -> List[Image.Image]:
        def _extract_video_frames(file_path: str) -> List[Image.Image]:
            images = []
            try:
                clip = AudioFileClip(file_path)
                if clip.duration is None or clip.duration <= 0:
                    logger.error(f"Failed to retrieve duration for video {file_path}")
                    raise ValueError("Failed to retrieve video duration.")

                duration = clip.duration
                sample_rate = self.config.custom_config.get("sample_rate", 10)
                num_thumbnails = max(1, int(duration / sample_rate))
                for i in range(num_thumbnails):
                    t = min(
                        i * sample_rate, duration - 0.1
                    )  # avoid querying past the video duration
                    frame = clip.get_frame(t)
                    image = Image.fromarray(frame).convert("RGB")
                    images.append(image)
                clip.close()
                logger.info(
                    f"MediaProcessor: Extracted {len(images)} images from {file_path}."
                )
            except Exception as e:
                logger.error(
                    f"MediaProcessor: Error extracting images from {file_path}: {e}"
                )
            return images

        ext = os.path.splitext(file_path)[1].lower()

        if ext not in [".mp3", ".flac", ".wav"]:
            logger.info(
                f"MediaProcessor: No images to extract from the file {file_path}."
            )
            return []

        return _extract_video_frames(file_path)

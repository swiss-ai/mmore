import os
import logging
import tempfile
import torch
from typing import List
from PIL import Image
from transformers import pipeline
from accelerate import Accelerator
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from src.mmore.type import FileDescriptor, MultimodalSample
from .processor import Processor

logger = logging.getLogger(__name__)

class MediaProcessor(Processor):
    def __init__(self, config=None):
        super().__init__(config=config)
        self.device = None
        self.transcription_pipeline = None
        self.accelerator = Accelerator()  

    def load_models(self, device=None, fast_mode=False):
        """Load models using accelerate's device handling"""
        self.device = self.accelerator.device
        model_name = (self.config.custom_config.get("fast_model", "openai/whisper-tiny") 
                     if fast_mode else 
                     self.config.custom_config.get("normal_model", "openai/whisper-large-v3"))
        
        try:
            self.transcription_pipeline = pipeline(
                self.config.custom_config.get("type", "automatic-speech-recognition"),
                model=model_name,
                device=self.device,
                return_timestamps=True,
            )
            # prepare pipeline for distributed inference
            self.transcription_pipeline = self.accelerator.prepare(self.transcription_pipeline)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.transcription_pipeline = None

    def process_batch(self, files, fast_mode, num_workers):
        if fast_mode:
            return super().process_batch(files, fast_mode, num_workers)
        
        # split files across processes
        with self.accelerator.split_between_processes(files) as files_chunk:
            if not self.transcription_pipeline:
                self.load_models(fast_mode=False)
            
            results = []
            for file_path in files_chunk:
                try:
                    sample = self.process(file_path, fast=False)
                    results.append(sample)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
            
            # gather results from all processes
            gathered_results = self.accelerator.gather(results)
            return gathered_results

    @classmethod
    def accepts(cls, file: FileDescriptor) -> bool: 
        """
        Returns:
            bool: True if the file is a supported media format, False otherwise.
        """
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
        """
        Returns:
            tuple: A tuple (True) indicating GPU requirement for both standard and fast modes.
        """
        return True

    def process(self, file_path: str, fast: bool = False) -> MultimodalSample:
        """
        Process a media file in standard mode.

        Args:
            file_path (str): Path to the media file.

        Returns:
            dict: A dictionary containing the transcription and extracted images.
        """
        text = self._extract_text(file_path, fast_mode=fast)
        images = self._extract_images(file_path)
        return self.create_sample([text], images, file_path)

    def _extract_text(self, file_path: str, fast_mode=False) -> str:
        """
        Extract transcription text from a media file.

        Args:
            file_path (str): Path to the media file.
            fast_mode (bool): Whether to use fast processing mode.

        Returns:
            str: Transcription text.
        """
        def _prepare_audio_file(file_path: str, ext: str, temp_audio):
            """
            Prepare the audio file for transcription by extracting or converting audio.

            Args:
                file_path (str): Path to the media file.
                ext (str): File extension of the media file.
                temp_audio: Temporary file object for audio storage.
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
        """
        Extract frames as images from a video file.

        Args:
            file_path (str): Path to the media file.

        Returns:
            list: List of extracted PIL images.
        """
        def _extract_video_frames(file_path: str) -> List[Image.Image]:
            """
            Extract video frames at intervals specified by the sample rate.

            Args:
                file_path (str): Path to the video file.

            Returns:
                list: List of extracted PIL images.
            """
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

        if ext in [".mp3", ".flac", ".wav"]:
            logger.info(
                f"MediaProcessor: No images to extract from the file {file_path}."
            )
            return []

        return _extract_video_frames(file_path)

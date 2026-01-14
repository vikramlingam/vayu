"""
Vayu STT Service
Wrapper for faster-whisper with LOCAL model loading.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from typing import Generator
from faster_whisper import WhisperModel
from colorama import Fore, Style

import config
from src.interfaces import ISTTService


class STTService(ISTTService):
    """
    Speech-to-Text service using faster-whisper.
    CRITICAL: Loads model from LOCAL path, not from the internet.
    """
    
    def __init__(self):
        """Initialize the STT service with a local model."""
        print(f"{Fore.CYAN}[STT] Loading model from: {config.STT_MODEL_PATH}{Style.RESET_ALL}")
        
        # CRITICAL: Pass model_path_or_size as string path to load LOCAL model
        self.model = WhisperModel(
            model_size_or_path=str(config.STT_MODEL_PATH),
            device=config.STT_DEVICE,
            compute_type=config.STT_COMPUTE_TYPE,
        )
        
        self.language = config.STT_LANGUAGE
        self.beam_size = config.STT_BEAM_SIZE
        
        print(f"{Fore.GREEN}[STT] Model loaded successfully ✓{Style.RESET_ALL}")
    
    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio data as numpy array (float32, 16kHz, mono)
            
        Returns:
            Transcribed text string
        """
        if audio is None or len(audio) == 0:
            return ""
        
        # Ensure audio is float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Normalize if needed
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()
        
        segments, info = self.model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=True,  # Use VAD to filter silence
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=300,
            )
        )
        
        # Collect all segment texts
        transcription = " ".join(segment.text.strip() for segment in segments)
        
        return transcription.strip()
    
    def transcribe_stream(self, audio_stream: Generator[np.ndarray, None, None]) -> Generator[str, None, None]:
        """
        Transcribe streaming audio to text.
        
        For faster-whisper, we accumulate chunks and transcribe periodically.
        
        Args:
            audio_stream: Generator yielding audio chunks
            
        Yields:
            Partial transcription strings
        """
        buffer = []
        buffer_duration = 0.0
        min_duration = 1.0  # Minimum seconds before attempting transcription
        
        for chunk in audio_stream:
            buffer.append(chunk)
            buffer_duration += len(chunk) / config.SAMPLE_RATE
            
            if buffer_duration >= min_duration:
                # Concatenate buffer and transcribe
                audio = np.concatenate(buffer)
                transcription = self.transcribe(audio)
                
                if transcription:
                    yield transcription
                
                # Keep last 0.5 seconds for context overlap
                overlap_samples = int(0.5 * config.SAMPLE_RATE)
                if len(audio) > overlap_samples:
                    buffer = [audio[-overlap_samples:]]
                    buffer_duration = 0.5
                else:
                    buffer = []
                    buffer_duration = 0.0
        
        # Process remaining buffer
        if buffer:
            audio = np.concatenate(buffer)
            transcription = self.transcribe(audio)
            if transcription:
                yield transcription


if __name__ == "__main__":
    # Test the STT service
    print("Testing STT Service...")
    stt = STTService()
    
    # Create a test audio signal (silence)
    test_audio = np.zeros(16000, dtype=np.float32)
    result = stt.transcribe(test_audio)
    print(f"Test result (silence): '{result}'")

"""
Vayu Interfaces Module
Abstract Base Classes defining the Strategy Pattern for all services.
"""

from abc import ABC, abstractmethod
from typing import Generator, AsyncGenerator, Optional
import numpy as np


class ISTTService(ABC):
    """Interface for Speech-to-Text services."""
    
    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> str:
        """
        Transcribe audio to text.
        
        Args:
            audio: Audio data as numpy array (float32, 16kHz, mono)
            
        Returns:
            Transcribed text string
        """
        pass
    
    @abstractmethod
    def transcribe_stream(self, audio_stream: Generator[np.ndarray, None, None]) -> Generator[str, None, None]:
        """
        Transcribe streaming audio to text.
        
        Args:
            audio_stream: Generator yielding audio chunks
            
        Yields:
            Partial transcription strings
        """
        pass


class ITTSService(ABC):
    """Interface for Text-to-Speech services."""
    
    @abstractmethod
    def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesize text to audio.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Audio data as numpy array
        """
        pass
    
    @abstractmethod
    def synthesize_stream(self, text: str) -> Generator[np.ndarray, None, None]:
        """
        Synthesize text to streaming audio.
        
        Args:
            text: Text to synthesize
            
        Yields:
            Audio chunks as numpy arrays
        """
        pass
    
    @abstractmethod
    def get_sample_rate(self) -> int:
        """Get the sample rate of synthesized audio."""
        pass


class ILLMService(ABC):
    """Interface for Large Language Model services."""
    
    @abstractmethod
    async def generate(self, prompt: str, context: Optional[list] = None) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User's input prompt
            context: Optional conversation context
            
        Returns:
            Generated response string
        """
        pass
    
    @abstractmethod
    async def generate_stream(self, prompt: str, context: Optional[list] = None) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response from the LLM.
        
        Args:
            prompt: User's input prompt
            context: Optional conversation context
            
        Yields:
            Partial response strings as they're generated
        """
        pass


class IAudioManager(ABC):
    """Interface for Audio I/O management."""
    
    @abstractmethod
    def start_recording(self) -> None:
        """Start recording audio from the microphone."""
        pass
    
    @abstractmethod
    def stop_recording(self) -> None:
        """Stop recording audio."""
        pass
    
    @abstractmethod
    def get_audio_chunk(self) -> Optional[np.ndarray]:
        """
        Get the next audio chunk from the recording buffer.
        
        Returns:
            Audio chunk as numpy array, or None if no audio available
        """
        pass
    
    @abstractmethod
    def play_audio(self, audio: np.ndarray, sample_rate: int) -> None:
        """
        Play audio through the speakers.
        
        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of the audio
        """
        pass
    
    @abstractmethod
    def play_audio_stream(self, audio_stream: Generator[np.ndarray, None, None], sample_rate: int) -> None:
        """
        Play streaming audio through the speakers.
        
        Args:
            audio_stream: Generator yielding audio chunks
            sample_rate: Sample rate of the audio
        """
        pass
    
    @abstractmethod
    def interrupt(self) -> None:
        """Interrupt current audio playback (barge-in)."""
        pass
    
    @abstractmethod
    def is_voice_detected(self, audio: np.ndarray) -> bool:
        """
        Check if voice activity is detected in the audio.
        
        Args:
            audio: Audio data to analyze
            
        Returns:
            True if voice is detected, False otherwise
        """
        pass


class IOrchestrator(ABC):
    """Interface for the main orchestration loop."""
    
    @abstractmethod
    async def start(self) -> None:
        """Start the voice assistant orchestration loop."""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """Stop the voice assistant orchestration loop."""
        pass
    
    @abstractmethod
    def is_running(self) -> bool:
        """Check if the orchestrator is currently running."""
        pass

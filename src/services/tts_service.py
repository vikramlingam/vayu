"""
Vayu TTS Service
Wrapper for kokoro-onnx with LOCAL model loading.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import asyncio
import numpy as np
from typing import Generator
from kokoro_onnx import Kokoro
from colorama import Fore, Style

import config
from src.interfaces import ITTSService


class TTSService(ITTSService):
    """
    Text-to-Speech service using kokoro-onnx.
    Loads model from LOCAL paths.
    """
    
    def __init__(self, voice: str = None, speed: float = None):
        """
        Initialize the TTS service with local models.
        
        Args:
            voice: Voice to use (default from config)
            speed: Speech speed multiplier (default from config)
        """
        print(f"{Fore.CYAN}[TTS] Loading model from: {config.TTS_MODEL_PATH}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}[TTS] Loading voices from: {config.TTS_VOICES_PATH}{Style.RESET_ALL}")
        
        # Load kokoro with LOCAL model paths
        try:
            self.kokoro = Kokoro(
                model_path=str(config.TTS_MODEL_PATH),
                voices_path=str(config.TTS_VOICES_PATH)
            )
        except Exception as e:
            if "pickle" in str(e).lower() or "unpickling" in str(e).lower():
                print(f"{Fore.RED}[TTS] CRITICAL ERROR: Pickle error loading voices.{Style.RESET_ALL}")
                print(f"{Fore.YELLOW}Tip: Ensure 'voices.npz' exists and is a valid numpy archive.{Style.RESET_ALL}")
            raise e
        
        self.voice = voice or config.TTS_VOICE
        self.speed = speed or config.TTS_SPEED
        self._sample_rate = 24000  # Kokoro outputs 24kHz audio
        self.is_speaking = False  # Flag to track if TTS is currently generating/speaking
        
        print(f"{Fore.GREEN}[TTS] Model loaded successfully ✓{Style.RESET_ALL}")
        print(f"{Fore.CYAN}[TTS] Using voice: {self.voice}{Style.RESET_ALL}")
    
    @staticmethod
    def get_available_voices() -> list[str]:
        """
        Get list of available voices from the voices.npz file.
        
        Returns:
            List of voice names
        """
        try:
            if not config.TTS_VOICES_PATH.exists():
                return []
                
            # Load the npz file to get keys
            with np.load(config.TTS_VOICES_PATH) as data:
                return sorted(list(data.keys()))
        except Exception as e:
            print(f"{Fore.RED}[TTS] Error listing voices: {e}{Style.RESET_ALL}")
            return []
    
    def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesize text to audio.
        
        Args:
            text: Text to synthesize
            
        Returns:
            Audio data as numpy array (float32)
        """
        if not text or not text.strip():
            return np.array([], dtype=np.float32)
        
        # Generate audio using kokoro
        self.is_speaking = True
        try:
            samples, sample_rate = self.kokoro.create(
                text=text,
                voice=self.voice,
                speed=self.speed
            )
        finally:
            self.is_speaking = False
        
        self._sample_rate = sample_rate
        
        # Ensure float32 output
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        
        return samples
    
    async def synthesize_stream(self, text: str) -> Generator[np.ndarray, None, None]:
        """
        Synthesize text to streaming audio.
        
        Kokoro supports streaming synthesis.
        
        Args:
            text: Text to synthesize
            
        Yields:
            Audio chunks as numpy arrays
        """
        if not text or not text.strip():
            return
        
        # Use kokoro's streaming capability
        self.is_speaking = True
        try:
            # Note: Kokoro's create_stream might be a synchronous generator depending on version,
            # but usually for ONNX/async implementations it might be async. 
            # Based on standard usage of kokoro-onnx 0.3+, create_stream is synchronous 
            # but we want to run it without blocking the event loop if possible.
            # However, since we need to iterate it, we'll assume standard iteration for now
            # but wrapped in a way that doesn't block.
            
            # Actually, standard kokoro-onnx create_stream is a normal generator.
            # getting chunks might block slightly during inference.
            # To be safe for our async pipeline, we yield chunks.
            
            stream = self.kokoro.create_stream(
                text=text,
                voice=self.voice,
                speed=self.speed
            )
            
            async for samples, sample_rate in stream:
                self._sample_rate = sample_rate
                
                if samples.dtype != np.float32:
                    samples = samples.astype(np.float32)
                
                # Yield to event loop to allow other tasks to run
                await asyncio.sleep(0)
                yield samples
                
        finally:
            self.is_speaking = False
    
    def get_sample_rate(self) -> int:
        """Get the sample rate of synthesized audio."""
        return self._sample_rate
    
    def set_voice(self, voice: str) -> None:
        """
        Change the voice.
        
        Args:
            voice: Voice identifier to use
        """
        self.voice = voice
        print(f"{Fore.CYAN}[TTS] Voice changed to: {self.voice}{Style.RESET_ALL}")
    
    def set_speed(self, speed: float) -> None:
        """
        Change the speech speed.
        
        Args:
            speed: Speed multiplier (1.0 = normal)
        """
        self.speed = speed
        print(f"{Fore.CYAN}[TTS] Speed changed to: {self.speed}{Style.RESET_ALL}")


if __name__ == "__main__":
    # Test the TTS service
    print("Testing TTS Service...")
    tts = TTSService()
    
    # Synthesize a test phrase
    audio = tts.synthesize("Hello! I am EchoLoop, your voice assistant.")
    print(f"Generated audio: {len(audio)} samples at {tts.get_sample_rate()}Hz")
    print(f"Duration: {len(audio) / tts.get_sample_rate():.2f} seconds")

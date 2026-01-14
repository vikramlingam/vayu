"""
Vayu Audio Manager
Handles microphone input, speaker output, VAD, and barge-in logic.
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import threading
import queue
import time
import numpy as np
import sounddevice as sd
import torch
from typing import Generator, Optional, Callable
from colorama import Fore, Style

import config
from src.interfaces import IAudioManager


class AudioManager(IAudioManager):
    """
    Audio I/O manager with VAD and barge-in support.
    
    Features:
    - Microphone recording with buffering
    - Speaker playback with streaming support
    - Voice Activity Detection (VAD) using Silero
    - Soft Interrupt (barge-in) via threading.Event
    """
    
    def __init__(self, on_voice_start: Optional[Callable] = None, on_voice_end: Optional[Callable] = None):
        """
        Initialize the audio manager.
        
        Args:
            on_voice_start: Callback when voice activity starts
            on_voice_end: Callback when voice activity ends
        """
        self.sample_rate = config.SAMPLE_RATE
        self.chunk_size = config.CHUNK_SIZE
        self.channels = config.CHANNELS
        self.vad_threshold = config.VAD_THRESHOLD
        
        # Recording state
        self._recording = False
        self._audio_queue: queue.Queue = queue.Queue()
        self._input_stream: Optional[sd.InputStream] = None
        
        # Playback state
        self._playing = False
        self._output_stream: Optional[sd.OutputStream] = None
        
        # BARGE-IN: Interrupt event for soft interruption
        self.interrupt_event = threading.Event()
        
        # VAD state
        self._vad_model = None
        self._vad_iterator = None
        self._voice_active = False
        self._silence_counter = 0
        self._pause_start_time = None  # Track when pause started for smart detection
        
        # Callbacks
        self.on_voice_start = on_voice_start
        self.on_voice_end = on_voice_end
        
        # Cooldown state: Track when playback ended to avoid residual echo
        self._playback_ended_at = 0.0
        
        # SMART BARGE-IN: Pending audio buffer (captures audio during playback)
        self._pending_audio_buffer: list = []
        
        # VOICE FINGERPRINT: Store TTS audio for echo detection comparison
        self._current_tts_audio: Optional[np.ndarray] = None
        
        # Initialize VAD
        self._init_vad()
        
        print(f"{Fore.GREEN}[Audio] Manager initialized ✓{Style.RESET_ALL}")
    
    def _init_vad(self):
        """Initialize Silero VAD model."""
        try:
            print(f"{Fore.CYAN}[Audio] Loading Silero VAD model...{Style.RESET_ALL}")
            
            # Load Silero VAD from torch.hub
            self._vad_model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            
            (self._get_speech_timestamps,
             self._save_audio,
             self._read_audio,
             self._VADIterator,
             self._collect_chunks) = utils
            
            # Create VAD iterator for streaming
            self._vad_iterator = self._VADIterator(
                self._vad_model,
                threshold=self.vad_threshold,
                sampling_rate=self.sample_rate,
                min_silence_duration_ms=config.VAD_MIN_SILENCE_MS,
                speech_pad_ms=config.VAD_SPEECH_PAD_MS
            )
            
            print(f"{Fore.GREEN}[Audio] VAD model loaded ✓{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.YELLOW}[Audio] Warning: Could not load Silero VAD: {e}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}[Audio] Falling back to simple energy-based VAD{Style.RESET_ALL}")
            self._vad_model = None
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio input stream."""
        if status:
            print(f"{Fore.YELLOW}[Audio] Input status: {status}{Style.RESET_ALL}")
        
        # Copy the audio data to avoid buffer issues
        audio_chunk = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
        self._audio_queue.put(audio_chunk)
    
    def start_recording(self) -> None:
        """Start recording audio from the microphone."""
        if self._recording:
            return
        
        self._recording = True
        self._audio_queue = queue.Queue()
        
        # Clear interrupt event when starting new recording
        self.interrupt_event.clear()
        
        self._input_stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype='float32',
            blocksize=self.chunk_size,
            callback=self._audio_callback
        )
        self._input_stream.start()
        
        print(f"{Fore.GREEN}[Audio] Recording started ✓{Style.RESET_ALL}")
    
    def stop_recording(self) -> None:
        """Stop recording audio."""
        if not self._recording:
            return
        
        self._recording = False
        
        if self._input_stream:
            self._input_stream.stop()
            self._input_stream.close()
            self._input_stream = None
        
        print(f"{Fore.CYAN}[Audio] Recording stopped{Style.RESET_ALL}")
    
    def get_audio_chunk(self) -> Optional[np.ndarray]:
        """
        Get the next audio chunk from the recording buffer.
        
        Returns:
            Audio chunk as numpy array, or None if no audio available
        """
        try:
            return self._audio_queue.get(timeout=0.1)
        except queue.Empty:
            return None
    
    def get_audio_stream(self) -> Generator[np.ndarray, None, None]:
        """
        Generator that yields audio chunks as they're recorded.
        
        Yields:
            Audio chunks as numpy arrays
        """
        while self._recording or not self._audio_queue.empty():
            chunk = self.get_audio_chunk()
            if chunk is not None:
                yield chunk
    
    def play_audio(self, audio: np.ndarray, sample_rate: int) -> None:
        """
        Play audio through the speakers.
        
        Args:
            audio: Audio data as numpy array
            sample_rate: Sample rate of the audio
        """
        if audio is None or len(audio) == 0:
            return
        
        # Clear interrupt before playing
        self.interrupt_event.clear()
        
        try:
            sd.play(audio, sample_rate)
            sd.wait()
        except Exception as e:
            print(f"{Fore.RED}[Audio] Playback error: {e}{Style.RESET_ALL}")
    
    def play_audio_stream(self, audio_stream: Generator[np.ndarray, None, None], sample_rate: int) -> None:
        """
        Play streaming audio through the speakers with barge-in support.
        
        BARGE-IN LOGIC: Checks interrupt_event before processing each chunk.
        
        Args:
            audio_stream: Generator yielding audio chunks
            sample_rate: Sample rate of the audio
        """
        self._playing = True
        
        # Clear interrupt before starting playback
        self.interrupt_event.clear()
        
        try:
            with sd.OutputStream(samplerate=sample_rate, channels=1, dtype='float32') as stream:
                for chunk in audio_stream:
                    # SOFT INTERRUPT: Check if interrupted before each chunk
                    if self.interrupt_event.is_set():
                        print(f"{Fore.YELLOW}[Audio] Playback interrupted (barge-in){Style.RESET_ALL}")
                        return
                    
                    if chunk is not None and len(chunk) > 0:
                        # Ensure the chunk is the right shape
                        if chunk.ndim == 1:
                            chunk = chunk.reshape(-1, 1)
                        stream.write(chunk)
        except Exception as e:
            print(f"{Fore.RED}[Audio] Streaming playback error: {e}{Style.RESET_ALL}")
        finally:
            self._playing = False
            self._playback_ended_at = time.time()  # Start cooldown timer
    
    def interrupt(self) -> None:
        """Interrupt current audio playback (barge-in)."""
        print(f"{Fore.YELLOW}[Audio] Interrupt signal received{Style.RESET_ALL}")
        self.interrupt_event.set()
        
        # Also stop any direct playback
        try:
            sd.stop()
        except Exception:
            pass
    
    def is_playing(self) -> bool:
        """Check if audio is currently playing."""
        return self._playing
    
    def is_voice_detected(self, audio: np.ndarray) -> bool:
        """
        Check if voice activity is detected in the audio.
        
        Args:
            audio: Audio data to analyze
            
        Returns:
            True if voice is detected, False otherwise
        """
        if audio is None or len(audio) == 0:
            return False
            
        # SMART BARGE-IN: During playback, capture audio to pending buffer
        # This allows us to process user's interrupted speech after AI finishes
        if self.is_playing():
            # Always capture audio to pending buffer during playback
            self._pending_audio_buffer.append(audio.copy())
            
            # Check if barge-in is enabled and volume is loud enough
            if config.ENABLE_BARGE_IN:
                volume = np.linalg.norm(audio) / np.sqrt(len(audio))
                if volume > config.BARGE_IN_VOLUME_THRESHOLD:
                    print(f"{Fore.LIGHTRED_EX}[Vayu] 🛑 Barge-in detected! (Vol: {volume:.4f}){Style.RESET_ALL}")
                    self.interrupt_event.set()
                    return True
            
            # Don't trigger VAD during playback (prevents echo)
            return False
        
        # COOLDOWN CHECK: After playback ends, wait before re-enabling VAD
        # This prevents residual echo from triggering false positives.
        cooldown_seconds = config.PLAYBACK_COOLDOWN_MS / 1000.0
        time_since_playback = time.time() - self._playback_ended_at
        if time_since_playback < cooldown_seconds:
            # Still capture audio during cooldown (might be user speaking)
            self._pending_audio_buffer.append(audio.copy())
            return False
        
        # NORMAL MODE: Use Silero VAD as usual
        if self._vad_model is not None:
            return self._silero_vad_detect(audio)
        else:
            # Fallback to simple energy-based detection
            return self._energy_vad_detect(audio)
    
    def _silero_vad_detect(self, audio: np.ndarray) -> bool:
        """Use Silero VAD for voice detection."""
        try:
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio).float()
            
            # Ensure correct length for Silero (at least 512 samples)
            if len(audio_tensor) < 512:
                return False
            
            # Get speech probability
            speech_prob = self._vad_model(audio_tensor, self.sample_rate).item()
            
            return speech_prob > self.vad_threshold
            
        except Exception as e:
            print(f"{Fore.YELLOW}[Audio] VAD error: {e}{Style.RESET_ALL}")
            return self._energy_vad_detect(audio)
    
    def _energy_vad_detect(self, audio: np.ndarray) -> bool:
        """Simple energy-based VAD fallback."""
        energy = np.sqrt(np.mean(audio ** 2))
        threshold = 0.01  # Adjustable threshold
        return energy > threshold
    
    def process_vad_stream(self, audio_chunk: np.ndarray) -> dict:
        """
        Process a single audio chunk for VAD with SMART PAUSE DETECTION.
        
        Intelligence: Learns from your speaking pattern in real-time.
        - Starts FAST (1.5s) for quick commands
        - If you pause and resume, adds bonus time (0.8s per resume)
        - Caps at 3.5s max to stay responsive
        
        Args:
            audio_chunk: Audio data chunk
            
        Returns:
            Dict with 'voice_active' and 'speech_started'/'speech_ended' flags
        """
        result = {
            'voice_active': False,
            'speech_started': False,
            'speech_ended': False
        }
        
        is_voice = self.is_voice_detected(audio_chunk)
        result['voice_active'] = is_voice
        
        # Track transitions with SMART pattern detection
        if is_voice and not self._voice_active:
            # Speech started
            self._voice_active = True
            self._silence_counter = 0
            self._speech_start_time = time.time()
            self._pause_start_time = None
            
            # Initialize resume counter for this utterance
            if not hasattr(self, '_resume_count'):
                self._resume_count = 0
                
            result['speech_started'] = True
            if self.on_voice_start:
                self.on_voice_start()
                
        elif is_voice and self._voice_active:
            # User RESUMED speaking after a pause - they're mid-thought!
            if hasattr(self, '_pause_start_time') and self._pause_start_time is not None:
                pause_duration = time.time() - self._pause_start_time
                if pause_duration > 0.3:  # Significant pause that was recovered from
                    self._resume_count = getattr(self, '_resume_count', 0) + 1
                    bonus_s = (self._resume_count * config.VAD_RESUME_BONUS_MS) / 1000.0
                    print(f"{Fore.LIGHTBLACK_EX}[VAD] Resumed after {pause_duration:.1f}s (bonus: +{bonus_s:.1f}s){Style.RESET_ALL}")
            self._pause_start_time = None
            self._silence_counter = 0
            
        elif not is_voice and self._voice_active:
            # Silence detected
            
            if self._pause_start_time is None:
                self._pause_start_time = time.time()
            
            pause_duration = time.time() - self._pause_start_time
            
            # SMART THRESHOLD: Base + Resume Bonus (capped)
            base_s = config.VAD_MIN_SILENCE_MS / 1000.0
            resume_bonus_s = min(
                getattr(self, '_resume_count', 0) * config.VAD_RESUME_BONUS_MS / 1000.0,
                config.VAD_MAX_RESUME_BONUS_MS / 1000.0
            )
            required_pause = min(base_s + resume_bonus_s, config.VAD_MAX_SILENCE_MS / 1000.0)
            
            # Visual feedback every second
            self._silence_counter += 1
            if self._silence_counter % int(self.sample_rate / self.chunk_size) == 0:
                remaining = required_pause - pause_duration
                if remaining > 0:
                    print(f"{Fore.LIGHTBLACK_EX}[VAD] Pause: {pause_duration:.1f}s / {required_pause:.1f}s...{Style.RESET_ALL}")
            
            if pause_duration >= required_pause:
                print(f"{Fore.CYAN}[VAD] Done: {pause_duration:.1f}s pause{Style.RESET_ALL}")
                self._voice_active = False
                self._speech_start_time = None
                self._pause_start_time = None
                self._resume_count = 0  # Reset for next utterance
                result['speech_ended'] = True
                if self.on_voice_end:
                    self.on_voice_end()
        
        return result
    
    def reset_vad_state(self):
        """Reset VAD state."""
        self._voice_active = False
        self._silence_counter = 0
        self._pause_start_time = None
        self._resume_count = 0
        if self._vad_iterator:
            self._vad_iterator.reset_states()
    
    def get_pending_audio(self) -> Optional[np.ndarray]:
        """
        Get audio captured during playback (pending buffer).
        
        Returns:
            Concatenated audio from pending buffer, or None if empty
        """
        if not self._pending_audio_buffer:
            return None
        
        # Concatenate all pending chunks
        full_audio = np.concatenate(self._pending_audio_buffer)
        return full_audio
    
    def clear_pending_audio(self) -> None:
        """Clear the pending audio buffer."""
        self._pending_audio_buffer = []
    
    def has_pending_audio(self) -> bool:
        """Check if there's audio in the pending buffer."""
        return len(self._pending_audio_buffer) > 0
    
    def set_tts_audio(self, audio: np.ndarray) -> None:
        """
        Store the TTS audio being played for echo detection.
        
        Args:
            audio: The TTS audio data
        """
        self._current_tts_audio = audio.copy() if audio is not None else None
    
    def get_tts_audio(self) -> Optional[np.ndarray]:
        """Get the stored TTS audio for fingerprint comparison."""
        return self._current_tts_audio
    
    def clear_tts_audio(self) -> None:
        """Clear the stored TTS audio."""
        self._current_tts_audio = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_recording()
        self.interrupt()


if __name__ == "__main__":
    import time
    
    print("Testing Audio Manager...")
    
    with AudioManager() as audio_manager:
        print("\nRecording for 3 seconds...")
        audio_manager.start_recording()
        
        chunks = []
        start_time = time.time()
        
        while time.time() - start_time < 3:
            chunk = audio_manager.get_audio_chunk()
            if chunk is not None:
                chunks.append(chunk)
                vad_result = audio_manager.is_voice_detected(chunk)
                if vad_result:
                    print("🎤 Voice detected!")
        
        audio_manager.stop_recording()
        
        if chunks:
            full_audio = np.concatenate(chunks)
            print(f"\nRecorded {len(full_audio)} samples ({len(full_audio)/16000:.2f} seconds)")
            print("Playing back...")
            audio_manager.play_audio(full_audio, 16000)
        
        print("\nTest complete!")

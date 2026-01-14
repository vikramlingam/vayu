"""
Vayu Orchestrator
Async Actor Model implementation for the voice AI pipeline.

Pipeline Architecture:
    Loop 1: Audio Input -> VAD -> STT -> llm_queue
    Loop 2: llm_queue -> LLM (Stream) -> tts_queue  
    Loop 3: tts_queue -> TTS -> Audio Output
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import threading
import numpy as np
from typing import Optional
from colorama import Fore, Style
import queue as sync_queue

import config
from src.interfaces import IOrchestrator
from src.services.audio_manager import AudioManager
from src.services.stt_service import STTService
from src.services.tts_service import TTSService
from src.services.llm_service import LLMService
from src.utils.voice_fingerprint import is_echo
from src.utils.sound_effects import get_thinking_chime
import sounddevice as sd


class Orchestrator(IOrchestrator):
    """
    Async orchestrator implementing the Actor Model for voice AI.
    
    Three concurrent loops handle the voice interaction pipeline:
    1. Input Loop: Audio -> VAD -> STT -> LLM Queue
    2. LLM Loop: LLM Queue -> Streaming LLM -> TTS Queue
    3. Output Loop: TTS Queue -> TTS -> Audio Output
    """
    
    def __init__(self):
        """Initialize the orchestrator with all services."""
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}  Vayu Voice AI - Initializing{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        
        # Initialize services
        self.audio_manager = AudioManager(
            on_voice_start=self._on_voice_start,
            on_voice_end=self._on_voice_end
        )
        self.stt_service = STTService()
        self.tts_service = TTSService()
        self.llm_service = LLMService()
        
        # Async queues for actor communication
        self.llm_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=config.QUEUE_MAXSIZE)
        self.tts_queue: asyncio.Queue[str] = asyncio.Queue(maxsize=config.QUEUE_MAXSIZE)
        
        # State
        self._running = False
        self._tasks: list[asyncio.Task] = []
        
        # Audio buffer for VAD-based collection
        self._audio_buffer: list[np.ndarray] = []
        self._is_speaking = False  # User is speaking
        self._is_responding = False  # AI is responding
        
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}  Vayu Ready!{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    
    def _on_voice_start(self):
        """Callback when voice activity starts."""
        # Don't log voice start during playback (it's just echo detection)
        if self.audio_manager.is_playing():
            return
        
        print(f"{Fore.YELLOW}[Vayu] Voice activity started - listening...{Style.RESET_ALL}")
        
        # BARGE-IN: Only trigger if enabled AND audio is actually playing
        # (not just during LLM generation before TTS starts)
        if config.ENABLE_BARGE_IN and self._is_responding and self.audio_manager.is_playing():
            print(f"{Fore.MAGENTA}[Vayu] BARGE-IN detected!{Style.RESET_ALL}")
            self.audio_manager.interrupt()
            self._is_responding = False
    
    def _on_voice_end(self):
        """Callback when voice activity ends."""
        # Don't log voice end during playback
        if self.audio_manager.is_playing():
            return
        
        print(f"{Fore.YELLOW}[Vayu] Voice activity ended - processing...{Style.RESET_ALL}")
    
    async def start(self) -> None:
        """Start the voice assistant orchestration loops."""
        if self._running:
            print(f"{Fore.YELLOW}[Vayu] Already running{Style.RESET_ALL}")
            return
        
        self._running = True
        
        print(f"\n{Fore.GREEN}🎤 Vayu is now listening...{Style.RESET_ALL}")
        print(f"{Fore.CYAN}   Speak naturally. Press Ctrl+C to stop.{Style.RESET_ALL}\n")
        
        # Start audio recording
        self.audio_manager.start_recording()
        
        # Create and run the three actor loops
        self._tasks = [
            asyncio.create_task(self._input_loop(), name="input_loop"),
            asyncio.create_task(self._llm_loop(), name="llm_loop"),
            asyncio.create_task(self._output_loop(), name="output_loop"),
        ]
        
        try:
            # Wait for all tasks (they run forever until stopped)
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            print(f"\n{Fore.CYAN}[Vayu] Shutting down...{Style.RESET_ALL}")
    
    async def stop(self) -> None:
        """Stop the voice assistant orchestration loops."""
        if not self._running:
            return
        
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for cancellation
        await asyncio.gather(*self._tasks, return_exceptions=True)
        
        # Stop audio
        self.audio_manager.stop_recording()
        self.audio_manager.interrupt()
        
        print(f"{Fore.GREEN}[Vayu] Stopped{Style.RESET_ALL}")
    
    def is_running(self) -> bool:
        """Check if the orchestrator is currently running."""
        return self._running
    
    async def _play_thinking_chime(self):
        """Play a short chime to indicate AI is about to respond."""
        try:
            chime = get_thinking_chime(sample_rate=24000)
            # Play in thread to not block
            await asyncio.to_thread(sd.play, chime, 24000)
            await asyncio.to_thread(sd.wait)
        except Exception as e:
            # Don't let chime errors break the flow
            pass
    
    async def _check_pending_speech(self):
        """
        Check if user spoke during playback and process that speech.
        
        Uses VOICE FINGERPRINT COMPARISON - a novel technique to distinguish
        between user speech and AI echo without hardware AEC.
        
        The key insight: We know exactly what the AI is playing (TTS output).
        If the recorded audio has similar spectral fingerprint, it's echo.
        If it's different, it's likely the user speaking.
        """
        # Wait for cooldown to settle
        await asyncio.sleep(config.POST_PLAYBACK_LISTEN_MS / 1000.0)
        
        # Check if there's pending audio
        if not self.audio_manager.has_pending_audio():
            self.audio_manager.clear_tts_audio()
            return
        
        pending_audio = self.audio_manager.get_pending_audio()
        tts_audio = self.audio_manager.get_tts_audio()
        
        # Clear buffers
        self.audio_manager.clear_pending_audio()
        self.audio_manager.clear_tts_audio()
        
        if pending_audio is None or len(pending_audio) == 0:
            return
        
        # Check basic volume (filter out silence)
        volume = np.linalg.norm(pending_audio) / np.sqrt(len(pending_audio))
        if volume < 0.01:  # Too quiet to be meaningful
            return
        
        # VOICE FINGERPRINT COMPARISON
        # Compare pending audio against TTS audio to detect echo
        if tts_audio is not None and len(tts_audio) > 0:
            is_echo_detected, similarity = is_echo(
                tts_audio, 
                pending_audio, 
                sample_rate=config.SAMPLE_RATE,
                threshold=0.5  # Tunable - higher = more strict
            )
            
            if is_echo_detected:
                # print(f"{Fore.LIGHTBLACK_EX}[Vayu] Echo detected (similarity: {similarity:.2f}) - ignoring{Style.RESET_ALL}")
                return
            
            # print(f"{Fore.MAGENTA}[Vayu] User speech detected! (similarity: {similarity:.2f}, vol: {volume:.4f}){Style.RESET_ALL}")
        else:
            # No TTS audio to compare - this might be valid user speech
            print(f"{Fore.MAGENTA}[Vayu] Processing speech (vol: {volume:.4f}){Style.RESET_ALL}")
        
        # Check duration (must be significant)
        duration = len(pending_audio) / config.SAMPLE_RATE
        if duration < config.MIN_SPEECH_DURATION:
            print(f"{Fore.LIGHTBLACK_EX}[Vayu] Too short ({duration:.2f}s) - ignoring{Style.RESET_ALL}")
            return
        
        # Transcribe the pending audio
        transcription = await asyncio.to_thread(
            self.stt_service.transcribe, pending_audio
        )
        
        if transcription and transcription.strip():
            print(f"\n{Fore.GREEN}👤 You (interrupted): {transcription}{Style.RESET_ALL}")
            
            # Reset VAD state before queueing
            self.audio_manager.reset_vad_state()
            
            # Queue for LLM processing
            await self.llm_queue.put(transcription)
    
    async def _input_loop(self):
        """
        Input Actor Loop: Audio -> VAD -> STT -> LLM Queue
        
        Collects audio while voice is active, then transcribes and queues for LLM.
        """
        while self._running:
            try:
                # Get audio chunk (non-blocking with small timeout)
                audio_chunk = await asyncio.to_thread(self.audio_manager.get_audio_chunk)
                
                if audio_chunk is None:
                    await asyncio.sleep(0.01)
                    continue
                
                # Process VAD
                vad_result = self.audio_manager.process_vad_stream(audio_chunk)
                
                if vad_result['voice_active']:
                    self._is_speaking = True
                    self._audio_buffer.append(audio_chunk)
                
                elif vad_result['speech_ended'] and self._audio_buffer:
                    self._is_speaking = False
                    
                    # Concatenate collected audio
                    full_audio = np.concatenate(self._audio_buffer)
                    self._audio_buffer = []
                    
                    # Anti-Clap: Check duration
                    duration = len(full_audio) / config.SAMPLE_RATE
                    if duration < config.MIN_SPEECH_DURATION:
                        print(f"{Fore.LIGHTBLACK_EX}[Vayu] Ignored short noise ({duration:.2f}s){Style.RESET_ALL}")
                        continue  # Skip STT, continue listening
                    
                    # Transcribe in a thread to not block
                    transcription = await asyncio.to_thread(
                        self.stt_service.transcribe, full_audio
                    )
                    
                    if transcription and transcription.strip():
                        print(f"\n{Fore.GREEN}👤 You: {transcription}{Style.RESET_ALL}")
                        
                        # Reset VAD state before queueing to prevent residual triggers
                        self.audio_manager.reset_vad_state()
                        
                        # Queue for LLM processing
                        await self.llm_queue.put(transcription)
                    else:
                        print(f"{Fore.YELLOW}[Vayu] No speech detected{Style.RESET_ALL}")
                
            except Exception as e:
                print(f"{Fore.RED}[Input Loop] Error: {e}{Style.RESET_ALL}")
                await asyncio.sleep(0.1)
    
    async def _llm_loop(self):
        """
        LLM Actor Loop: LLM Queue -> Streaming LLM -> TTS Queue
        
        Gets transcriptions from the LLM queue, streams responses to TTS queue.
        """
        while self._running:
            try:
                # Wait for transcription from input loop
                transcription = await self.llm_queue.get()
                
                # CRITICAL: Clear any stale interrupt flag from previous barge-in
                self.audio_manager.interrupt_event.clear()
                
                # Play chime to indicate AI is about to respond
                await self._play_thinking_chime()
                
                print(f"{Fore.CYAN}🤖 Vayu: {Style.RESET_ALL}", end="", flush=True)
                
                self._is_responding = True
                
                # Stream LLM response and queue chunks for TTS
                async for text_chunk in self.llm_service.generate_stream(transcription):
                    # Check if we were interrupted (barge-in)
                    if self.audio_manager.interrupt_event.is_set():
                        print(f" {Fore.YELLOW}[interrupted]{Style.RESET_ALL}")
                        self._is_responding = False
                        # Clear the TTS queue
                        while not self.tts_queue.empty():
                            try:
                                self.tts_queue.get_nowait()
                            except asyncio.QueueEmpty:
                                break
                        break
                    
                    if text_chunk and text_chunk.strip():
                        print(f"{text_chunk} ", end="", flush=True)
                        await self.tts_queue.put(text_chunk)
                
                # Signal end of response
                if self._is_responding:
                    print()  # New line after response
                    await self.tts_queue.put(None)  # End marker
                
                self._is_responding = False
                
            except Exception as e:
                print(f"{Fore.RED}[LLM Loop] Error: {e}{Style.RESET_ALL}")
                self._is_responding = False
                await asyncio.sleep(0.1)
    
    async def _output_loop(self):
        """
        Output Actor Loop: TTS Queue -> TTS -> Audio Output
        
        Gets text chunks from TTS queue, synthesizes and plays audio.
        Uses streaming with a CONTINUOUS playback session to avoid audio gaps.
        """
        while self._running:
            try:
                # 1. Wait for START of response
                text_chunk = await self.tts_queue.get()
                
                if text_chunk is None:
                    continue
                
                # Check for interrupt
                if self.audio_manager.interrupt_event.is_set():
                    continue
                
                # 2. Initialize Continuous Playback Session
                self.audio_manager.clear_pending_audio()
                playback_queue = sync_queue.Queue()
                session_active = True
                playback_thread = None
                sample_rate = None

                # Helper to push audio to queue
                async def process_and_queue(text):
                    nonlocal sample_rate, playback_thread
                    
                    stream_gen = self.tts_service.synthesize_stream(text)
                    
                    try:
                        async for chunk in stream_gen:
                            if self.audio_manager.interrupt_event.is_set():
                                return False

                            # If this is the very first chunk of the session, start the thread
                            if playback_thread is None:
                                sample_rate = self.tts_service.get_sample_rate()
                                
                                # Consumes queue until None
                                def queue_audio_generator():
                                    while True:
                                        try:
                                            q_chunk = playback_queue.get(timeout=0.1)
                                            if q_chunk is None: break
                                            yield q_chunk
                                        except sync_queue.Empty:
                                            if not session_active: break
                                            continue
                                
                                playback_thread = threading.Thread(
                                    target=self.audio_manager.play_audio_stream,
                                    args=(queue_audio_generator(), sample_rate)
                                )
                                playback_thread.start()

                            playback_queue.put(chunk)
                            self._update_fingerprint_buffer(chunk, is_new=(playback_queue.qsize() == 1))
                            
                        return True
                    except Exception as e:
                        print(f"{Fore.RED}[Output Loop] Synthesis Error: {e}{Style.RESET_ALL}")
                        return False

                # Process the initial text chunk
                success = await process_and_queue(text_chunk)
                if not success: continue

                # 3. Process SUBSEQUENT chunks until end of response
                while True:
                    # Check interrupt
                    if self.audio_manager.interrupt_event.is_set():
                        session_active = False
                        break
                        
                    # Peek or wait for next text
                    text = await self.tts_queue.get()
                    
                    if text is None: # End of response
                        session_active = False
                        break
                        
                    # Process text
                    success = await process_and_queue(text)
                    if not success: # Interrupted or error
                        session_active = False
                        break
                
                # 4. Cleanup Session
                playback_queue.put(None) # Signal thread to stop
                if playback_thread:
                    await asyncio.to_thread(playback_thread.join)
                
                await self._check_pending_speech()
                
            except Exception as e:
                print(f"{Fore.RED}[Output Loop] Error: {e}{Style.RESET_ALL}")
                await asyncio.sleep(0.1)

    def _update_fingerprint_buffer(self, audio_chunk, is_new=False):
        """Helper to accumulate TTS audio for echo detection."""
        sample_rate = self.tts_service.get_sample_rate()
        
        if sample_rate != config.SAMPLE_RATE:
            ratio = sample_rate // config.SAMPLE_RATE
            resampled = audio_chunk[::ratio]
        else:
            resampled = audio_chunk
            
        if is_new:
            self.audio_manager.set_tts_audio(resampled)
        else:
            current = self.audio_manager.get_tts_audio()
            if current is None:
                self.audio_manager.set_tts_audio(resampled)
            else:
                self.audio_manager.set_tts_audio(np.concatenate([current, resampled]))


async def main():
    """Main entry point for the orchestrator."""
    orchestrator = Orchestrator()
    
    try:
        await orchestrator.start()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Keyboard interrupt received{Style.RESET_ALL}")
    finally:
        await orchestrator.stop()


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Vayu Voice AI - Main Entry Point

Enterprise-grade modular voice assistant with:
- Local STT (faster-whisper)
- Local TTS (kokoro-onnx)
- Cloud LLM (OpenRouter)
- Barge-in support
- Async Actor Model architecture

Usage:
    python main.py
"""

import sys
import asyncio
from pathlib import Path

# Ensure we can import from the project
sys.path.insert(0, str(Path(__file__).parent))

from colorama import init, Fore, Style

# Initialize colorama for cross-platform colored output
init(autoreset=True)

import config
from src.orchestrator import Orchestrator


def print_banner():
    """Print the Vayu startup banner."""
    banner = f"""
{Fore.CYAN}╔════════════════════════════════════════════════╗
║                                                ║
║                                                ║
║   ██╗   ██╗ █████╗ ██╗   ██╗██╗   ██╗          ║
║   ██║   ██║██╔══██╗╚██╗ ██╔╝██║   ██║          ║
║   ██║   ██║███████║ ╚████╔╝ ██║   ██║          ║
║   ╚██╗ ██╔╝██╔══██║  ╚██╔╝  ██║   ██║          ║
║    ╚████╔╝ ██║  ██║   ██║   ╚██████╔╝          ║
║     ╚═══╝  ╚═╝  ╚═╝   ╚═╝    ╚═════╝           ║
║                                                ║
║                                                ║
║      ⚡ Realtime Voice AI Agent ⚡             ║
║                                                ║
╚════════════════════════════════════════════════╝{Style.RESET_ALL}
"""
    print(banner)


def print_config_info():
    """Print configuration information."""
    print(f"\n{Fore.CYAN}📁 Configuration:{Style.RESET_ALL}")
    print(f"   Base Directory:  {config.BASE_DIR}")
    print(f"   Models Directory: {config.MODELS_DIR}")
    print(f"   STT Model:       {config.STT_MODEL_PATH.name}")
    print(f"   TTS Model:       {config.TTS_MODEL_PATH.name}")
    print(f"   LLM Model:       {config.LLM_MODEL}")
    print(f"   Sample Rate:     {config.SAMPLE_RATE} Hz")
    print(f"   VAD Threshold:   {config.VAD_THRESHOLD}")


def select_voice():
    """Interactive voice selection CLI."""
    try:
        from src.services.tts_service import TTSService
        voices = TTSService.get_available_voices()
        
        # Filter out unwanted voices
        excluded_voices = {"am_adam", "bf_isabella"}
        voices = [v for v in voices if v not in excluded_voices]
        
        if not voices:
            print(f"{Fore.YELLOW}Warning: No voices found in {config.TTS_VOICES_PATH}{Style.RESET_ALL}")
            return
            
        print(f"\n{Fore.CYAN}🎤 Select a Voice:{Style.RESET_ALL}")
        print(f"{'─'*30}")
        
        # Filter for English voices (usually start with af_ or am_)
        # But show all just in case
        for i, voice in enumerate(voices, 1):
            # Highlight current default
            marker = "*" if voice == config.TTS_VOICE else " "
            print(f"{Fore.GREEN}{i:2d}.{Style.RESET_ALL} {voice} {marker}")
            
        print(f"{'─'*30}")
        print(f"Enter number (default: {config.TTS_VOICE}): ", end="")
        
        choice = input().strip()
        
        if not choice:
            return  # Keep default
            
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(voices):
                selected_voice = voices[idx]
                config.TTS_VOICE = selected_voice
                print(f"{Fore.GREEN}✓ Selected: {selected_voice}{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Invalid selection, keeping default.{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.RED}Invalid input, keeping default.{Style.RESET_ALL}")
            
    except Exception as e:
        print(f"{Fore.RED}Error selecting voice: {e}{Style.RESET_ALL}")
async def run_vayu():
    """Run the Vayu voice assistant."""
    # Validate configuration
    try:
        config.validate_config()
        print(f"\n{Fore.GREEN}✓ Configuration validated{Style.RESET_ALL}")
    except ValueError as e:
        print(f"\n{Fore.RED}✗ Configuration error:{Style.RESET_ALL}")
        print(f"  {e}")
        print(f"\n{Fore.YELLOW}Please ensure all models are in the 'models/' directory{Style.RESET_ALL}")
        return
    
    # Interactive Voice Selection
    select_voice()
    
    # Initialize and run orchestrator
    orchestrator = Orchestrator()
    
    print(f"\n{Fore.GREEN}{'─'*60}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Press Ctrl+C to stop{Style.RESET_ALL}")
    print(f"{Fore.GREEN}{'─'*60}{Style.RESET_ALL}\n")
    
    try:
        await orchestrator.start()
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}⚡ Shutting down Vayu...{Style.RESET_ALL}")
    finally:
        await orchestrator.stop()
        print(f"\n{Fore.CYAN}👋 Goodbye!{Style.RESET_ALL}\n")


def main():
    """Main entry point."""
    print_banner()
    print_config_info()
    
    # Check Python version
    if sys.version_info < (3, 10):
        print(f"\n{Fore.RED}Error: Python 3.10+ required (found {sys.version_info.major}.{sys.version_info.minor}){Style.RESET_ALL}")
        sys.exit(1)
    
    # Run the async main function
    try:
        asyncio.run(run_vayu())
    except Exception as e:
        print(f"\n{Fore.RED}Fatal error: {e}{Style.RESET_ALL}")
        sys.exit(1)


if __name__ == "__main__":
    main()

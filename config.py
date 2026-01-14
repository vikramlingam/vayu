"""
Vayu Configuration Module
Central configuration for paths, constants, and environment variables.
"""

from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# ============================================================================
# BASE PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

# ============================================================================
# MODEL PATHS (Local Models Only)
# ============================================================================
STT_MODEL_PATH = MODELS_DIR / "faster-whisper-tiny.en"
TTS_MODEL_PATH = MODELS_DIR / "kokoro-v0_19.onnx"
TTS_VOICES_PATH = MODELS_DIR / "voices.npz"

# ============================================================================
# AUDIO CONFIGURATION
# ============================================================================
SAMPLE_RATE = 16000  # Hz - Standard for speech recognition
CHUNK_SIZE = 512     # Audio chunk size for streaming
CHANNELS = 1         # Mono audio

# ============================================================================
# VAD (Voice Activity Detection) CONFIGURATION  
# Smart Pause Detection: Fast for commands, patient for explanations
# ============================================================================
VAD_THRESHOLD = 0.6         # Confidence threshold for voice detection
VAD_MIN_SILENCE_MS = 1500   # Base silence (1.5s) - fast for quick commands
VAD_MAX_SILENCE_MS = 3500   # Max silence (3.5s) - cap for long explanations
VAD_SPEECH_PAD_MS = 500     # Padding around detected speech
MIN_SPEECH_DURATION = 0.5   # Minimum duration (seconds) to consider valid speech

# SMART PAUSE DETECTION: Learns from your speaking pattern
VAD_RESUME_BONUS_MS = 800   # Add 0.8s to threshold each time user pauses and resumes
VAD_MAX_RESUME_BONUS_MS = 2000  # Cap: max 2s extra from pause-resume pattern

# ============================================================================
# BARGE-IN CONFIGURATION (Smart Interruption Handling)
# ============================================================================
ENABLE_BARGE_IN = False     # Disable immediate barge-in (prevents echo issues)
BARGE_IN_VOLUME_THRESHOLD = 0.50 # RMS threshold (only used if ENABLE_BARGE_IN is True)
PLAYBACK_COOLDOWN_MS = 500  # Milliseconds to wait after playback before VAD
POST_PLAYBACK_LISTEN_MS = 300  # Time to check for pending speech after playback

# ============================================================================
# STT CONFIGURATION
# ============================================================================
STT_DEVICE = "cpu"
STT_COMPUTE_TYPE = "int8"
STT_LANGUAGE = "en"
STT_BEAM_SIZE = 5

# ============================================================================
# TTS CONFIGURATION
# ============================================================================
TTS_VOICE = "af_bella"  # Default voice from kokoro
TTS_SPEED = 1.0  # Normal speech speed

# ============================================================================
# LLM CONFIGURATION (OpenRouter)
# ============================================================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
LLM_MODEL = "sao10k/l3-lunaris-8b"  # OpenRouter model ID
LLM_MAX_TOKENS = 1024
LLM_TEMPERATURE = 0.7

# System prompt for the voice assistant
SYSTEM_PROMPT = """You are Vayu, a helpful real-time voice assistant. 
Keep responses concise and conversational since they will be spoken aloud.
Respond naturally as if having a conversation.

IMPORTANT: Your responses will be read aloud by a text-to-speech system.
Do NOT use any markdown formatting, symbols, or special characters like:
- No asterisks (*), hashes (#), dashes (-), or bullet points
- No bold, italic, or code formatting
- No numbered lists with periods (use "first, second, third" instead)
- No special symbols or emojis
Write in plain, natural speech only."""

# ============================================================================
# QUEUE CONFIGURATION
# ============================================================================
QUEUE_MAXSIZE = 100  # Maximum items in async queues

# ============================================================================
# VALIDATION
# ============================================================================
def validate_config():
    """Validate that all required paths and configurations exist."""
    errors = []
    
    if not MODELS_DIR.exists():
        errors.append(f"Models directory not found: {MODELS_DIR}")
    
    if not STT_MODEL_PATH.exists():
        errors.append(f"STT model not found: {STT_MODEL_PATH}")
    
    if not TTS_MODEL_PATH.exists():
        errors.append(f"TTS model not found: {TTS_MODEL_PATH}")
    
    if not TTS_VOICES_PATH.exists():
        errors.append(f"TTS voices file not found: {TTS_VOICES_PATH}")
    
    if not OPENROUTER_API_KEY:
        errors.append("OPENROUTER_API_KEY not found in environment variables")
    
    if errors:
        raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
    
    return True


if __name__ == "__main__":
    print("Vayu Configuration")
    print("=" * 50)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"STT Model: {STT_MODEL_PATH}")
    print(f"TTS Model: {TTS_MODEL_PATH}")
    print(f"TTS Voices: {TTS_VOICES_PATH}")
    print(f"OpenRouter API Key: {'Set' if OPENROUTER_API_KEY else 'NOT SET'}")
    
    try:
        validate_config()
        print("\n✅ Configuration is valid!")
    except ValueError as e:
        print(f"\n❌ {e}")

import os
import requests
from huggingface_hub import snapshot_download

# Configuration
MODELS_DIR = os.path.join(os.getcwd(), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

print(f"📂 Setting up models in: {MODELS_DIR}")

# 1. Download Kokoro-82M (TTS)
print("⬇️  Downloading Kokoro ONNX...")
kokoro_path = os.path.join(MODELS_DIR, "kokoro-v0_19.onnx")
if not os.path.exists(kokoro_path):
    url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx"
    with open(kokoro_path, "wb") as f:
        f.write(requests.get(url).content)

# 2. Download Voices.npz
print("⬇️  Downloading Voices Config...")
voices_path = os.path.join(MODELS_DIR, "voices.npz")
if not os.path.exists(voices_path):
    url = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.npz"
    with open(voices_path, "wb") as f:
        f.write(requests.get(url).content)

# 3. Download Faster-Whisper Tiny (STT)
print("⬇️  Downloading Faster-Whisper (Tiny.en)...")
# This downloads the model files explicitly to your local folder
try:
    snapshot_download(
        repo_id="systran/faster-whisper-tiny.en",
        local_dir=os.path.join(MODELS_DIR, "faster-whisper-tiny.en"),
        local_dir_use_symlinks=False
    )
except ImportError:
    print("❌ Please run: pip install huggingface_hub")

print("\n✅ All models downloaded locally. You are ready for the AI Agent.")
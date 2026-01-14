"""
Vayu Voice Fingerprint Module
Novel echo detection using MFCC spectral fingerprints.

This module enables distinguishing between user speech and AI echo
by comparing acoustic fingerprints of the TTS output vs microphone input.
"""

import numpy as np
from typing import Optional, Tuple


def extract_mfcc(audio: np.ndarray, sample_rate: int = 16000, n_mfcc: int = 13) -> np.ndarray:
    """
    Extract MFCC (Mel-frequency cepstral coefficients) features from audio.
    
    This is a pure numpy implementation - no external dependencies required.
    
    Args:
        audio: Audio signal as numpy array
        sample_rate: Sample rate of the audio
        n_mfcc: Number of MFCC coefficients to extract
        
    Returns:
        MFCC features as numpy array (n_frames, n_mfcc)
    """
    if audio is None or len(audio) == 0:
        return np.zeros((1, n_mfcc))
    
    # Ensure audio is float and normalized
    audio = audio.astype(np.float32)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    
    # Pre-emphasis filter (boost high frequencies)
    pre_emphasis = 0.97
    emphasized = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
    
    # Frame parameters
    frame_size = 0.025  # 25ms frames
    frame_stride = 0.01  # 10ms hop
    frame_length = int(round(frame_size * sample_rate))
    frame_step = int(round(frame_stride * sample_rate))
    signal_length = len(emphasized)
    
    # Ensure at least 1 frame
    num_frames = max(1, int(np.ceil((signal_length - frame_length) / frame_step)) + 1)
    
    # Pad signal to ensure we have enough samples
    pad_length = (num_frames - 1) * frame_step + frame_length
    padded = np.pad(emphasized, (0, max(0, pad_length - signal_length)), mode='constant')
    
    # Frame the signal
    indices = np.arange(frame_length).reshape(1, -1) + np.arange(num_frames).reshape(-1, 1) * frame_step
    frames = padded[indices.astype(np.int32)]
    
    # Apply Hamming window
    hamming = np.hamming(frame_length)
    frames *= hamming
    
    # FFT
    NFFT = 512
    mag_frames = np.abs(np.fft.rfft(frames, NFFT))
    pow_frames = (1.0 / NFFT) * (mag_frames ** 2)
    
    # Mel filterbank
    n_filters = 26
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filters + 2)
    hz_points = 700 * (10 ** (mel_points / 2595) - 1)
    bin_points = np.floor((NFFT + 1) * hz_points / sample_rate).astype(int)
    
    # Create filterbank
    filterbank = np.zeros((n_filters, int(NFFT / 2 + 1)))
    for i in range(1, n_filters + 1):
        left = bin_points[i - 1]
        center = bin_points[i]
        right = bin_points[i + 1]
        
        for j in range(left, center):
            filterbank[i - 1, j] = (j - left) / (center - left)
        for j in range(center, right):
            filterbank[i - 1, j] = (right - j) / (right - center)
    
    # Apply filterbank
    filter_banks = np.dot(pow_frames, filterbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)  # Convert to dB
    
    # DCT to get MFCCs
    mfcc = np.zeros((num_frames, n_mfcc))
    for i in range(n_mfcc):
        mfcc[:, i] = np.sum(
            filter_banks * np.cos(np.pi * i * (np.arange(n_filters) + 0.5) / n_filters),
            axis=1
        )
    
    return mfcc


def compute_fingerprint(audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    Compute a compact voice fingerprint from audio.
    
    Args:
        audio: Audio signal
        sample_rate: Sample rate
        
    Returns:
        Fingerprint vector (mean MFCC across frames)
    """
    mfcc = extract_mfcc(audio, sample_rate)
    
    # Average across frames to get a single fingerprint vector
    fingerprint = np.mean(mfcc, axis=0)
    
    # Also include standard deviation for more robustness
    fingerprint_std = np.std(mfcc, axis=0)
    
    return np.concatenate([fingerprint, fingerprint_std])


def compute_similarity(fingerprint1: np.ndarray, fingerprint2: np.ndarray) -> float:
    """
    Compute cosine similarity between two fingerprints.
    
    Args:
        fingerprint1: First fingerprint vector
        fingerprint2: Second fingerprint vector
        
    Returns:
        Similarity score between 0 and 1 (1 = identical)
    """
    # Handle edge cases
    if fingerprint1 is None or fingerprint2 is None:
        return 0.0
    
    norm1 = np.linalg.norm(fingerprint1)
    norm2 = np.linalg.norm(fingerprint2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # Cosine similarity
    similarity = np.dot(fingerprint1, fingerprint2) / (norm1 * norm2)
    
    # Clamp to [0, 1]
    return float(np.clip(similarity, 0.0, 1.0))


def is_echo(
    tts_audio: np.ndarray,
    recorded_audio: np.ndarray,
    sample_rate: int = 16000,
    threshold: float = 0.6
) -> Tuple[bool, float]:
    """
    Determine if recorded audio is an echo of the TTS output.
    
    This is the core of the novel echo detection technique.
    
    Args:
        tts_audio: The audio that was played through speakers (TTS output)
        recorded_audio: The audio picked up by the microphone
        sample_rate: Sample rate of both audio signals
        threshold: Similarity threshold above which we consider it echo
        
    Returns:
        Tuple of (is_echo: bool, similarity_score: float)
    """
    if tts_audio is None or len(tts_audio) == 0:
        return False, 0.0
    
    if recorded_audio is None or len(recorded_audio) == 0:
        return False, 0.0
    
    # Extract fingerprints
    tts_fingerprint = compute_fingerprint(tts_audio, sample_rate)
    recorded_fingerprint = compute_fingerprint(recorded_audio, sample_rate)
    
    # Compute similarity
    similarity = compute_similarity(tts_fingerprint, recorded_fingerprint)
    
    # If similarity is high, it's likely echo
    is_echo_detected = similarity > threshold
    
    return is_echo_detected, similarity


# Quick test
if __name__ == "__main__":
    print("Testing Voice Fingerprint Module...")
    
    # Generate test signals
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Original signal (simulating TTS)
    original = np.sin(2 * np.pi * 440 * t) * 0.5
    
    # Echo (same signal with noise and scaling)
    echo = original * 0.3 + np.random.randn(len(t)) * 0.05
    
    # Different signal (simulating user speech)
    different = np.sin(2 * np.pi * 220 * t + np.sin(2 * np.pi * 5 * t)) * 0.4
    
    # Test echo detection
    is_echo_result, sim_echo = is_echo(original, echo, sample_rate)
    print(f"Echo test: is_echo={is_echo_result}, similarity={sim_echo:.4f}")
    
    is_different_result, sim_different = is_echo(original, different, sample_rate)
    print(f"Different test: is_echo={is_different_result}, similarity={sim_different:.4f}")
    
    print("\nExpected: Echo should have high similarity, Different should have low similarity")

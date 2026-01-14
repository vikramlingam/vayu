"""
Vayu Sound Effects Module
Generates pleasant notification sounds programmatically.
"""

import numpy as np
from typing import Optional


def generate_chime(
    duration: float = 0.3,
    sample_rate: int = 24000,
    frequencies: tuple = (880, 1320, 1760),  # A5, E6, A6 chord
    fade_out: float = 0.15
) -> np.ndarray:
    """
    Generate a pleasant chime sound.
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate
        frequencies: Tuple of frequencies for the chord
        fade_out: Fade out duration in seconds
        
    Returns:
        Audio samples as float32 numpy array
    """
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    
    # Generate harmonics
    chime = np.zeros_like(t)
    for i, freq in enumerate(frequencies):
        # Each note slightly delayed and with different amplitude
        delay_samples = int(i * 0.02 * sample_rate)  # 20ms stagger
        amplitude = 0.4 / (i + 1)  # Decreasing amplitude for higher notes
        
        # Sine wave with slight decay
        note = amplitude * np.sin(2 * np.pi * freq * t) * np.exp(-t * 3)
        
        # Add delay by shifting
        if delay_samples > 0:
            note = np.concatenate([np.zeros(delay_samples, dtype=np.float32), note[:-delay_samples]])
        
        chime += note
    
    # Apply fade out
    fade_samples = int(fade_out * sample_rate)
    if fade_samples > 0 and fade_samples < len(chime):
        fade_curve = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        chime[-fade_samples:] *= fade_curve
    
    # Normalize
    max_val = np.abs(chime).max()
    if max_val > 0:
        chime = chime / max_val * 0.5  # 50% volume
    
    return chime.astype(np.float32)


def generate_thinking_chime(sample_rate: int = 24000) -> np.ndarray:
    """
    Generate a soft, pleasant 'thinking' chime - played before AI responds.
    
    Two quick ascending notes that feel like "I'm processing..."
    """
    duration = 0.25
    t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
    
    # Two ascending notes: C6 (1047 Hz) -> G6 (1568 Hz)
    note1_freq = 1047  # C6
    note2_freq = 1568  # G6
    
    # First note (short, first half)
    half = len(t) // 2
    note1 = 0.3 * np.sin(2 * np.pi * note1_freq * t[:half]) * np.exp(-t[:half] * 8)
    
    # Second note (slightly longer, second half)
    note2 = 0.35 * np.sin(2 * np.pi * note2_freq * t[:half + half//2]) * np.exp(-t[:half + half//2] * 6)
    
    # Combine with slight overlap
    chime = np.zeros_like(t)
    chime[:half] = note1
    chime[half - half//4:half - half//4 + len(note2)] += note2[:min(len(note2), len(t) - (half - half//4))]
    
    # Normalize
    max_val = np.abs(chime).max()
    if max_val > 0:
        chime = chime / max_val * 0.4  # 40% volume - subtle
    
    return chime.astype(np.float32)


# Pre-generate the chime for instant playback
_cached_thinking_chime: Optional[np.ndarray] = None

def get_thinking_chime(sample_rate: int = 24000) -> np.ndarray:
    """Get the cached thinking chime sound."""
    global _cached_thinking_chime
    if _cached_thinking_chime is None:
        _cached_thinking_chime = generate_thinking_chime(sample_rate)
    return _cached_thinking_chime


if __name__ == "__main__":
    import sounddevice as sd
    
    print("Testing sound effects...")
    
    # Test thinking chime
    print("Playing thinking chime...")
    chime = get_thinking_chime(24000)
    sd.play(chime, 24000)
    sd.wait()
    
    print("Done!")

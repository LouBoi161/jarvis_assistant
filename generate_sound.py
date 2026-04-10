import numpy as np
import wave
import struct

def generate_chime(filename="notification.wav"):
    sample_rate = 16000
    duration = 0.3  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Mix of two frequencies for a pleasant "harmonic" chime (A5 and E6)
    # Plus an exponential decay envelope
    envelope = np.exp(-10 * t)
    wave_data = (np.sin(2 * np.pi * 880 * t) + 0.5 * np.sin(2 * np.pi * 1320 * t)) * envelope
    
    # Normalize to 16-bit range
    wave_data = (wave_data / np.max(np.abs(wave_data)) * 32767).astype(np.int16)
    
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for sample in wave_data:
            wf.writeframes(struct.pack('<h', sample))
    print(f"Notification sound saved to {filename}")

if __name__ == "__main__":
    generate_chime()

import pyaudio
import numpy as np
import openwakeword
from openwakeword.model import Model
import torch
import time
import wave
import os

# PyAudio setup
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280 # 80ms chunks for wake word

audio = pyaudio.PyAudio()

# Silero VAD Setup
print("Loading Silero VAD...")
try:
    # Use local cache if possible, or download
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
except Exception as e:
    print(f"Error loading VAD: {e}")
    vad_model = None

print("VAD loaded.")

def get_audio_stream(chunk_size):
    return audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=chunk_size)

def play_notification(filename="notification.wav"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, filename)
    
    if not os.path.exists(full_path):
        return
    
    p_play = None
    try:
        wf = wave.open(full_path, 'rb')
        p_play = pyaudio.PyAudio()
        
        stream = p_play.open(format=p_play.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True,
                        frames_per_buffer=2048)
        
        data = wf.readframes(1024)
        while len(data) > 0:
            stream.write(data)
            data = wf.readframes(1024)
        
        time.sleep(0.1)
        stream.stop_stream()
        stream.close()
    except Exception as e:
        print(f"Fehler beim Abspielen des Benachrichtigungstons: {e}")
    finally:
        if p_play:
            p_play.terminate()

def listen_for_wakeword(interrupt_check=None):
    try:
        oww_model = Model(wakeword_models=["hey_jarvis"])
    except Exception:
        oww_model = Model()

    mic_stream = get_audio_stream(CHUNK)
    
    for _ in range(6):
        mic_stream.read(CHUNK, exception_on_overflow=False)
        
    try:
        while True:
            if interrupt_check and interrupt_check():
                mic_stream.stop_stream()
                mic_stream.close()
                return False

            data = mic_stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            prediction = oww_model.predict(audio_data)
            
            for mdl in oww_model.prediction_buffer.keys():
                if oww_model.prediction_buffer[mdl][-1] > 0.5:
                    if any(x in mdl.lower() for x in ['jarvis', 'hey_jarvis']):
                        mic_stream.stop_stream()
                        mic_stream.close()
                        play_notification()
                        return True
    except KeyboardInterrupt:
        mic_stream.stop_stream()
        mic_stream.close()
        exit(0)

def record_until_silence(silence_duration=4.0):
    """Records audio until no voice is detected for silence_duration seconds."""
    # VAD works best with specific chunk sizes (e.g., 512, 1024, 1536)
    CHUNK_SIZE = 512
    mic_stream = get_audio_stream(CHUNK_SIZE)
    
    recording = []
    no_voice_start = None
    has_spoken = False
    
    print("Listening for voice input...")
    
    try:
        while True:
            data = mic_stream.read(CHUNK_SIZE, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            recording.append(audio_data)
            
            if vad_model:
                # Normalization and Tensor conversion for Silero
                tensor_data = torch.from_numpy(audio_data.astype(np.float32) / 32768.0)
                # Check for speech probability
                speech_prob = vad_model(tensor_data, RATE).item()
                
                # Speech detected
                if speech_prob > 0.45: # Slightly lower threshold for robustness
                    no_voice_start = None
                    has_spoken = True
                else:
                    if has_spoken:
                        if no_voice_start is None:
                            no_voice_start = time.time()
                        elif time.time() - no_voice_start > silence_duration:
                            print(f"Voice ended (no voice for {silence_duration}s). Stopping recording.")
                            break
                    else:
                        # Timeout if no speech is detected at all
                        if no_voice_start is None:
                            no_voice_start = time.time()
                        elif time.time() - no_voice_start > 10.0:
                            print("Timeout: No voice detected.")
                            break
            else:
                # Fallback to simple RMS if VAD is missing (less robust)
                rms = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
                if rms > 500: # Simple threshold
                    no_voice_start = None
                    has_spoken = True
                else:
                    if has_spoken:
                        if no_voice_start is None:
                            no_voice_start = time.time()
                        elif time.time() - no_voice_start > silence_duration:
                            break
                    elif no_voice_start is None:
                        no_voice_start = time.time()
                    elif time.time() - no_voice_start > 10.0:
                        break
    finally:
        mic_stream.stop_stream()
        mic_stream.close()
        
    if not recording:
        return np.array([], dtype=np.int16)
        
    full_audio = np.concatenate(recording)
    return full_audio

def save_wav(filename, audio_data):
    if len(audio_data) == 0:
        return
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(audio_data.tobytes())

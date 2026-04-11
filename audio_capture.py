import pyaudio
import numpy as np
import openwakeword
from openwakeword.model import Model
import torch
import time
import wave
import os

# openwakeword model setup
# The models are usually downloaded automatically when the Model() class is initialized.
# We skip the explicit utility call that is causing the AttributeError.

# PyAudio setup
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280 # 80ms chunks for wake word

audio = pyaudio.PyAudio()

# Silero VAD Setup
print("Loading Silero VAD...")
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
print("VAD loaded.")

def get_audio_stream(chunk_size):
    return audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=chunk_size)

def play_notification(filename="notification.wav"):
    # Ermittle den absoluten Pfad zur Sounddatei relativ zu diesem Skript
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, filename)
    
    if not os.path.exists(full_path):
        return
    try:
        wf = wave.open(full_path, 'rb')
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True,
                        frames_per_buffer=4096)
        data = wf.readframes(4096)
        while len(data) > 0:
            stream.write(data)
            data = wf.readframes(4096)
        stream.stop_stream()
        stream.close()
        p.terminate()
    except Exception as e:
        print(f"Fehler beim Abspielen des Benachrichtigungstons: {e}")

def listen_for_wakeword(interrupt_check=None):
    # Wir laden das Modell bei jedem neuen Durchlauf frisch in den Speicher. 
    # Das dauert nur wenige Millisekunden, garantiert aber, dass der interne Puffer 
    # (der für das sofortige Auslösen verantwortlich war) absolut leer ist.
    try:
        oww_model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
    except Exception:
        oww_model = Model()

    mic_stream = get_audio_stream(CHUNK)
    
    # 0.5 Sekunden "Stille" einlesen, um Hardware-Puffer vom TTS auszulöschen
    for _ in range(6):
        mic_stream.read(CHUNK, exception_on_overflow=False)
        
    try:
        while True:
            # Überprüfe, ob der Modus gewechselt wurde
            if interrupt_check and interrupt_check():
                mic_stream.stop_stream()
                mic_stream.close()
                return False

            data = mic_stream.read(CHUNK, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            # Predict wake word
            prediction = oww_model.predict(audio_data)
            
            for mdl in oww_model.prediction_buffer.keys():
                # Trigger threshold 0.5
                if oww_model.prediction_buffer[mdl][-1] > 0.5:
                    if 'jarvis' in mdl.lower() or 'alexa' in mdl.lower() or 'mycroft' in mdl.lower():
                        mic_stream.stop_stream()
                        mic_stream.close()
                        # Spiele den Sound ab
                        play_notification()
                        return True
    except KeyboardInterrupt:
        mic_stream.stop_stream()
        mic_stream.close()
        exit(0)

def record_until_silence(silence_duration=5.0):
    # VAD works well with 512 frames
    mic_stream = get_audio_stream(512)
    
    recording = []
    silence_start = None
    has_spoken = False
    
    try:
        while True:
            data = mic_stream.read(512, exception_on_overflow=False)
            audio_data = np.frombuffer(data, dtype=np.int16)
            recording.append(audio_data)
            
            # Convert to float32 tensor for Silero VAD
            tensor_data = torch.FloatTensor(audio_data) / 32768.0
            
            # Check for speech
            speech_prob = vad_model(tensor_data, RATE).item()
            
            if speech_prob > 0.5:
                # Reset silence timer if speech is detected
                silence_start = None
                has_spoken = True
            else:
                if has_spoken:
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > silence_duration:
                        break
                else:
                    # If user hasn't spoken yet after 10s, timeout
                    if silence_start is None:
                        silence_start = time.time()
                    elif time.time() - silence_start > 10.0:
                        break
                    
    finally:
        mic_stream.stop_stream()
        mic_stream.close()
        
    full_audio = np.concatenate(recording)
    return full_audio

def save_wav(filename, audio_data):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(audio_data.tobytes())

if __name__ == "__main__":
    while True:
        listen_for_wakeword()
        audio_data = record_until_silence()
        print(f"Recorded {len(audio_data) / RATE:.2f} seconds of audio.")
        # Debug: save to file
        save_wav("latest_input.wav", audio_data)
        print("Saved to latest_input.wav. Ready for next loop...\n")

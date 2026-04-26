import os
import torch
import whisper
import soundfile as sf
import pyaudio
import wave
import numpy as np
import re
import threading
import queue
import time
import subprocess
import urllib.request
import gc

# Check if kokoro-onnx is available
try:
    from kokoro_onnx import Kokoro
except ImportError:
    Kokoro = None

class TTSEngine:
    def __init__(self, config=None, use_gpu=True, stt_model=None):
        self.config = config or {}
        self.tts_type = self.config.get("tts_type", "kokoro-tts")
        self.piper_voice = self.config.get("piper_voice", "de_DE-thorsten-high")
        self.kokoro_voice = self.config.get("kokoro_voice", "gf_eva")
        self.qwen_voice = self.config.get("qwen_voice", "default.wav")
        self.stt_model = stt_model
        self.use_gpu = use_gpu
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        self.kokoro_instance = None 
        self.qwen_model = None
        self.qwen_tokenizer = None
        
        # Paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.piper_dir = os.path.join(script_dir, "piper_models")
        self.kokoro_dir = os.path.join(script_dir, "kokoro_models")
        self.voices_dir = os.path.join(script_dir, "voices")
        
        for d in [self.piper_dir, self.kokoro_dir, self.voices_dir]:
            if not os.path.exists(d): os.makedirs(d)
            
        self.piper_binary = os.path.join(script_dir, ".venv", "bin", "piper")
        if not os.path.exists(self.piper_binary): self.piper_binary = "piper"

        print(f"TTS Engine initialized. Mode: {self.tts_type}")

    def unload_models(self):
        if self.kokoro_instance is not None:
            del self.kokoro_instance
            self.kokoro_instance = None
        if self.qwen_model is not None:
            del self.qwen_model
            self.qwen_model = None
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

    def _ensure_kokoro_loaded(self):
        if self.kokoro_instance is not None: return True
        if Kokoro is None: return False
        m_path = os.path.join(self.kokoro_dir, "kokoro-v1.0.int8.onnx")
        v_path = os.path.join(self.kokoro_dir, "voices-v1.0.bin")
        if not os.path.exists(m_path):
            urllib.request.urlretrieve("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.int8.onnx", m_path)
        if not os.path.exists(v_path):
            urllib.request.urlretrieve("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin", v_path)
        try:
            self.kokoro_instance = Kokoro(m_path, v_path)
            return True
        except: return False

    def _ensure_piper_loaded(self):
        model_path = os.path.join(self.piper_dir, f"{self.piper_voice}.onnx")
        config_path = os.path.join(self.piper_dir, f"{self.piper_voice}.onnx.json")
        if not os.path.exists(model_path) or not os.path.exists(config_path):
            base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
            parts = self.piper_voice.split('-')
            if len(parts) >= 3:
                lang_country = parts[0]; voice_name = parts[1]; quality = parts[2]; lang_code = lang_country.split('_')[0]
                repo_path = f"{lang_code}/{lang_country}/{voice_name}/{quality}/{self.piper_voice}.onnx"
                urllib.request.urlretrieve(f"{base_url}/{repo_path}", model_path)
                urllib.request.urlretrieve(f"{base_url}/{repo_path}.json", config_path)
                return True
            return False
        return True

    def _ensure_qwen_loaded(self):
        if self.qwen_model is not None: return True
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model_id = "Qwen/Qwen3-TTS-12Hz-0.6B"
            self.qwen_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            self.qwen_model = AutoModelForCausalLM.from_pretrained(model_id, device_map=self.device, trust_remote_code=True)
            return True
        except Exception as e:
            print(f"Fehler beim Laden von Qwen3-TTS: {e}")
            return False

    def speak(self, text: str, interrupt_event=None):
        if not text.strip() or self.tts_type == "none": return
        text = re.sub(r"\[.*?\]", "", text)
        text = re.sub(r"<.*?>", "", text, flags=re.DOTALL)
        text = text.replace("\n", " ").strip()
        if not text: return

        if self.tts_type == "kokoro-tts":
            if self._ensure_kokoro_loaded(): self._speak_kokoro(text, interrupt_event)
        elif self.tts_type == "piper-tts":
            if self._ensure_piper_loaded(): self._speak_piper(text, interrupt_event)
        elif self.tts_type == "qwen3-tts":
            if self._ensure_qwen_loaded(): self._speak_qwen(text, interrupt_event)

    def _speak_kokoro(self, text, interrupt_event):
        try:
            lang = "de" if any(x in self.kokoro_voice for x in ["gf_", "gm_"]) else "en-us"
            samples, sr = self.kokoro_instance.create(text, voice=self.kokoro_voice, speed=1.0, lang=lang)
            temp = f"temp_kokoro_{threading.get_ident()}.wav"
            sf.write(temp, samples, sr)
            self._play_wav(temp, interrupt_event)
            if os.path.exists(temp): os.remove(temp)
        except: pass

    def _speak_piper(self, text, interrupt_event):
        temp = f"temp_piper_{threading.get_ident()}.wav"
        model_path = os.path.join(self.piper_dir, f"{self.piper_voice}.onnx")
        try:
            process = subprocess.Popen([self.piper_binary, "--model", model_path, "--output_file", temp], stdin=subprocess.PIPE)
            process.communicate(input=text.encode('utf-8'))
            if os.path.exists(temp):
                self._play_wav(temp, interrupt_event); os.remove(temp)
        except: pass

    def _speak_qwen(self, text, interrupt_event):
        try:
            voice_path = os.path.join(self.voices_dir, self.qwen_voice)
            if not os.path.exists(voice_path):
                print(f"Stimme {self.qwen_voice} nicht gefunden. Nutze Standard.")
                # Fallback oder Abbruch
                return

            # Voice Cloning Logic
            # Hinweis: Das ist ein Platzhalter für das tatsächliche Inference-Skript von Qwen3-TTS
            # Da Qwen3-TTS oft Audio-Ref als Input nimmt:
            inputs = self.qwen_tokenizer(text, voice_ref=voice_path, return_tensors="pt").to(self.device)
            with torch.no_grad():
                output_audio = self.qwen_model.generate(**inputs)
            
            temp = f"temp_qwen_{threading.get_ident()}.wav"
            sf.write(temp, output_audio.cpu().numpy(), 24000) # Beispiel SR
            self._play_wav(temp, interrupt_event)
            if os.path.exists(temp): os.remove(temp)
        except Exception as e:
            print(f"Fehler bei Qwen-TTS: {e}")

    def _play_wav(self, path, interrupt_event):
        p = pyaudio.PyAudio()
        try:
            wf = wave.open(path, 'rb')
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(), rate=wf.getframerate(), output=True)
            data = wf.readframes(1024)
            while len(data) > 0:
                if interrupt_event and interrupt_event.is_set(): break
                stream.write(data); data = wf.readframes(1024)
            stream.stop_stream(); stream.close(); wf.close()
        except: pass
        finally: p.terminate()

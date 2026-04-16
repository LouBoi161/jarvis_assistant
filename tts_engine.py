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

# Check if qwen-tts is available, but don't fail here
try:
    from qwen_tts import Qwen3TTSModel
except ImportError:
    Qwen3TTSModel = None

class TTSEngine:
    def __init__(self, config=None, use_gpu=True, stt_model=None):
        self.config = config or {}
        self.tts_type = self.config.get("tts_type", "qwen3-tts")
        self.piper_voice = self.config.get("piper_voice", "de_DE-thorsten-high")
        self.qwen_voice = self.config.get("qwen_voice", "default.wav")
        self.stt_model = stt_model
        self.use_gpu = use_gpu
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        self.model = None # Qwen model
        self.voice_clone_prompt = None
        self.ref_wav = None
        self.ref_text = ""
        
        # Piper paths
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.piper_dir = os.path.join(script_dir, "piper_models")
        if not os.path.exists(self.piper_dir):
            os.makedirs(self.piper_dir)
            
        self.piper_binary = os.path.join(script_dir, ".venv", "bin", "piper")
        if not os.path.exists(self.piper_binary):
            self.piper_binary = "piper" # Fallback to PATH

        # Lazy loading: Don't load anything in __init__ if not needed
        # only if tts_type is not "none", we might want to pre-load, 
        # but to save RAM we wait for the first 'speak' call.
        print(f"TTS Engine initialized (Lazy). Type: {self.tts_type}")

    def unload_models(self):
        """Unloads all models from memory/VRAM."""
        print("Unloading TTS models to save memory...")
        if self.model is not None:
            del self.model
            self.model = None
        
        self.voice_clone_prompt = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def _ensure_qwen_loaded(self):
        if self.model is not None:
            return
        
        if Qwen3TTSModel is None:
            print("Error: qwen-tts package not installed.")
            return

        print(f"Loading Qwen3-TTS on {self.device}...")
        try:
            self.model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                device_map=self.device,
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
            )
        except Exception as e:
            print(f"Qwen loading failed: {e}. Falling back to CPU.")
            self.device = "cpu"
            self.model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                device_map=self.device,
                dtype=torch.float32
            )

        # Prepare reference voice for cloning
        if not self.stt_model:
            self.stt_model = whisper.load_model("base")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        voices_dir = os.path.join(script_dir, "voices")
        self.ref_wav = os.path.join(voices_dir, self.qwen_voice)
        
        if os.path.exists(self.ref_wav):
            ref_txt_file = self.ref_wav.rsplit(".", 1)[0] + ".txt"
            if os.path.exists(ref_txt_file):
                with open(ref_txt_file, "r") as f:
                    self.ref_text = f.read().strip()
            else:
                print(f"Analyzing reference voice {self.qwen_voice}...")
                # Automatische Spracherkennung für Referenz-Stimme
                result = self.stt_model.transcribe(self.ref_wav)
                self.ref_text = result["text"].strip()
                with open(ref_txt_file, "w") as f:
                    f.write(self.ref_text)
            
            try:
                self.voice_clone_prompt = self.model.create_voice_clone_prompt(
                    ref_audio=self.ref_wav,
                    ref_text=self.ref_text
                )
            except Exception as e:
                print(f"Voice clone prompt error: {e}")

    def _ensure_piper_loaded(self):
        """Ensures the piper model and config are downloaded."""
        model_path = os.path.join(self.piper_dir, f"{self.piper_voice}.onnx")
        config_path = os.path.join(self.piper_dir, f"{self.piper_voice}.onnx.json")
        
        if not os.path.exists(model_path) or not os.path.exists(config_path):
            print(f"Piper model {self.piper_voice} not found. Downloading...")
            base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
            
            # Construct download URL (this is a heuristic for piper's repo structure)
            lang_code = self.piper_voice.split('-')[0]
            country_code = self.piper_voice.split('-')[1]
            voice_name = self.piper_voice.split('-')[2]
            quality = self.piper_voice.split('-')[3]
            
            # Example: de/de_DE/thorsten/high/de_DE-thorsten-high.onnx
            repo_path = f"{lang_code}/{lang_code}_{country_code.upper()}/{voice_name}/{quality}/{self.piper_voice}.onnx"
            
            try:
                print(f"Downloading from {base_url}/{repo_path}...")
                urllib.request.urlretrieve(f"{base_url}/{repo_path}", model_path)
                urllib.request.urlretrieve(f"{base_url}/{repo_path}.json", config_path)
                print("Download complete.")
            except Exception as e:
                print(f"Download failed: {e}")
                # Fallback to a simpler structure if needed or report error
                return False
        return True

    def speak(self, text: str, interrupt_event=None):
        if not text.strip() or self.tts_type == "none":
            self.unload_models() # Ensure memory is free if TTS is disabled
            return

        # Prepare text (remove tags, format time, etc.)
        display_text = re.sub(r"\[[A-Za-zäöüß ]+\]", "", text)
        display_text = re.sub(r"<[^>]+>.*?</[^>]+>", "", display_text, flags=re.DOTALL)
        display_text = re.sub(r"<[^>]+>", "", display_text).strip()
        
        clean_text = re.sub(r"(\d{1,2}):(\d{2})", r"\1 Uhr \2", display_text)
        clean_text = clean_text.replace("\n", " ").replace("###", "").replace("***", "").replace("`", "").replace("*", "").replace("_", "")
        clean_text = clean_text.replace("(", " ").replace(")", " ")
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

        if not clean_text:
            return

        if self.tts_type == "qwen3-tts":
            self._ensure_qwen_loaded()
            self._speak_qwen(text, clean_text, interrupt_event)
        elif self.tts_type == "piper-tts":
            # Before loading Piper, we can unload Qwen to save VRAM
            if self.model is not None:
                self.unload_models()
            
            if self._ensure_piper_loaded():
                self._speak_piper(clean_text, interrupt_event)

    def _speak_qwen(self, original_text, clean_text, interrupt_event):
        tags = re.findall(r"\[([A-Za-zäöüß ]+)+\]", original_text)
        instruction = "Speak with high emotional variance, very expressive, natural pauses, and a friendly tone."
        
        if tags:
            tag = tags[0].lower()
            tag_map = {
                "aufgeregt": "Speak with extreme excitement, very high energy, fast and joyful!",
                "freundlich": "Speak in a very friendly, professional, and pleasant voice.",
                "traurig": "Speak with a very sad, shaky, and emotional voice, low energy.",
                "wütend": "Speak with an angry, loud, and aggressive tone, very frustrated.",
                "glücklich": "Speak with a wide smile in your voice, very cheerful and bright.",
                "nachdenklich": "Speak slowly, thoughtfully, with realistic 'hmm' pauses and soft tone."
            }
            instruction = tag_map.get(tag, instruction)

        MAX_CHARS = 400
        chunks = [clean_text] if len(clean_text) <= MAX_CHARS else [clean_text[:MAX_CHARS], clean_text[MAX_CHARS:]]
        audio_queue = queue.Queue()
        
        def generator_worker():
            for i, chunk in enumerate(chunks):
                if interrupt_event and interrupt_event.is_set(): break
                if not chunk: continue
                output_file = f"temp_chunk_{i}_{threading.get_ident()}.wav"
                try:
                    if self.voice_clone_prompt is not None:
                        wavs, sr = self.model.generate_voice_clone(
                            text=chunk, language="German", 
                            voice_clone_prompt=self.voice_clone_prompt, 
                            instruction=instruction
                        )
                    elif self.ref_wav and os.path.exists(self.ref_wav):
                        wavs, sr = self.model.generate_voice_clone(
                            text=chunk, language="German", ref_audio=self.ref_wav,
                            ref_text=self.ref_text, instruction=instruction
                        )
                    else:
                        wavs, sr = self.model.generate_custom_voice(
                            text=chunk, language="German", speaker="Claribel Dervla",
                            instruction=instruction
                        )
                    sf.write(output_file, wavs[0], sr)
                    audio_queue.put(output_file)
                except Exception as e:
                    print(f"Qwen generation error: {e}")
                    audio_queue.put(None)
            audio_queue.put("DONE")

        threading.Thread(target=generator_worker).start()
        self._play_audio_queue(audio_queue, interrupt_event)

    def _speak_piper(self, text, interrupt_event):
        output_file = f"piper_temp_{threading.get_ident()}.wav"
        model_path = os.path.join(self.piper_dir, f"{self.piper_voice}.onnx")
        
        try:
            # Piper expects text via stdin
            process = subprocess.Popen(
                [self.piper_binary, "--model", model_path, "--output_file", output_file],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate(input=text.encode('utf-8'))
            
            if process.returncode != 0:
                print(f"Piper process failed (code {process.returncode}): {stderr.decode()}")
                return

            if os.path.exists(output_file):
                self._play_wav(output_file, interrupt_event)
                try:
                    os.remove(output_file)
                except:
                    pass
        except Exception as e:
            print(f"Piper error: {e}")

    def _play_audio_queue(self, audio_queue, interrupt_event):
        p = pyaudio.PyAudio()
        while True:
            if interrupt_event and interrupt_event.is_set(): break
            try:
                file_path = audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if file_path == "DONE": break
            if file_path is None: continue
            
            self._play_wav_with_pyaudio(p, file_path, interrupt_event)
            if os.path.exists(file_path): 
                try:
                    os.remove(file_path)
                except:
                    pass
        p.terminate()

    def _play_wav(self, file_path, interrupt_event):
        p = pyaudio.PyAudio()
        self._play_wav_with_pyaudio(p, file_path, interrupt_event)
        p.terminate()

    def _play_wav_with_pyaudio(self, p, file_path, interrupt_event):
        try:
            wf = wave.open(file_path, 'rb')
            stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                            channels=wf.getnchannels(), rate=wf.getframerate(), output=True,
                            frames_per_buffer=2048)
            data = wf.readframes(1024)
            while len(data) > 0:
                if interrupt_event and interrupt_event.is_set(): break
                stream.write(data)
                data = wf.readframes(1024)
            stream.stop_stream()
            stream.close()
            wf.close()
        except Exception as e:
            print(f"Audio playback error: {e}")

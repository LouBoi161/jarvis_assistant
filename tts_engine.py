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
from qwen_tts import Qwen3TTSModel

class TTSEngine:
    def __init__(self, config=None, use_gpu=True, stt_model=None):
        self.config = config or {}
        self.tts_type = self.config.get("tts_type", "qwen3-tts")
        self.piper_voice = self.config.get("piper_voice", "de_DE-thorsten-high")
        self.qwen_voice = self.config.get("qwen_voice", "default.wav")
        self.stt_model = stt_model
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        self.model = None
        self.voice_clone_prompt = None
        self.ref_wav = None
        self.ref_text = ""
        
        # Path to local piper binary in .venv
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.piper_binary = os.path.join(script_dir, ".venv", "bin", "piper")
        if not os.path.exists(self.piper_binary):
            self.piper_binary = "piper" # Fallback to PATH

        if self.tts_type == "qwen3-tts":
            self._init_qwen()
        elif self.tts_type == "piper-tts":
            self._init_piper()
        
        print(f"TTS Engine initialized with type: {self.tts_type}")

    def _init_qwen(self):
        print("Lade Qwen3-TTS (0.6B Base)...")
        try:
            self.model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                device_map=self.device,
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float32
            )
        except Exception as e:
            print(f"Fehler beim Laden von Qwen3: {e}. Nutze CPU-Fallback...")
            self.device = "cpu"
            self.model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                device_map=self.device,
                dtype=torch.float32
            )
        
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
                print(f"Analyzing {self.qwen_voice} automatically...")
                result = self.stt_model.transcribe(self.ref_wav)
                self.ref_text = result["text"].strip()
            
            try:
                self.voice_clone_prompt = self.model.create_voice_clone_prompt(
                    ref_audio=self.ref_wav,
                    ref_text=self.ref_text
                )
            except Exception as e:
                print(f"Warning: Could not pre-calculate voice clone prompt: {e}")

    def _init_piper(self):
        print(f"Using Piper TTS with voice: {self.piper_voice}")
        pass

    def speak(self, text: str, interrupt_event=None):
        if not text.strip() or self.tts_type == "none":
            return

        display_text = re.sub(r"\[[A-Za-zäöüß ]+\]", "", text)
        display_text = re.sub(r"<[^>]+>.*?</[^>]+>", "", display_text, flags=re.DOTALL)
        display_text = re.sub(r"<[^>]+>", "", display_text).strip()
        
        clean_text = re.sub(r"(\d{1,2}):(\d{2})", r"\1 Uhr \2", display_text)
        clean_text = clean_text.replace("\n", " ").replace("###", "").replace("***", "").replace("`", "").replace("*", "").replace("_", "")
        clean_text = clean_text.replace("(", " ").replace(")", " ")
        clean_text = re.sub(r"\s+", " ", clean_text).strip()

        if self.tts_type == "qwen3-tts":
            self._speak_qwen(text, clean_text, interrupt_event)
        elif self.tts_type == "piper-tts":
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
                except Exception:
                    audio_queue.put(None)
            audio_queue.put("DONE")

        threading.Thread(target=generator_worker).start()
        self._play_audio_queue(audio_queue, interrupt_event)

    def _speak_piper(self, text, interrupt_event):
        output_file = f"piper_temp_{threading.get_ident()}.wav"
        try:
            # Check if model exists, piper will auto-download if name is provided
            # Piper command: piper --model <name_or_path> --output_file <path>
            process = subprocess.Popen(
                [self.piper_binary, "--model", self.piper_voice, "--output_file", output_file],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            process.communicate(input=text.encode('utf-8'))
            
            if os.path.exists(output_file):
                self._play_wav(output_file, interrupt_event)
                os.remove(output_file)
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
            if os.path.exists(file_path): os.remove(file_path)
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
        except:
            pass

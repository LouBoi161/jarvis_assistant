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
from qwen_tts import Qwen3TTSModel

class TTSEngine:
    def __init__(self, use_gpu=True, stt_model=None):
        # Wir nutzen jetzt die GPU, da wir den VRAM-Verbrauch von Ollama optimieren.
        print("Lade Qwen3-TTS (0.6B Base) auf GPU (CUDA)...")
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        try:
            self.model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                device_map=self.device,
                dtype=torch.bfloat16
            )
        except Exception as e:
            print(f"Fehler beim Laden von Qwen3 auf GPU: {e}. Nutze CPU-Fallback...")
            self.device = "cpu"
            self.model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                device_map=self.device,
                dtype=torch.float32
            )
        
        # Nutze das bereits geladene Whisper Modell falls vorhanden, sonst lade es neu
        if stt_model:
            self.stt_model = stt_model
        else:
            self.stt_model = whisper.load_model("base")
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.ref_wav = os.path.join(script_dir, "voice.wav")
        self.ref_txt_file = os.path.join(script_dir, "voice.txt")
        self.ref_text = ""
        self.voice_clone_prompt = None
        
        if os.path.exists(self.ref_wav):
            if os.path.exists(self.ref_txt_file):
                with open(self.ref_txt_file, "r") as f:
                    self.ref_text = f.read().strip()
            else:
                print("Analyzing voice.wav automatically for perfect cloning...")
                result = self.stt_model.transcribe(self.ref_wav)
                self.ref_text = result["text"].strip()
            
            print(f"Reference text for Qwen3: '{self.ref_text}'")
            print("Pre-calculating voice clone prompt for maximum speed...")
            try:
                self.voice_clone_prompt = self.model.create_voice_clone_prompt(
                    ref_audio=self.ref_wav,
                    ref_text=self.ref_text
                )
                print("Voice clone prompt cached successfully.")
            except Exception as e:
                print(f"Warning: Could not pre-calculate voice clone prompt: {e}")
        
        print("Qwen3-TTS ready.")

    def speak(self, text: str, interrupt_event=None):
        """Generates audio with Qwen3-TTS and plays it. Uses background generation for speed."""
        if not text.strip():
            return
            
        import re
        import threading
        import queue
        
        tags = re.findall(r"\[([A-Za-zäöüß ]+)+\]", text)
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

        display_text = re.sub(r"\[[A-Za-zäöüß ]+\]", "", text)
        display_text = re.sub(r"<[^>]+>.*?</[^>]+>", "", display_text, flags=re.DOTALL)
        display_text = re.sub(r"<[^>]+>", "", display_text).strip()
        
        # --- TEXT NORMALISIERUNG FÜR TTS ---
        clean_text = re.sub(r"(\d{1,2}):(\d{2})", r"\1 Uhr \2", display_text)
        clean_text = clean_text.replace("\n", " ").replace("###", "").replace("***", "").replace("`", "").replace("*", "").replace("_", "")
        clean_text = clean_text.replace("(", " ").replace(")", " ")
        clean_text = re.sub(r"\s+", " ", clean_text).strip()
        
        MAX_CHARS = 400
        chunks = [clean_text] if len(clean_text) <= MAX_CHARS else [clean_text[:MAX_CHARS], clean_text[MAX_CHARS:]]

        audio_queue = queue.Queue()
        
        def generator_worker():
            for i, chunk in enumerate(chunks):
                if interrupt_event and interrupt_event.is_set():
                    break
                if not chunk: continue
                output_file = f"temp_chunk_{i}_{threading.get_ident()}.wav"
                try:
                    if self.voice_clone_prompt is not None:
                        wavs, sr = self.model.generate_voice_clone(
                            text=chunk, language="German", 
                            voice_clone_prompt=self.voice_clone_prompt, 
                            instruction=instruction,
                            temperature=0.8, top_p=0.9, repetition_penalty=1.1
                        )
                    elif os.path.exists(self.ref_wav):
                        wavs, sr = self.model.generate_voice_clone(
                            text=chunk, language="German", ref_audio=self.ref_wav,
                            ref_text=self.ref_text, instruction=instruction,
                            temperature=0.8, top_p=0.9, repetition_penalty=1.1
                        )
                    else:
                        try:
                            wavs, sr = self.model.generate_custom_voice(
                                text=chunk, language="German", speaker="Claribel Dervla",
                                instruction=instruction, temperature=0.8
                            )
                        except:
                            break
                    
                    sf.write(output_file, wavs[0], sr)
                    audio_queue.put(output_file)
                except Exception:
                    audio_queue.put(None)
            audio_queue.put("DONE")

        gen_thread = threading.Thread(target=generator_worker)
        gen_thread.start()

        try:
            p = pyaudio.PyAudio()
            while True:
                if interrupt_event and interrupt_event.is_set():
                    break
                if not gen_thread.is_alive() and audio_queue.empty():
                    break

                try:
                    file_path = audio_queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                    
                if file_path == "DONE":
                    break
                if file_path is None:
                    continue
                
                wf = wave.open(file_path, 'rb')
                stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                                channels=wf.getnchannels(), rate=wf.getframerate(), output=True,
                                frames_per_buffer=2048)
                
                data = wf.readframes(1024)
                while len(data) > 0:
                    if interrupt_event and interrupt_event.is_set():
                        break
                    stream.write(data)
                    data = wf.readframes(1024)
                
                stream.stop_stream()
                stream.close()
                wf.close()
                if os.path.exists(file_path):
                    os.remove(file_path)
            p.terminate()
        except Exception:
            pass
        
        gen_thread.join(timeout=1.0)

import os
import torch
import whisper
import soundfile as sf
import pyaudio
import wave
import numpy as np
import re
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
                # Read manual text from file
                print(f"Reading manual reference text from {self.ref_txt_file}...")
                with open(self.ref_txt_file, "r") as f:
                    self.ref_text = f.read().strip()
            else:
                # Automatic transcription via Whisper
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
        """Generates audio with Qwen3-TTS and plays it. Uses background generation for speed and is interruptible."""
        if not text.strip():
            return
            
        import re
        import threading
        import queue
        
        # 1. Extract tag and clean text
        tags = re.findall(r"\[([A-Za-zäöüß]+)\]", text)
        
        # Default mood: friendly and clear
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
        # Alle Tags in spitzen Klammern komplett entfernen (auch den Inhalt!)
        display_text = re.sub(r"<[^>]+>.*?</[^>]+>", "", display_text, flags=re.DOTALL)
        display_text = re.sub(r"<[^>]+>", "", display_text).strip()
        
        # --- TEXT NORMALISIERUNG FÜR TTS ---
        # 1. Uhrzeiten umwandeln (10:08 -> 10 Uhr 08), da Doppelpunkte oft Teile überspringen lassen
        clean_text = re.sub(r"(\d{1,2}):(\d{2})", r"\1 Uhr \2", display_text)
        
        # 2. Zeilenumbrüche und Markdown-Reste entfernen
        clean_text = clean_text.replace("\n", " ").replace("###", "").replace("***", "").replace("`", "").replace("*", "").replace("_", "")
        
        # 3. Klammern entfernen (oft verwirrend für die KI)
        clean_text = clean_text.replace("(", " ").replace(")", " ")
        
        # 4. Mehrfache Leerzeichen säubern
        clean_text = re.sub(r"\s+", " ", clean_text).strip()
        
        # 2. Splitting-Logik
        MAX_CHARS = 400
        chunks = []
        if len(clean_text) <= MAX_CHARS:
            chunks = [clean_text]
        else:
            mid = len(clean_text) // 2
            best_split = -1
            for i in range(MAX_CHARS // 2):
                for pos in [mid + i, mid - i]:
                    if pos < len(clean_text) and clean_text[pos] in ".!?":
                        best_split = pos + 1
                        break
                if best_split != -1: break
            
            if best_split != -1:
                chunks = [clean_text[:best_split].strip(), clean_text[best_split:].strip()]
            else:
                space_split = clean_text.rfind(" ", 0, MAX_CHARS)
                if space_split != -1:
                    chunks = [clean_text[:space_split].strip(), clean_text[space_split:].strip()]
                else:
                    chunks = [clean_text]

        # 3. Parallelisierung: Generierung im Hintergrund, Abspielen im Vordergrund
        audio_queue = queue.Queue()
        
        def generator_worker():
            for i, chunk in enumerate(chunks):
                if interrupt_event and interrupt_event.is_set():
                    break
                if not chunk: continue
                output_file = f"temp_chunk_{i}_{threading.get_ident()}.wav"
                try:
                    if self.voice_clone_prompt is not None:
                        # Use the pre-calculated prompt for speed
                        wavs, sr = self.model.generate_voice_clone(
                            text=chunk, language="German", 
                            voice_clone_prompt=self.voice_clone_prompt, 
                            instruction=instruction,
                            temperature=0.8, top_p=0.9, repetition_penalty=1.1
                        )
                    elif os.path.exists(self.ref_wav):
                        # Fallback if prompt wasn't cached but wav exists
                        wavs, sr = self.model.generate_voice_clone(
                            text=chunk, language="German", ref_audio=self.ref_wav,
                            ref_text=self.ref_text, instruction=instruction,
                            temperature=0.8, top_p=0.9, repetition_penalty=1.1
                        )
                    else:
                        # Fallback try custom voice (if instruct model)
                        try:
                            wavs, sr = self.model.generate_custom_voice(
                                text=chunk, language="German", 
                                speaker="Claribel Dervla",
                                instruction=instruction, 
                                temperature=0.8
                            )
                        except:
                            # If no voice cloning is possible and custom voice fails, 
                            # we silently fail here or use a simpler TTS if needed.
                            # For now, just break without error spam.
                            break
                    
                    sf.write(output_file, wavs[0], sr)
                    audio_queue.put(output_file)
                except Exception:
                    # Silent failure in standard mode
                    audio_queue.put(None)
            audio_queue.put("DONE")

        # Worker starten
        gen_thread = threading.Thread(target=generator_worker)
        gen_thread.start()

        try:
            p = pyaudio.PyAudio()
            while True:
                if interrupt_event and interrupt_event.is_set():
                    break
                    
                # Safety check: if thread died without putting DONE, break to avoid hang
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
                
                # Abspielen
                wf = wave.open(file_path, 'rb')
                stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                                channels=wf.getnchannels(), rate=wf.getframerate(), output=True,
                                frames_per_buffer=2048)
                
                data = wf.readframes(1024)
                while len(data) > 0:
                    if interrupt_event and interrupt_event.is_set():
                        break
                    stream.write(data)
                    # Schnelleres Auslesen für stabilen Puffer-Fluss
                    data = wf.readframes(1024)
                
                stream.stop_stream()
                stream.close()
                wf.close()
                # Datei löschen nachdem sie abgespielt wurde
                if os.path.exists(file_path):
                    os.remove(file_path)
            p.terminate()
        except Exception as e:
            print(f"Fehler bei Audiowiedergabe: {e}")
        
        gen_thread.join()

if __name__ == "__main__":
    # Test
    engine = TTSEngine()
    engine.speak("[glücklich] Hallo, ich bin Jarvis. Dein lokaler Assistent. Wie kann ich dir heute helfen?")

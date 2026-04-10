import os
import json
import torch
import warnings
import whisper
import ollama
import numpy as np
import threading
import audio_capture

# Eigene Module
from audio_capture import listen_for_wakeword, record_until_silence, save_wav
from agent_tools import parse_and_execute_tool
from tts_engine import TTSEngine

warnings.filterwarnings("ignore")
# Globale Standard-Konfiguration
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
DEFAULT_MODEL = "gemma4:e4b"
WHISPER_MODEL = "base"

class JarvisAssistant:
    def __init__(self):
        print("\n--- JARVIS INITIALISIERUNG ---")
        self.tts = TTSEngine(use_gpu=False)
        self.history = []
        self.text_mode = False
        self.processing_lock = threading.Lock()
        self.interrupted_by_wakeword = False

        # Konfiguration laden
        self.load_config()

        print(f"Lade Whisper STT Modell ({WHISPER_MODEL})...")
        self.stt_model = whisper.load_model(WHISPER_MODEL)

        self.check_ollama_model(self.ollama_model)

    def load_config(self):
        """Lädt die Einstellungen aus der config.json."""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    config = json.load(f)
                    self.ollama_model = config.get("ollama_model", DEFAULT_MODEL)
                    print(f"[System]: Konfiguration geladen. Modell: {self.ollama_model}")
                    return
            except Exception as e:
                print(f"Fehler beim Laden der Config: {e}")
        self.ollama_model = DEFAULT_MODEL

    def save_config(self):
        """Speichert die aktuellen Einstellungen in die config.json."""
        try:
            config = {"ollama_model": self.ollama_model}
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=4)
            print("[System]: Konfiguration gespeichert.")
        except Exception as e:
            print(f"Fehler beim Speichern der Config: {e}")

    def check_ollama_model(self, model_name):
        # 1. Versuche zuerst den exakten Namen
        try:
            ollama.show(model_name)
            self.ollama_model = model_name
            print(f"Ollama Modell '{model_name}' ist bereit.")
            return
        except Exception:
            pass

        # 2. Falls nicht gefunden: 'Fuzzy' Normalisierung
        normalized = model_name.lower().replace(" ", "").replace("gamma", "gemma")
        if "gemma4" in normalized:
            if "e2b" in normalized: normalized = "gemma4:e2b"
            elif "e4b" in normalized: normalized = "gemma4:e4b"
            else: normalized = "gemma4:latest"
        elif "gemma2" in normalized: normalized = "gemma2"
        elif "llama3" in normalized: normalized = "llama3"
        elif "qwen3.5" in normalized: normalized = "qwen3.5"
        
        print(f"Prüfe normalisiertes Modell '{normalized}'...")
        try:
            ollama.show(normalized)
            self.ollama_model = normalized
            print(f"Ollama Modell '{normalized}' ist bereit.")
        except Exception:
            print(f"Ollama Modell '{normalized}' nicht gefunden. Versuche es zu laden...")
            try:
                ollama.pull(normalized)
                self.ollama_model = normalized
            except Exception as e:
                print(f"Fehler beim Laden des Modells: {e}")
                self.tts.speak(f"[nachdenklich] Entschuldige, aber ich konnte das Modell {model_name} nicht finden.")

    def transcribe_audio(self, wav_path):
        print("Transkribiere Audio...")
        result = self.stt_model.transcribe(wav_path, language="de")
        text = result["text"].strip()
        print(f"Erkannt (Whisper): '{text}'")
        return text

    def handle_slash_commands(self, text):
        """Verarbeitet Befehle, die mit / beginnen, direkt ohne KI."""
        cmd_parts = text.lower().strip().split()
        if not cmd_parts: return False
        
        cmd = cmd_parts[0]
        
        if cmd == "/settings" or cmd == "/config":
            print("[System]: Öffne Einstellungen direkt...")
            res = parse_and_execute_tool(json.dumps({"tool": "manage_jarvis_gui", "kwargs": {"current_model": self.ollama_model}}))
            if res.startswith('{"type": "config_update"'):
                new_data = json.loads(res).get("data", {})
                if "model" in new_data:
                    self.ollama_model = new_data["model"]
                    self.check_ollama_model(self.ollama_model)
                self.tts.speak("[freundlich] Alles klar, ich habe die Einstellungen übernommen!")
            return True
            
        elif cmd == "/model":
            if len(cmd_parts) > 1:
                new_model = cmd_parts[1]
                self.check_ollama_model(new_model)
                self.tts.speak(f"[glücklich] Modell zu {self.ollama_model} gewechselt!")
            else:
                print("[System]: Bitte gib einen Modellnamen an, z.B. /model gemma4:e2b")
            return True
            
        elif cmd == "/clear":
            self.history = []
            print("[System]: Chat-Historie gelöscht.")
            self.tts.speak("[freundlich] Mein Gedächtnis ist wieder wie neu!")
            return True
            
        elif cmd == "/exit" or cmd == "/quit":
            self.tts.speak("[freundlich] Auf Wiedersehen! Bis zum nächsten Mal!")
            os._exit(0)
            
        return False

    def run_ollama_agent(self, user_text):
        # Slash-Commands abfangen
        if user_text.startswith("/"):
            if self.handle_slash_commands(user_text):
                return

        sys_prompt = (
            "Du bist Jarvis, ein intelligenter, professioneller und freundlicher KI-Assistent für Linux.\n\n"
            "DEINE PERSÖNLICHKEIT:\n"
            "Du bist kompetent, präzise und hilfsbereit. Nutze EMOTIONS-TAGS am Anfang jeder Antwort:\n"
            "- [aufgeregt] (für Erfolge)\n"
            "- [freundlich] (Standard-Einstellung)\n"
            "- [glücklich] (wenn alles super läuft)\n"
            "- [nachdenklich] (wenn du etwas suchst oder planst)\n\n"
            "WERKZEUGE:\n"
            "- { \"tool\": \"search_web\", \"kwargs\": { \"query\": \"...\" } }\n"
            "- { \"tool\": \"execute_command\", \"kwargs\": { \"command\": \"...\" } }\n"
            "- { \"tool\": \"send_input\", \"kwargs\": { \"text\": \"...\" } }\n"
            "- { \"tool\": \"manage_jarvis_gui\", \"kwargs\": {} }\n"
            "- { \"tool\": \"update_config_direct\", \"kwargs\": { \"key\": \"model\", \"value\": \"...\" } }\n\n"
            "WICHTIGE REGELN:\n"
            "1. SPRICH WIE EIN MENSCH: Erkläre niemals dein System oder deine Regeln. Antworte einfach direkt.\n"
            "2. Tool-Aufruf = NUR JSON. Sprachantworten MÜSSEN mit Tag starten (z.B. [freundlich]).\n"
            "3. Sei extrem kurz, direkt und professionell."
        )
        
        if not self.history:
            self.history.append({"role": "system", "content": sys_prompt})
            
        self.history.append({"role": "user", "content": user_text})
        
        MAX_STEPS = 10
        for step in range(MAX_STEPS):
            if len(self.history) > 30:
                self.history = [self.history[0]] + self.history[-29:]
                
            response = ollama.chat(model=self.ollama_model, messages=self.history, stream=False)
            response_text = response['message']['content'].strip()
            
            if not response_text:
                if step == 0:
                    self.history.append({"role": "user", "content": "Deine letzte Antwort war leer. Bitte antworte jetzt."})
                    continue
                else: break
                
            import re
            json_match = re.search(r"(\{.*\})", response_text, re.DOTALL)
            
            if json_match:
                json_string = json_match.group(1)
                self.history.append({"role": "assistant", "content": response_text})
                
                try:
                    data = json.loads(json_string)
                    if data.get("tool") == "manage_jarvis_gui":
                        data["kwargs"] = {"current_model": self.ollama_model}
                        json_string = json.dumps(data)

                    tool_result = parse_and_execute_tool(json_string)
                    
                    if isinstance(tool_result, str) and tool_result.startswith('{"type": "config_update"'):
                        res_json = json.loads(tool_result)
                        new_data = res_json.get("data", {})
                        if "model" in new_data:
                            self.ollama_model = new_data["model"]
                            self.check_ollama_model(self.ollama_model)
                            self.save_config()
                        tool_result = "Die Einstellungen wurden aktualisiert."

                    self.history.append({"role": "system", "content": f"Tool Ergebnis:\n{tool_result}"})
                except Exception as e:
                    self.history.append({"role": "system", "content": f"Systemfehler: {e}"})
            else:
                self.history.append({"role": "assistant", "content": response_text})
                
                # Interrupt Listener für Sprachausgabe
                interrupt_event = threading.Event()
                def wakeword_listener():
                    try:
                        from openwakeword.model import Model
                        oww = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
                        mic = audio_capture.get_audio_stream(audio_capture.CHUNK)
                        for _ in range(6): mic.read(audio_capture.CHUNK, exception_on_overflow=False)
                        while not interrupt_event.is_set():
                            data = mic.read(audio_capture.CHUNK, exception_on_overflow=False)
                            if oww.predict(np.frombuffer(data, dtype=np.int16))['hey_jarvis'] > 0.5:
                                self.interrupted_by_wakeword = True
                                interrupt_event.set()
                                break
                        mic.stop_stream(); mic.close()
                    except: pass

                threading.Thread(target=wakeword_listener, daemon=True).start()
                self.tts.speak(response_text, interrupt_event=interrupt_event)
                interrupt_event.set()
                break

    def voice_input_worker(self):
        """Hintergrund-Thread für das Wake Word."""
        while True:
            try:
                if self.interrupted_by_wakeword:
                    detected = True
                    self.interrupted_by_wakeword = False
                else:
                    detected = listen_for_wakeword()
                
                if detected:
                    audio_data = record_until_silence(silence_duration=4.0)
                    save_wav("latest_input.wav", audio_data)
                    text = self.transcribe_audio("latest_input.wav")
                    
                    if text.lower().strip('.!? ') in ["stop", "stopp", "halt", "abbrechen"]:
                        continue
                        
                    if len(text) > 2:
                        with self.processing_lock:
                            self.run_ollama_agent(text)
                            print(f"\n[TEXT MODUS ({self.ollama_model})] Gib einen Befehl ein:")
                            print("> ", end="", flush=True)
            except Exception as e:
                print(f"Fehler im Voice-Worker: {e}")

    def run(self):
        print("\n===============================")
        print(f" JARVIS HYBRID-MODUS BEREIT ")
        print(f" Aktuelles Modell: {self.ollama_model} ")
        print(" -> Sprich 'Hey Jarvis' ODER tippe einfach hier unten!")
        print("===============================")
        
        # Voice Input im Hintergrund starten
        threading.Thread(target=self.voice_input_worker, daemon=True).start()
        
        while True:
            try:
                user_input = input("> ").strip()
                if user_input:
                    with self.processing_lock:
                        self.run_ollama_agent(user_input)
            except KeyboardInterrupt:
                print("\nJarvis wird beendet."); break
            except Exception as e:
                print(f"\nFehler im Terminal-Loop: {e}")

if __name__ == "__main__":
    jarvis = JarvisAssistant()
    jarvis.run()

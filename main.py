import os
import json
import torch
import warnings
import whisper
import ollama
import numpy as np
import threading
import audio_capture
import logging

# Eigene Module
from audio_capture import listen_for_wakeword, record_until_silence, save_wav
from agent_tools import parse_and_execute_tool
from tts_engine import TTSEngine

# Warnungen und Logs unterdrücken
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR) # Ollama nutzt httpx
# Globale Standard-Konfiguration
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
DEFAULT_MODEL = "gemma4:e4b"
WHISPER_MODEL = "base"

class JarvisAssistant:
    def __init__(self):
        # Standardmäßig debug, bis config geladen ist
        self.view_mode = "debug"
        self.ollama_model = None
        
        # Konfiguration laden
        self.load_config()
        
        self.log("\n--- JARVIS INITIALISIERUNG ---", "standard")

        # Whisper STT zuerst laden, um es an TTSEngine zu übergeben
        self.log(f"Lade Whisper STT Modell ({WHISPER_MODEL})...", "debug")
        self.stt_model = whisper.load_model(WHISPER_MODEL)
        
        # TTSEngine das bereits geladene Whisper Modell mitgeben (spart RAM)
        self.tts = TTSEngine(use_gpu=True, stt_model=self.stt_model)
        
        self.history = []
        self.text_mode = False
        self.processing_lock = threading.Lock()
        self.interrupted_by_wakeword = False

        # Falls kein Modell in der Config ist -> Erststart-Modus
        if not self.ollama_model:
            self.log("Erster Start erkannt. Bitte wähle ein Modell in den Einstellungen.", "standard")
            self.open_settings()

        if self.ollama_model:
            self.check_ollama_model(self.ollama_model)

    def log(self, message, level="debug"):
        """Filtert Ausgaben basierend auf dem view_mode."""
        if not message: return
        
        import re
        if self.view_mode == "standard":
            if level == "standard":
                # 1. Entferne [Emotion]-Tags
                clean_message = re.sub(r"\[[A-Za-zäöüß ]+\]", "", message)
                # 2. Entferne <Tag>...</Tag> inklusive Inhalt
                clean_message = re.sub(r"<[^>]+>.*?</[^>]+>", "", clean_message, flags=re.DOTALL)
                # 3. Entferne verbleibende einzelne <Tags>
                clean_message = re.sub(r"<[^>]+>", "", clean_message)
                
                clean_message = clean_message.strip()
                if clean_message:
                    print(clean_message)
        else:
            # Im Debug-Modus alles anzeigen
            print(message)

    def load_config(self):
        """Lädt die Einstellungen aus der config.json."""
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    config = json.load(f)
                    self.ollama_model = config.get("ollama_model")
                    self.view_mode = config.get("view_mode", "standard")
                    self.log(f"Konfiguration geladen. Modell: {self.ollama_model}, Mode: {self.view_mode}", "debug")
                    return
            except Exception as e:
                self.log(f"Fehler beim Laden der Config: {e}", "debug")
        self.ollama_model = None
        self.view_mode = "standard"

    def open_settings(self):
        """Öffnet die GUI-Einstellungen zur Modellauswahl."""
        self.log("Öffne Einstellungen...", "debug")
        res = parse_and_execute_tool(json.dumps({
            "tool": "manage_jarvis_gui", 
            "kwargs": {
                "current_model": self.ollama_model or "Wähle ein Modell...",
                "current_view_mode": self.view_mode
            }
        }))
        if res.startswith('{"type": "config_update"'):
            new_data = json.loads(res).get("data", {})
            if "model" in new_data:
                self.ollama_model = new_data["model"]
            if "view_mode" in new_data:
                self.view_mode = new_data["view_mode"]
            self.save_config()
            self.log(f"Einstellungen übernommen: {self.ollama_model} ({self.view_mode})", "standard")
        
        if not self.ollama_model:
            self.log("Kein Modell ausgewählt. Jarvis kann nicht starten.", "standard")
            os._exit(1)

    def save_config(self):
        """Speichert die aktuellen Einstellungen in die config.json."""
        try:
            config = {
                "ollama_model": self.ollama_model,
                "view_mode": self.view_mode
            }
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f, indent=4)
            self.log("Konfiguration gespeichert.", "debug")
        except Exception as e:
            self.log(f"Fehler beim Speichern der Config: {e}", "debug")

    def check_ollama_model(self, model_name):
        # 1. Versuche zuerst den exakten Namen
        try:
            ollama.show(model_name)
            self.ollama_model = model_name
            self.log(f"Ollama Modell '{model_name}' ist bereit.", "debug")
            return
        except Exception:
            pass

        # 2. Falls nicht gefunden -> GUI öffnen
        self.log(f"Modell '{model_name}' wurde nicht gefunden.", "standard")
        self.tts.speak(f"[nachdenklich] Ich konnte das Modell {model_name} nicht finden. Bitte wähle ein installiertes Modell aus.")
        self.open_settings()

    def transcribe_audio(self, wav_path):
        self.log("Transkribiere Audio...", "debug")
        result = self.stt_model.transcribe(wav_path, language="de")
        text = result["text"].strip()
        if text:
            self.log(f"Du: {text}", "standard")
        return text

    def handle_slash_commands(self, text):
        """Verarbeitet Befehle, die mit / beginnen, direkt ohne KI."""
        cmd_parts = text.lower().strip().split()
        if not cmd_parts: return False
        
        cmd = cmd_parts[0]
        
        if cmd == "/settings" or cmd == "/config":
            self.open_settings()
            return True
            
        elif cmd == "/model":
            if len(cmd_parts) > 1:
                new_model = cmd_parts[1]
                self.check_ollama_model(new_model)
                self.tts.speak(f"[glücklich] Modell zu {self.ollama_model} gewechselt!")
            else:
                self.log("Bitte gib einen Modellnamen an, z.B. /model gemma4:e2b", "standard")
            return True
            
        elif cmd == "/clear":
            self.history = []
            self.log("Chat-Historie gelöscht.", "standard")
            self.tts.speak("[freundlich] Mein Gedächtnis ist wieder wie neu!")
            return True
            
        elif cmd == "/exit" or cmd == "/quit":
            self.tts.speak("[freundlich] Auf Wiedersehen! Bis zum nächsten Mal!")
            os._exit(0)
            
        return False

    def speak_with_interrupt(self, text):
        """Spricht Text aus und hört gleichzeitig auf das Wake Word zum Unterbrechen."""
        if not text or not text.strip(): return
        
        interrupt_event = threading.Event()
        self.interrupted_by_wakeword = False

        def wakeword_listener():
            try:
                from openwakeword.model import Model
                # Fehler-Fix für AudioFeatures.__init__() - inference_framework entfernt
                try:
                    oww = Model(wakeword_models=["hey_jarvis"])
                except:
                    # Fallback auf Standard-Modell falls Version inkompatibel
                    oww = Model()
                
                mic = audio_capture.get_audio_stream(audio_capture.CHUNK)
                for _ in range(6): mic.read(audio_capture.CHUNK, exception_on_overflow=False)
                
                while not interrupt_event.is_set():
                    data = mic.read(audio_capture.CHUNK, exception_on_overflow=False)
                    preds = oww.predict(np.frombuffer(data, dtype=np.int16))
                    # Suche nach irgendeinem Wake Word mit hoher Wahrscheinlichkeit
                    if any(v > 0.5 for k, v in preds.items() if "jarvis" in k.lower() or "hey_jarvis" in k.lower()):
                        self.log("Unterbrechung durch Wake Word erkannt!", "debug")
                        self.interrupted_by_wakeword = True
                        interrupt_event.set()
                        break
                mic.stop_stream()
                mic.close()
            except Exception as e:
                self.log(f"Fehler im Interrupt-Listener: {e}", "debug")

        listener_thread = threading.Thread(target=wakeword_listener, daemon=True)
        listener_thread.start()
        
        self.tts.speak(text, interrupt_event=interrupt_event)
        interrupt_event.set()
        listener_thread.join(timeout=1.0)

    def run_ollama_agent(self, user_text):
        # Slash-Commands abfangen
        if user_text.startswith("/"):
            if self.handle_slash_commands(user_text):
                return

        sys_prompt = (
            "Du bist Jarvis, ein autonomer, hochintelligenter KI-Agent mit direktem Zugriff auf dieses Linux-System.\n\n"
            "INTERNES DENKEN:\n"
            "Nutze IMMER `<thought>...</thought>` Tags am Anfang deiner Antwort, um intern zu planen. Dieser Bereich wird dem Nutzer NICHT vorgelesen.\n\n"
            "WERKZEUGE (NUR JSON ERLAUBT):\n"
            "Deine Werkzeug-Aufrufe MÜSSEN IMMER exakt dieses JSON-Format am ENDE deiner Antwort haben:\n"
            "{ \"tool\": \"search_web\", \"kwargs\": { \"query\": \"...\" } }\n"
            "{ \"tool\": \"execute_command\", \"kwargs\": { \"command\": \"...\" } }\n\n"
            "WICHTIGE REGELN (STRENGSTENS EINHALTEN):\n"
            "1. KEIN CODE: Nutze NIEMALS `tool_code`, `python` oder `print(...)`. Werkzeuge dürfen AUSSCHLIESSLICH als JSON-Block gesendet werden.\n"
            "2. FORMAT-ZWANG: Ein Tool-Aufruf ohne JSON ist ungültig. Schreibe das JSON außerhalb von Code-Blöcken.\n"
            "3. PROAKTIVITÄT: Frage NICHT nach dem OS. Führe erst `execute_command` mit `cat /etc/os-release` aus.\n"
            "4. REIHENFOLGE: <thought> Planung </thought> [Emotion] Sprechtext { \"tool\": \"...\" }.\n"
            "5. Sei extrem kurz und professionell."
        )
        
        if not self.history:
            self.history.append({"role": "system", "content": sys_prompt})
            
        self.history.append({"role": "user", "content": user_text})
        
        MAX_STEPS = 10
        for step in range(MAX_STEPS):
            if self.interrupted_by_wakeword: break
            
            try:
                response = ollama.chat(model=self.ollama_model, messages=self.history, stream=False)
                response_text = response['message']['content'].strip()
            except Exception as e:
                self.log(f"Ollama Fehler: {e}", "debug")
                break
            
            if not response_text: break
                
            import re
            
            # 1. Denken extrahieren (unterstützt <thought> und <think> von Modellen wie DeepSeek)
            thought_match = re.search(r"<(thought|think)>(.*?)</\1>", response_text, re.DOTALL)
            if thought_match:
                thought_content = thought_match.group(2).strip()
                self.log(f"[Thinking]: {thought_content}", "debug")
                # Alle Denken-Blöcke aus der Antwort für den User entfernen
                response_text = re.sub(r"<(thought|think)>.*?</\1>", "", response_text, flags=re.DOTALL).strip()

            # 2. JSON extrahieren (Werkzeuge)
            json_match = re.search(r"(\{.*\})", response_text, re.DOTALL)
            
            if json_match:
                json_string = json_match.group(1)
                
                speech_text = response_text.replace(json_string, "").strip()
                if speech_text:
                    self.log(f"[{self.ollama_model}]: {speech_text}", "standard")
                    self.speak_with_interrupt(speech_text)
                    if self.interrupted_by_wakeword: break
                
                self.history.append({"role": "assistant", "content": response_text})
                
                try:
                    data = json.loads(json_string)
                    # Tool Execution EXTREM PROMINENT hervorheben
                    tool_name = data.get('tool')
                    tool_args = data.get('kwargs')
                    self.log(f"\n>>>> [JARVIS AKTIV]: Nutze Werkzeug '{tool_name}'", "standard")
                    self.log(f">>>> [DETAILS]: {tool_args}\n", "standard")
                    
                    tool_result = parse_and_execute_tool(json_string)
                    self.log(f">>>> [INFO]: Werkzeug-Ausführung abgeschlossen. Verarbeite Ergebnisse...\n", "standard")
                    self.history.append({"role": "system", "content": f"Tool Ergebnis:\n{tool_result}"})
                except Exception as e:
                    self.history.append({"role": "system", "content": f"Systemfehler: {e}"})
            else:
                self.history.append({"role": "assistant", "content": response_text})
                # Nur einmal loggen im Standard-Modus
                self.log(f"[{self.ollama_model}]: {response_text}", "standard")
                self.speak_with_interrupt(response_text)
                break

    def voice_input_worker(self):
        """Hintergrund-Thread für das Wake Word."""
        while True:
            try:
                if self.interrupted_by_wakeword:
                    audio_capture.play_notification()
                    detected = True
                    self.interrupted_by_wakeword = False
                else:
                    detected = listen_for_wakeword()
                
                if detected:
                    audio_data = record_until_silence(silence_duration=4.0)
                    save_wav("latest_input.wav", audio_data)
                    text = self.transcribe_audio("latest_input.wav")
                    
                    if text and text.lower().strip('.!? ') in ["stop", "stopp", "halt", "abbrechen"]:
                        self.log("Befehl abgebrochen.", "standard")
                        continue
                        
                    if text and len(text) > 2:
                        with self.processing_lock:
                            self.run_ollama_agent(text)
                            self.log(f"\n[TEXT MODUS ({self.ollama_model})] Gib einen Befehl ein:", "debug")
                            self.log("> ", "debug")
            except Exception as e:
                self.log(f"Fehler im Voice-Worker: {e}", "debug")

    def run(self):
        self.log("\n===============================", "standard")
        self.log(f" JARVIS HYBRID-MODUS BEREIT ", "standard")
        self.log(f" Aktuelles Modell: {self.ollama_model} ", "standard")
        self.log(" -> Sprich 'Hey Jarvis' ODER tippe einfach hier unten!", "standard")
        self.log("===============================", "standard")
        
        # Voice Input im Hintergrund starten
        threading.Thread(target=self.voice_input_worker, daemon=True).start()
        
        while True:
            try:
                user_input = input("> ").strip()
                if user_input:
                    with self.processing_lock:
                        self.run_ollama_agent(user_input)
            except KeyboardInterrupt:
                self.log("\nJarvis wird beendet.", "standard"); break
            except Exception as e:
                self.log(f"\nFehler im Terminal-Loop: {e}", "debug")

if __name__ == "__main__":
    jarvis = JarvisAssistant()
    jarvis.run()

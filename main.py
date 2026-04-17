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
import re
import sys

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
WHISPER_MODEL = "base"

class JarvisAssistant:
    def __init__(self):
        # Standard-Werte
        self.view_mode = "standard"
        self.ollama_model = None
        self.security_mode = True
        self.language = "de"
        self.last_detected_lang = "de"
        self.tts_type = "qwen3-tts"
        self.piper_voice = "de_DE-thorsten-high"
        self.qwen_voice = "default.wav"
        
        # Konfiguration laden
        self.load_config()
        
        self.log("\n--- JARVIS INITIALISIERUNG ---", "standard")

        # Whisper STT zuerst laden, um es an TTSEngine zu übergeben
        self.log(f"Lade Whisper STT Modell ({WHISPER_MODEL})...", "debug")
        self.stt_model = whisper.load_model(WHISPER_MODEL)
        
        # TTSEngine initialisieren
        self.init_tts()
        
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
            # Modell im Hintergrund laden (Preload)
            threading.Thread(target=lambda: ollama.generate(model=self.ollama_model, prompt="hi"), daemon=True).start()

    def init_tts(self):
        config = {
            "tts_type": self.tts_type,
            "piper_voice": self.piper_voice,
            "qwen_voice": self.qwen_voice
        }
        self.tts = TTSEngine(config=config, use_gpu=True, stt_model=self.stt_model)

    def log(self, message, level="debug"):
        """Filtert Ausgaben basierend auf dem view_mode."""
        if not message: return
        
        if self.view_mode == "standard":
            if level == "standard":
                # Säuberung für die Konsole
                clean_message = re.sub(r"\[[A-Za-zäöüß ]+\]", "", message)
                clean_message = re.sub(r"<(thought|think)>.*?</\1>", "", clean_message, flags=re.DOTALL)
                clean_message = re.sub(r"<[^>]+>", "", clean_message).strip()
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
                    self.security_mode = config.get("security_mode", True)
                    self.language = config.get("language", "de")
                    self.tts_type = config.get("tts_type", "qwen3-tts")
                    self.piper_voice = config.get("piper_voice", "de_DE-thorsten-high")
                    self.qwen_voice = config.get("qwen_voice", "default.wav")
                    return
            except Exception as e:
                self.log(f"Fehler beim Laden der Config: {e}", "debug")
        self.ollama_model = None
        self.view_mode = "standard"
        self.security_mode = True
        self.language = "de"

    def open_settings(self):
        """Öffnet die GUI-Einstellungen."""
        self.log("Öffne Einstellungen...", "debug")
        res = parse_and_execute_tool(json.dumps({
            "tool": "manage_jarvis_gui", 
            "kwargs": {
                "ollama_model": self.ollama_model or "Wähle ein Modell...",
                "view_mode": self.view_mode,
                "security_mode": self.security_mode,
                "language": self.language,
                "tts_type": self.tts_type,
                "piper_voice": self.piper_voice,
                "qwen_voice": self.qwen_voice
            }
        }))
        if res.startswith('{"type": "config_update"'):
            new_data = json.loads(res).get("data", {})
            self.ollama_model = new_data.get("ollama_model", self.ollama_model)
            self.view_mode = new_data.get("view_mode", self.view_mode)
            self.security_mode = new_data.get("security_mode", self.security_mode)
            self.language = new_data.get("language", self.language)
            
            old_tts_type = self.tts_type
            old_piper_voice = self.piper_voice
            old_qwen_voice = self.qwen_voice
            
            self.tts_type = new_data.get("tts_type", self.tts_type)
            self.piper_voice = new_data.get("piper_voice", self.piper_voice)
            self.qwen_voice = new_data.get("qwen_voice", self.qwen_voice)
            
            self.save_config()
            
            # Re-init TTS if settings changed
            if (old_tts_type != self.tts_type or 
                old_piper_voice != self.piper_voice or 
                old_qwen_voice != self.qwen_voice):
                self.log("TTS-Einstellungen geändert. Initialisiere neu...", "standard")
                if hasattr(self, 'tts'):
                    self.tts.unload_models()
                self.init_tts()
            
            self.log(f"Einstellungen übernommen.", "standard")
        
        if not self.ollama_model:
            self.log("Kein Modell ausgewählt. Jarvis kann nicht starten.", "standard")
            os._exit(1)

    def save_config(self):
        """Speichert die aktuellen Einstellungen in die config.json."""
        try:
            config = {
                "ollama_model": self.ollama_model,
                "view_mode": self.view_mode,
                "security_mode": self.security_mode,
                "language": self.language,
                "tts_type": self.tts_type,
                "piper_voice": self.piper_voice,
                "qwen_voice": self.qwen_voice
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
        # Sprache auf None setzen für automatische Erkennung (DE/EN Fokus)
        result = self.stt_model.transcribe(wav_path)
        text = result["text"].strip()
        if text:
            detected_lang = result.get("language", "unknown")
            if detected_lang != "unknown":
                self.last_detected_lang = detected_lang
            self.log(f"Du ({detected_lang}): {text}", "standard")
        return text

    def handle_slash_commands(self, text):
        """Verarbeitet Befehle, die mit / beginnen, direkt ohne KI."""
        clean_text = text.strip().lower()
        if not clean_text.startswith("/"):
            return False
            
        cmd_parts = clean_text.split()
        if not cmd_parts: return False
        
        cmd = cmd_parts[0]
        
        # Aliases für Settings
        if cmd in ["/settings", "/config", "/einstellungen", "/setup"]:
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
                try:
                    oww = Model(wakeword_models=["hey_jarvis"])
                except:
                    oww = Model()
                
                mic = audio_capture.get_audio_stream(audio_capture.CHUNK)
                for _ in range(6): mic.read(audio_capture.CHUNK, exception_on_overflow=False)
                
                while not interrupt_event.is_set():
                    data = mic.read(audio_capture.CHUNK, exception_on_overflow=False)
                    preds = oww.predict(np.frombuffer(data, dtype=np.int16))
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
        # Slash-Commands SOFORT abfangen
        if user_text.strip().startswith("/"):
            if self.handle_slash_commands(user_text):
                return
            else:
                self.log(f"Unbekannter Befehl: {user_text}", "standard")
                return

        # Sprach-Logik für den System-Prompt
        target_lang = self.language
        if target_lang == "auto":
            target_lang = self.last_detected_lang

        if target_lang == "en":
            security_info = ""
            if self.security_mode:
                security_info = "\nSECURITY: ENABLED. execute_command is BLOCKED."
            else:
                security_info = "\nSECURITY: DISABLED. FULL SYSTEM ACCESS."

            sys_prompt = (
                "You are Jarvis, a highly autonomous AI agent.\n"
                f"{security_info}\n\n"
                "INTERNAL THOUGHTS:\n"
                "CRITICAL: You MUST put ALL your planning and reasoning inside `<thought>...</thought>` or `<think>...</think>` tags. The user does not see this.\n"
                "Write your spoken text AFTER the thought block.\n\n"
                "TOOLS:\n"
                "To use a tool, your response MUST end with exactly ONE JSON block. Do NOT use markdown formatting (```json) for it.\n"
                "{ \"tool\": \"search_web\", \"kwargs\": { \"query\": \"...\" } }\n"
                "{ \"tool\": \"execute_command\", \"kwargs\": { \"command\": \"...\" } }\n\n"
                "EXAMPLE:\n"
                "<thought>I need to check the OS first.</thought>\n"
                "Checking system information.\n"
                "{ \"tool\": \"execute_command\", \"kwargs\": { \"command\": \"cat /etc/os-release\" } }\n\n"
                "RULES:\n"
                "1. STEP BY STEP: Only execute ONE tool at a time. Wait for the result before planning the next step.\n"
                "2. NO GUESSING: Never guess the OS. Always use `cat /etc/os-release` first if you need to install something.\n"
                "3. BE CONCISE: Max 1 sentence outside thought tags. Just report status.\n"
                "4. Respond in ENGLISH."
            )
        else:
            # Standard: Deutsch
            security_info = ""
            if self.security_mode:
                security_info = "\nSICHERHEIT: AKTIVIERT. execute_command ist BLOCKIERT."
            else:
                security_info = "\nSICHERHEIT: DEAKTIVIERT. VOLLZUGRIFF."

            sys_prompt = (
                "Du bist Jarvis, ein hochgradig autonomer KI-Agent.\n"
                f"{security_info}\n\n"
                "INTERNES DENKEN:\n"
                "WICHTIG: Nutze für deine internen Überlegungen IMMER `<thought>...</thought>` oder `<think>...</think>`. Der Nutzer sieht diesen Text NICHT.\n"
                "Schreibe deinen sichtbaren Antworttext IMMER nach dem Gedanken-Block.\n\n"
                "WERKZEUGE:\n"
                "Um ein Werkzeug auszuführen, MUSS deine Antwort am Ende exakt EINEN JSON-Block enthalten. Nutze KEINE Markdown-Formatierung (```json) dafür.\n"
                "{ \"tool\": \"search_web\", \"kwargs\": { \"query\": \"...\" } }\n"
                "{ \"tool\": \"execute_command\", \"kwargs\": { \"command\": \"...\" } }\n\n"
                "BEISPIEL:\n"
                "<thought>Ich muss zuerst das OS prüfen.</thought>\n"
                "Ich prüfe das System.\n"
                "{ \"tool\": \"execute_command\", \"kwargs\": { \"command\": \"cat /etc/os-release\" } }\n\n"
                "REGELN:\n"
                "1. SCHRITT FÜR SCHRITT: Führe immer nur EIN Werkzeug aus und warte auf das Ergebnis, bevor du den nächsten Schritt planst.\n"
                "2. NICHT RATEN: Rate niemals das Betriebssystem. Nutze immer zuerst `cat /etc/os-release`.\n"
                "3. KURZ FASSEN: Außerhalb der Gedanken-Tags nur maximal einen kurzen Satz schreiben.\n"
                "4. Antworte auf DEUTSCH."
            )
        
        if not self.history:
            self.history.append({"role": "system", "content": sys_prompt})
        else:
            self.history[0] = {"role": "system", "content": sys_prompt}
            
        self.history.append({"role": "user", "content": user_text})
        
        MAX_STEPS = 10
        for step in range(MAX_STEPS):
            if self.interrupted_by_wakeword: break
            
            try:
                full_response = ""
                # Header nur im Debug anzeigen
                if self.view_mode == "debug":
                    print(f"[{self.ollama_model}]: ", end="", flush=True)
                
                # STREAMING
                for chunk in ollama.chat(model=self.ollama_model, messages=self.history, stream=True):
                    if self.interrupted_by_wakeword: break
                    token = chunk['message']['content']
                    full_response += token
                    if self.view_mode == "debug":
                        print(token, end="", flush=True)
                
                if self.view_mode == "debug": print()
                response_text = full_response.strip()
                
            except Exception as e:
                self.log(f"Ollama Fehler: {e}", "standard")
                break
            
            if not response_text: break
            
            # Markdown JSON Formatierung entfernen, falls das Modell sie trotzdem nutzt
            clean_response = re.sub(r"```json\s*", "", response_text, flags=re.IGNORECASE)
            clean_response = re.sub(r"```\s*$", "", clean_response).strip()
            
            # JSON extrahieren (nicht so gierig)
            json_match = re.search(r"(\{\s*\"tool\":.*?\})", clean_response, re.DOTALL)
            
            # Sprechtext säubern (Alle Varianten von Gedanken entfernen)
            speech_text = re.sub(r"<(thought|think)>.*?</\1>", "", clean_response, flags=re.DOTALL | re.IGNORECASE).strip()
            speech_text = re.sub(r"<(thought|think)>.*", "", speech_text, flags=re.DOTALL | re.IGNORECASE).strip() # Offene Tags fangen
            speech_text = re.sub(r"</(thought|think)>", "", speech_text, flags=re.IGNORECASE).strip() # Einzelne schließende Tags fangen
            
            if json_match:
                speech_text = speech_text.replace(json_match.group(1), "").strip()
            speech_text = re.sub(r"<[^>]+>", "", speech_text).strip()

            # Nur loggen/sprechen, wenn es Text gibt
            if speech_text:
                self.log(f"[{self.ollama_model}]: {speech_text}", "standard")
                self.speak_with_interrupt(speech_text)
                if self.interrupted_by_wakeword: break
            
            self.history.append({"role": "assistant", "content": response_text})
            
            if json_match:
                json_string = json_match.group(1)
                try:
                    data = json.loads(json_string)
                    tool_name = data.get('tool')
                    if tool_name:
                        print(f">>>> [JARVIS]: Nutze '{tool_name}'...")
                        tool_result = parse_and_execute_tool(json_string)
                        self.history.append({"role": "system", "content": f"Tool Ergebnis:\n{tool_result}"})
                    else:
                        break
                except:
                    break
            else:
                break

    def voice_input_worker(self):
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
            except Exception as e:
                self.log(f"Fehler im Voice-Worker: {e}", "debug")

    def run(self):
        self.log("\n===============================", "standard")
        self.log(f" JARVIS HYBRID-MODUS BEREIT ", "standard")
        self.log(f" Aktuelles Modell: {self.ollama_model} ", "standard")
        self.log(" -> Sprich 'Hey Jarvis' ODER tippe einfach hier unten!", "standard")
        self.log("===============================", "standard")
        
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

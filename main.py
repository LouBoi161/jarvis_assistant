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
import time

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
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latest_session.log")
WHISPER_MODEL = "base"

class JarvisAssistant:
    def __init__(self):
        # Initialisiere Session-Log (überschreiben)
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write(f"--- JARVIS SESSION LOG START ({time.strftime('%Y-%m-%d %H:%M:%S')}) ---\n")

        # Standard-Werte
        self.view_mode = "standard"
        self.ollama_model = None
        self.security_mode = True
        self.language = "de"
        self.last_detected_lang = "de"
        self.tts_type = "qwen3-tts"
        self.piper_voice = "de_DE-thorsten-high"
        self.qwen_voice = "default.wav"
        
        # GUI Callback
        self.on_status_change = None # Callback function(status_str)
        
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
        """Filtert Ausgaben basierend auf dem view_mode und schreibt ins Session-Log."""
        if not message: return
        
        # In Log-Datei schreiben
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                ts = time.strftime("%H:%M:%S")
                f.write(f"[{ts}] [{level.upper()}] {message}\n")
        except: pass

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

    def set_status(self, status):
        if self.on_status_change:
            self.on_status_change(status)

    def run_ollama_agent(self, user_text):
        self.set_status("thinking")
        # Slash-Commands SOFORT abfangen
        if user_text.strip().startswith("/"):
            if self.handle_slash_commands(user_text):
                self.set_status("idle")
                return
            else:
                self.log(f"Unbekannter Befehl: {user_text}", "standard")
                self.set_status("idle")
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
                "You are Jarvis, a highly autonomous AI agent running on Linux.\n"
                f"{security_info}\n\n"
                "CRITICAL WORKFLOW:\n"
                "1. IDENTIFY: Before any installation or system change, ALWAYS call `get_system_info` first.\n"
                "2. RESEARCH: Use `search_web` to find the OFFICIAL and BEST installation method for the detected distribution (e.g., check AUR for Arch/CachyOS).\n"
                "3. PREFER NATIVE: Use package managers (pacman, yay, apt) over 'curl | bash' scripts.\n\n"
                "INTERNAL THOUGHTS:\n"
                "Put ALL planning, technical analysis, and log interpretation inside `<thought>...</thought>` tags.\n\n"
                "TOOLS:\n"
                "Output exactly ONE JSON block at the end. No markdown.\n"
                "{ \"tool\": \"get_system_info\", \"kwargs\": {} }\n"
                "{ \"tool\": \"search_web\", \"kwargs\": { \"query\": \"...\" } }\n"
                "{ \"tool\": \"execute_command\", \"kwargs\": { \"command\": \"...\" } }\n\n"
                "RULES:\n"
                "1. BREVITY: Max 1-2 short sentences for speech.\n"
                "2. NO TECH JUNK: Do not read logs or paths.\n"
                "3. STEP BY STEP: One tool call per turn."
            )
        else:
            # Standard: Deutsch
            security_info = ""
            if self.security_mode:
                security_info = "\nSICHERHEIT: AKTIVIERT. execute_command ist BLOCKIERT."
            else:
                security_info = "\nSICHERHEIT: DEAKTIVIERT. VOLLZUGRIFF."

            sys_prompt = (
                "Du bist Jarvis, ein hochgradig autonomer KI-Agent auf Linux.\n"
                f"{security_info}\n\n"
                "WICHTIGER WORKFLOW:\n"
                "1. IDENTIFIZIEREN: Bevor du etwas installierst oder änderst, nutze IMMER zuerst `get_system_info`.\n"
                "2. RECHERCHE: Nutze `search_web`, um die OFFIZIELLE und BESTE Methode für die erkannte Distribution zu finden (z.B. AUR/yay bei Arch/CachyOS).\n"
                "3. NATIVE TOOLS: Bevorzuge Paketmanager (pacman, yay, apt) gegenüber 'curl | bash' Skripten.\n\n"
                "INTERNES DENKEN:\n"
                "Schreibe ALLE Planungen und technischen Analysen AUSSCHLIESSLICH in `<thought>...</thought>` Tags.\n\n"
                "WERKZEUGE:\n"
                "Gib am Ende exakt EINEN JSON-Block aus. Kein Markdown.\n"
                "{ \"tool\": \"get_system_info\", \"kwargs\": {} }\n"
                "{ \"tool\": \"search_web\", \"kwargs\": { \"query\": \"...\" } }\n"
                "{ \"tool\": \"execute_command\", \"kwargs\": { \"command\": \"...\" } }\n\n"
                "REGELN:\n"
                "1. KÜRZE: Max. 1-2 kurze Sätze für die Sprachausgabe.\n"
                "2. KEIN TECHNIK-MÜLL: Keine Pfade oder Logs vorlesen.\n"
                "3. SCHRITT FÜR SCHRITT: Nur ein Tool-Aufruf pro Antwort."
            )
        
        if not self.history:
            self.history.append({"role": "system", "content": sys_prompt})
        else:
            self.history[0] = {"role": "system", "content": sys_prompt}
            
        self.history.append({"role": "user", "content": user_text})
        
        spoken_history = [] # Verhindert doppeltes Vorlesen identischer Sätze in einer Kette
        
        MAX_STEPS = 10
        for step in range(MAX_STEPS):
            if self.interrupted_by_wakeword: break
            
            try:
                full_response = ""
                if self.view_mode == "debug":
                    print(f"[{self.ollama_model}]: ", end="", flush=True)
                
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
            
            # Markdown JSON Formatierung entfernen
            clean_response = re.sub(r"```json\s*", "", response_text, flags=re.IGNORECASE)
            clean_response = re.sub(r"```\s*$", "", clean_response).strip()
            
            # Robustes JSON-Matching
            json_string = None
            first_brace = clean_response.find('{')
            last_brace = clean_response.rfind('}')
            
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_string = clean_response[first_brace:last_brace+1]
                speech_source = clean_response[:first_brace].strip()
            else:
                speech_source = clean_response

            # Gedanken-Tags entfernen (robuster)
            speech_text = response_text
            if json_string:
                # Wenn JSON vorhanden ist, nehmen wir nur den Teil DAVOR für die Sprache
                speech_text = response_text[:response_text.find(json_string)].strip()
            
            # Alle Arten von Gedanken-Tags und deren Inhalt entfernen
            speech_text = re.sub(r"<(thought|think)>.*?</\1>", "", speech_text, flags=re.DOTALL | re.IGNORECASE).strip()
            speech_text = re.sub(r"<(thought|think)>.*", "", speech_text, flags=re.DOTALL | re.IGNORECASE).strip()
            speech_text = re.sub(r".*?</(thought|think)>", "", speech_text, flags=re.DOTALL | re.IGNORECASE).strip()
            
            # Markdown-Überbleibsel entfernen
            speech_text = re.sub(r"```[a-z]*", "", speech_text, flags=re.IGNORECASE)
            speech_text = re.sub(r"```", "", speech_text)
            
            # Letzte Aufräumung von HTML-ähnlichen Tags
            speech_text = re.sub(r"<[^>]+>", "", speech_text).strip()

            # Nur vorlesen, wenn es Text gibt, Buchstaben enthält und noch NICHT gesagt wurde
            if speech_text and re.search(r'[a-zA-ZäöüßÄÖÜ]', speech_text):
                # Wenn der Text nur aus einer Frage besteht, ob Hilfe benötigt wird (Halluzination nach Logs), ignorieren
                hallucination_patterns = [
                    "Was möchtest du wissen?",
                    "keine spezifische Frage",
                    "Um dir helfen zu können",
                    "brauche ich mehr Kontext",
                    "Bitte stelle deine Frage"
                ]
                is_hallucination = any(p.lower() in speech_text.lower() for p in hallucination_patterns)
                
                if speech_text not in spoken_history and not is_hallucination:
                    self.set_status("speaking")
                    self.log(f"[{self.ollama_model}]: {speech_text}", "standard")
                    self.speak_with_interrupt(speech_text)
                    spoken_history.append(speech_text)
                    self.set_status("thinking")
                    if self.interrupted_by_wakeword: break
            
            self.history.append({"role": "assistant", "content": response_text})
            
            if json_string:
                try:
                    data = None
                    try:
                        data = json.loads(json_string)
                    except:
                        temp_json = json_string
                        while temp_json.rfind('}') != -1:
                            temp_json = temp_json[:temp_json.rfind('}')+1]
                            try:
                                data = json.loads(temp_json)
                                break
                            except:
                                temp_json = temp_json[:-1]
                    
                    if data and data.get('tool'):
                        tool_name = data.get('tool')
                        tool_kwargs = data.get('kwargs', {})
                        self.log(f"TOOL CALL: {tool_name} ({tool_kwargs})", "debug")
                        print(f">>>> [JARVIS]: Nutze '{tool_name}'...")
                        tool_result = parse_and_execute_tool(json.dumps(data))
                        
                        # Fallback falls Tool-Result leer ist
                        if not tool_result: tool_result = "Keine Rückgabe vom Tool."
                        self.log(f"TOOL RESULT ({tool_name}): {tool_result}", "debug")
                        
                        # Output kürzen, falls zu lang (max 3000 Zeichen für LLM Kontext)
                        if tool_result and len(tool_result) > 3000:
                            tool_result = tool_result[:1500] + "\n... [Output von Jarvis gekürzt] ...\n" + tool_result[-1500:]
                        
                        # Kontext verstärken: Wir sind noch in einer Aufgabe!
                        context_msg = f"TOOL_RESULT ({tool_name}):\n{tool_result}\n\nIMPORTANT: You are an autonomous agent. Analyze this result and CONTINUE with the user's original request until finished. If finished, say 'Task complete'."
                        self.history.append({"role": "system", "content": context_msg})
                    else:
                        break
                except:
                    break
            else:
                break
        self.set_status("idle")

    def update_config(self, new_data):
        """Aktualisiert die laufende Konfiguration und speichert sie."""
        self.ollama_model = new_data.get("ollama_model", self.ollama_model)
        self.view_mode = new_data.get("view_mode", self.view_mode)
        self.security_mode = new_data.get("security_mode", self.security_mode)
        self.language = new_data.get("language", self.language)
        self.tts_type = new_data.get("tts_type", self.tts_type)
        self.piper_voice = new_data.get("piper_voice", self.piper_voice)
        self.qwen_voice = new_data.get("qwen_voice", self.qwen_voice)
        
        # TTS Engine neu initialisieren falls nötig
        self.init_tts()
        self.save_config()

    def run_voice_only(self):
        """Version von run(), die nicht das Terminal blockiert."""
        self.log("\n--- JARVIS VOICE-ONLY MODUS AKTIV ---", "standard")
        self.voice_input_worker() # Dies ist bereits ein Loop

    def voice_input_worker(self):
        while True:
            try:
                self.set_status("idle")
                if self.interrupted_by_wakeword:
                    audio_capture.play_notification()
                    detected = True
                    self.interrupted_by_wakeword = False
                else:
                    detected = listen_for_wakeword()
                
                if detected:
                    self.set_status("listening")
                    audio_data = record_until_silence(silence_duration=4.0)
                    self.set_status("thinking")
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

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
logging.getLogger("httpx").setLevel(logging.ERROR)

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
        self.on_status_change = None 
        
        # Konfiguration laden
        self.load_config()
        
        self.log("\n--- JARVIS INITIALISIERUNG ---", "standard")
        self.log(f"Lade Whisper STT Modell ({WHISPER_MODEL})...", "debug")
        self.stt_model = whisper.load_model(WHISPER_MODEL)
        
        # TTSEngine initialisieren
        self.init_tts()
        
        self.history = []
        self.text_mode = False
        self.processing_lock = threading.Lock()
        self.interrupted_by_wakeword = False

        if self.ollama_model:
            self.check_ollama_model(self.ollama_model)
            threading.Thread(target=lambda: ollama.generate(model=self.ollama_model, prompt="hi"), daemon=True).start()

    def init_tts(self):
        config = {"tts_type": self.tts_type, "piper_voice": self.piper_voice, "qwen_voice": self.qwen_voice}
        self.tts = TTSEngine(config=config, use_gpu=True, stt_model=self.stt_model)

    def log(self, message, level="debug"):
        if not message: return
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                ts = time.strftime("%H:%M:%S")
                f.write(f"[{ts}] [{level.upper()}] {message.strip()}\n")
                f.flush()
        except: pass

        if level == "standard" and self.on_status_change:
            # GUI wird über custom_log in gui.py informiert, hier nur Konsolen-Fallback
            if self.view_mode == "debug": print(f"[{level}] {message}")

    def load_config(self):
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
            except: pass

    def save_config(self):
        config = {"ollama_model": self.ollama_model, "view_mode": self.view_mode, "security_mode": self.security_mode, "language": self.language, "tts_type": self.tts_type, "piper_voice": self.piper_voice, "qwen_voice": self.qwen_voice}
        try:
            with open(CONFIG_FILE, "w") as f: json.dump(config, f, indent=4)
        except: pass

    def set_status(self, status):
        if self.on_status_change: self.on_status_change(status)

    def check_ollama_model(self, model_name):
        try:
            ollama.show(model_name)
            self.ollama_model = model_name
            self.log(f"Ollama Modell '{model_name}' ist bereit.", "debug")
        except:
            self.log(f"Modell '{model_name}' wurde nicht gefunden.", "standard")

    def transcribe_audio(self, wav_path):
        result = self.stt_model.transcribe(wav_path)
        text = result["text"].strip()
        if text:
            detected_lang = result.get("language", "de")
            self.last_detected_lang = detected_lang
            self.log(f"Du ({detected_lang}): {text}", "standard")
        return text

    def speak_with_interrupt(self, text):
        interrupt_event = threading.Event()
        def wakeword_listener():
            try:
                from openwakeword.model import Model
                oww = Model(wakeword_models=["hey_jarvis"])
                mic = audio_capture.get_audio_stream(audio_capture.CHUNK)
                while not interrupt_event.is_set():
                    data = mic.read(audio_capture.CHUNK, exception_on_overflow=False)
                    preds = oww.predict(np.frombuffer(data, dtype=np.int16))
                    if any(v > 0.5 for k, v in preds.items() if "jarvis" in k.lower()):
                        self.interrupted_by_wakeword = True; interrupt_event.set(); break
                mic.stop_stream(); mic.close()
            except: pass
        threading.Thread(target=wakeword_listener, daemon=True).start()
        self.tts.speak(text, interrupt_event=interrupt_event)
        interrupt_event.set()

    def run_ollama_agent(self, user_text):
        self.log(f"USER INPUT: {user_text}", "debug")
        self.set_status("thinking")
        
        target_lang = self.language if self.language != "auto" else self.last_detected_lang
        sec_info = "SECURITY: ENABLED." if self.security_mode else "SECURITY: DISABLED. FULL ACCESS."
        
        sys_prompt = (
            f"You are Jarvis, an autonomous AI on Linux. {sec_info}\n"
            "WORKFLOW: 1. get_system_info (if new session), 2. search_web (if unsure), 3. execute_command (native tools preferred).\n"
            "RULES: spoken responses max 2 sentences. Use <thought> tags for all analysis. Always respond in " + ("ENGLISH" if target_lang == "en" else "GERMAN") + "."
        )

        if not self.history: self.history.append({"role": "system", "content": sys_prompt})
        else: self.history[0] = {"role": "system", "content": sys_prompt}
        
        self.history.append({"role": "user", "content": user_text})
        spoken_history = []
        
        for step in range(10):
            if self.interrupted_by_wakeword: break
            try:
                full_response = ""
                for chunk in ollama.chat(model=self.ollama_model, messages=self.history, stream=True):
                    if self.interrupted_by_wakeword: break
                    full_response += chunk['message']['content']
                response_text = full_response.strip()
            except Exception as e:
                self.log(f"Ollama Fehler: {e}", "standard"); break

            if not response_text: break
            
            json_string = None
            f_brace = response_text.find('{'); l_brace = response_text.rfind('}')
            if f_brace != -1 and l_brace != -1 and l_brace > f_brace:
                json_string = response_text[f_brace:l_brace+1]
                speech_text = response_text[:f_brace].strip()
            else: speech_text = response_text

            # Filter thoughts
            speech_text = re.sub(r"<(thought|think)>.*?</\1>", "", speech_text, flags=re.DOTALL | re.IGNORECASE).strip()
            speech_text = re.sub(r"<(thought|think)>.*", "", speech_text, flags=re.DOTALL | re.IGNORECASE).strip()
            speech_text = re.sub(r"<[^>]+>", "", speech_text).strip()

            if speech_text and re.search(r'[a-zA-ZäöüßÄÖÜ]', speech_text):
                if speech_text not in spoken_history and not any(p in speech_text for p in ["Was möchtest du wissen", "Kontext"]):
                    self.set_status("speaking")
                    self.log(f"[{self.ollama_model}]: {speech_text}", "standard")
                    self.speak_with_interrupt(speech_text)
                    spoken_history.append(speech_text)
                    self.set_status("thinking")
            
            self.history.append({"role": "assistant", "content": response_text})
            
            if json_string:
                try:
                    data = json.loads(json_string)
                    if data.get('tool'):
                        tool_name = data.get('tool')
                        self.log(f"TOOL CALL: {tool_name}", "debug")
                        tool_result = parse_and_execute_tool(json.dumps(data))
                        if not tool_result: tool_result = "Tool executed successfully."
                        self.log(f"TOOL RESULT: {tool_result[:100]}...", "debug")
                        
                        if len(tool_result) > 3000: tool_result = tool_result[:1500] + "...[cut]..." + tool_result[-1500:]
                        self.history.append({"role": "system", "content": f"TOOL_RESULT: {tool_result}\nCONTINUE until task is done."})
                    else: break
                except: break
            else: break
        
        self.set_status("idle")

    def update_config(self, new_data):
        self.ollama_model = new_data.get("ollama_model", self.ollama_model)
        self.security_mode = new_data.get("security_mode", self.security_mode)
        self.language = new_data.get("language", self.language)
        self.tts_type = new_data.get("tts_type", self.tts_type)
        self.piper_voice = new_data.get("piper_voice", self.piper_voice)
        self.qwen_voice = new_data.get("qwen_voice", self.qwen_voice)
        self.init_tts(); self.save_config()

    def run_voice_only(self):
        self.log("--- JARVIS VOICE-ONLY MODUS AKTIV ---", "standard")
        self.voice_input_worker()

    def voice_input_worker(self):
        while True:
            try:
                self.set_status("idle")
                if self.interrupted_by_wakeword:
                    audio_capture.play_notification(); detected = True; self.interrupted_by_wakeword = False
                else: detected = listen_for_wakeword()
                
                if detected:
                    self.set_status("listening")
                    audio_data = record_until_silence(silence_duration=4.0)
                    self.set_status("thinking")
                    save_wav("latest_input.wav", audio_data)
                    text = self.transcribe_audio("latest_input.wav")
                    if text and len(text) > 2:
                        with self.processing_lock: self.run_ollama_agent(text)
            except Exception as e: self.log(f"Fehler im Voice-Worker: {e}", "debug")

    def run(self):
        threading.Thread(target=self.voice_input_worker, daemon=True).start()
        while True:
            try:
                user_input = input("> ").strip()
                if user_input:
                    with self.processing_lock: self.run_ollama_agent(user_input)
            except KeyboardInterrupt: break

if __name__ == "__main__":
    jarvis = JarvisAssistant(); jarvis.run()

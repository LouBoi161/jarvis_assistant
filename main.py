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

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.ERROR)

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latest_session.log")
WHISPER_MODEL = "base"

class JarvisAssistant:
    def __init__(self):
        with open(LOG_FILE, "w", encoding="utf-8") as f:
            f.write(f"--- JARVIS SESSION START ({time.ctime()}) ---\n")

        self.view_mode = "standard"
        self.ollama_model = "gemma4:e4b"
        self.security_mode = True
        self.language = "de"
        self.last_detected_lang = "de"
        self.tts_type = "qwen3-tts"
        self.piper_voice = "de_DE-thorsten-high"
        self.qwen_voice = "default.wav"
        self.on_status_change = None 
        
        self.load_config()
        self.log("--- JARVIS INITIALISIERUNG ---", "standard")
        self.stt_model = whisper.load_model(WHISPER_MODEL, device="cpu")
        self.init_tts()
        
        self.history = []
        self.processing_lock = threading.Lock()
        self.interrupted_by_wakeword = False

    def init_tts(self):
        config = {"tts_type": self.tts_type, "piper_voice": self.piper_voice, "qwen_voice": self.qwen_voice}
        self.tts = TTSEngine(config=config, use_gpu=True, stt_model=self.stt_model)

    def log(self, message, level="debug"):
        if not message: return
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] [{level.upper()}] {message.strip()}\n"
        if level == "standard": print(f"> {message}")
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(entry); f.flush()
        except: pass

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    config = json.load(f)
                    for k, v in config.items(): setattr(self, k, v)
            except: pass

    def save_config(self):
        config = {attr: getattr(self, attr) for attr in ["ollama_model", "view_mode", "security_mode", "language", "tts_type", "piper_voice", "qwen_voice"]}
        with open(CONFIG_FILE, "w") as f: json.dump(config, f, indent=4)

    def set_status(self, status):
        if self.on_status_change: self.on_status_change(status)

    def check_ollama_model(self, model_name):
        try:
            ollama.show(model_name)
            self.ollama_model = model_name
        except:
            self.log(f"Modell '{model_name}' fehlt.", "standard")

    def transcribe_audio(self, wav_path):
        try:
            result = self.stt_model.transcribe(wav_path)
            text = result["text"].strip()
            if text:
                self.last_detected_lang = result.get("language", "de")
                self.log(f"Du: {text}", "standard")
            return text
        except: return ""

    def run_ollama_agent(self, user_text):
        self.log(f"USER INPUT: {user_text}", "debug")
        self.set_status("thinking")
        
        lang = self.language if self.language != "auto" else self.last_detected_lang
        prompt = (f"You are Jarvis, a highly autonomous AI agent on Linux. Security: {self.security_mode}.\n"
                  "CRITICAL RULES: You MUST output exactly ONE JSON block at the VERY END of your response to use a tool. Spoken text max 2 sentences. Always pro-active.\n"
                  "AVAILABLE TOOLS:\n"
                  "{ \"tool\": \"get_system_info\", \"kwargs\": {} }\n"
                  "{ \"tool\": \"search_web\", \"kwargs\": { \"query\": \"...\" } }\n"
                  "{ \"tool\": \"execute_command\", \"kwargs\": { \"command\": \"...\" } }\n"
                  "{ \"tool\": \"write_file\", \"kwargs\": { \"file_path\": \"...\", \"content\": \"...\" } }\n"
                  "{ \"tool\": \"take_screenshot\", \"kwargs\": {} }\n"
                  "Lang: " + ("EN" if lang == "en" else "DE"))

        if not self.history: self.history.append({"role": "system", "content": prompt})
        else: self.history[0] = {"role": "system", "content": prompt}

        self.history.append({"role": "user", "content": user_text})
        spoken = []
        for _ in range(15):
            if self.interrupted_by_wakeword: break
            try:
                resp = ""
                for chunk in ollama.chat(model=self.ollama_model, messages=self.history, stream=True):
                    if self.interrupted_by_wakeword: break
                    resp += chunk['message']['content']
                response_text = resp.strip()
            except Exception as e:
                self.log(f"OLLAMA FEHLER: {e}", "standard")
                break
            if not response_text: break
            
            # JSON extrahieren
            json_str = None
            f_b = response_text.find('{'); l_b = response_text.rfind('}')
            if f_b != -1 and l_b != -1 and l_b > f_b: json_str = response_text[f_b:l_b+1]

            # Text säubern für Sprache
            speech = re.sub(r"```json\s*", "", response_text, flags=re.I)
            speech = re.sub(r"```\s*", "", speech).strip()
            speech = re.sub(r"<(thought|think)>.*?</\1>", "", speech, flags=re.S | re.I)
            speech = re.sub(r"<(thought|think)>.*", "", speech, flags=re.S | re.I)
            if json_str: speech = speech.replace(json_str, "")
            speech = re.sub(r"<[^>]+>", "", speech).strip()

            if speech and re.search(r'[a-zA-Zäöüß]', speech) and speech not in spoken:
                self.set_status("speaking")
                self.log(f"[Jarvis]: {speech}", "standard")
                self.tts.speak(speech)
                spoken.append(speech)
                self.set_status("thinking")
            
            self.history.append({"role": "assistant", "content": response_text})
            
            if json_str:
                try:
                    data = json.loads(json_str)
                    if data.get('tool'):
                        tool_name = data.get('tool'); tool_kwargs = data.get('kwargs', {})
                        
                        # Detailreiche Statusmeldung für GUI
                        if tool_name == "search_web": msg = f"TOOL: 🔍 Suche nach: '{tool_kwargs.get('query', '...')}'"
                        elif tool_name == "execute_command": msg = f"TOOL: 💻 Führe aus: `{tool_kwargs.get('command', '...')}`"
                        elif tool_name == "write_file": msg = f"TOOL: 📝 Schreibe: `{tool_kwargs.get('file_path', '...')}`"
                        elif tool_name == "take_screenshot": msg = "TOOL: 📸 Erstelle Screenshot..."
                        elif tool_name == "get_system_info": msg = "TOOL: 🛡️ Prüfe System..."
                        else: msg = f"TOOL: ⚙️ Nutze {tool_name}..."
                            
                        self.log(msg, "standard")
                        res = parse_and_execute_tool(json.dumps(data))
                        self.history.append({"role": "system", "content": f"TOOL_RESULT: {res or 'Done.'}"})
                    else: break
                except: break
            else: break
        self.set_status("idle")

    def run_voice_only(self):
        self.log("JARVIS ONLINE", "standard")
        while True:
            try:
                self.set_status("idle")
                if listen_for_wakeword():
                    self.set_status("listening")
                    audio_data = record_until_silence()
                    save_wav("latest_input.wav", audio_data)
                    text = self.transcribe_audio("latest_input.wav")
                    if text: self.run_ollama_agent(text)
            except: time.sleep(1)

if __name__ == "__main__":
    JarvisAssistant().run_voice_only()

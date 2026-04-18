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
import traceback

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
        try:
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                f.write(f"--- JARVIS MASTER SESSION START ({time.ctime()}) ---\n")
                f.flush()
        except: pass

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
        try:
            with open(CONFIG_FILE, "w") as f: json.dump(config, f, indent=4)
        except: pass

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
                self.log(f"Du ({self.last_detected_lang}): {text}", "standard")
            return text
        except: return ""

    def run_ollama_agent(self, user_text):
        # WICHTIG: Hintergrund-Thread für die GUI
        threading.Thread(target=self._run_agent_logic_safe, args=(user_text,), daemon=True).start()

    def _run_agent_logic_safe(self, user_text):
        try:
            with self.processing_lock:
                self._run_agent_loop(user_text)
        except Exception as e:
            self.log(f"CRITICAL ERROR: {e}\n{traceback.format_exc()}", "standard")
        finally:
            self.set_status("idle")

    def _run_agent_loop(self, user_text):
        self.log(f"USER INPUT: {user_text}", "debug")
        self.set_status("thinking")
        
        lang = self.language if self.language != "auto" else self.last_detected_lang
        
        prompt = (
            f"You are Jarvis, a highly autonomous AI agent on Linux. SECURITY_MODE: {self.security_mode}.\n\n"
            "CRITICAL RULES:\n"
            "1. NO FICTION: Never say 'I have done X' until the tool for X returned ERFOLG.\n"
            "2. SEARCH FIRST: If you need a recipe or info, use 'search_web' first.\n"
            "3. NO URL GUESSING: Never invent URLs. Only use URLs found via 'search_web'.\n"
            "4. TOOL FORMAT: Output exactly ONE raw JSON block at the end. No markdown blocks.\n"
            "5. ONE TOOL PER TURN: Wait for the result before the next action.\n"
            "6. BREVITY: While tools are running, speak max 1 short sentence about the current action.\n"
            "7. COMPLETION: Say 'Task complete' when done and NO JSON.\n"
            "8. LANGUAGE: Always respond in " + ("ENGLISH" if lang == "en" else "GERMAN") + ".\n\n"
            "AVAILABLE TOOLS:\n"
            "- get_system_info: { \"tool\": \"get_system_info\", \"kwargs\": {} }\n"
            "- search_web: { \"tool\": \"search_web\", \"kwargs\": { \"query\": \"...\" } }\n"
            "- execute_command: { \"tool\": \"execute_command\", \"kwargs\": { \"command\": \"...\" } }\n"
            "- write_file: { \"tool\": \"write_file\", \"kwargs\": { \"file_path\": \"...\", \"content\": \"...\" } }\n"
            "- take_screenshot: { \"tool\": \"take_screenshot\", \"kwargs\": {} }"
        )

        if not self.history: self.history.append({"role": "system", "content": prompt})
        else: self.history[0] = {"role": "system", "content": prompt}
        
        self.history.append({"role": "user", "content": user_text})
        spoken_history = []
        
        for step in range(20):
            if self.interrupted_by_wakeword: break
            try:
                full_resp = ""
                for chunk in ollama.chat(model=self.ollama_model, messages=self.history, stream=True):
                    if self.interrupted_by_wakeword: break
                    full_resp += chunk['message']['content']
                response_text = full_resp.strip()
            except Exception as e:
                self.log(f"Ollama Error: {e}", "standard"); break

            if not response_text: break
            self.log(f"RAW: {response_text}", "debug")
            
            # JSON extrahieren
            json_str = None
            f_brace = response_text.find('{'); l_brace = response_text.rfind('}')
            if f_brace != -1 and l_brace != -1 and l_brace > f_brace:
                json_str = response_text[f_brace:l_brace+1]

            # Text für Sprache säubern
            speech = response_text
            if json_str: speech = response_text[:f_brace].strip()
            
            speech = re.sub(r"<(thought|think)>.*?</\1>", "", speech, flags=re.S | re.I)
            speech = re.sub(r"<(thought|think)>.*", "", speech, flags=re.S | re.I)
            speech = re.sub(r"```[a-z]*", "", speech, flags=re.I)
            speech = re.sub(r"http[s]?://\S+", "", speech)
            speech = speech.replace("json", "").replace("[/json]", "").replace("```", "").strip()

            did_action = False

            if speech and re.search(r'[a-zA-ZäöüßÄÖÜ]', speech):
                if speech not in spoken_history:
                    self.set_status("speaking")
                    self.log(f"[Jarvis]: {speech}", "standard")
                    self.tts.speak(speech)
                    spoken_history.append(speech)
                    self.set_status("thinking")
                    did_action = True
            
            self.history.append({"role": "assistant", "content": response_text})
            
            if json_str:
                try:
                    data = json.loads(json_str)
                    if data.get('tool'):
                        tool_name = data.get('tool'); tool_kwargs = data.get('kwargs', {})
                        msg = f"TOOL: {tool_name} wird ausgeführt..."
                        if tool_name == "search_web": msg = f"TOOL: 🔍 Suche nach: '{tool_kwargs.get('query', '...')}'"
                        elif tool_name == "execute_command": msg = f"TOOL: 💻 Führe aus: `{tool_kwargs.get('command', '...')}`"
                        elif tool_name == "write_file": msg = f"TOOL: 📝 Schreibe: `{tool_kwargs.get('file_path', '...')}`"
                        elif tool_name == "take_screenshot": msg = "TOOL: 📸 Erstelle Screenshot..."
                        self.log(msg, "standard")
                        
                        res = parse_and_execute_tool(json.dumps(data))
                        self.history.append({"role": "system", "content": f"TOOL_RESULT: {res or 'Done.'}\nNEXT STEP: Continue until task is finished."})
                        did_action = True
                except Exception as e:
                    self.log(f"PARSE ERROR: {e}", "debug")
            
            if "task complete" in response_text.lower() or not did_action: break

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

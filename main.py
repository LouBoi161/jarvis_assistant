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
import gc
import requests

# Eigene Module
from audio_capture import listen_for_wakeword, record_until_silence, save_wav
from agent_tools import parse_and_execute_tool
from tts_engine import TTSEngine

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.ERROR)

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latest_session.log")
WHISPER_MODEL_NAME = "base"

class JarvisAssistant:
    def __init__(self):
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"\n\n--- SESSION START ({time.ctime()}) ---\n")
                f.flush()
        except: pass

        self.view_mode = "standard"
        self.ollama_model = "gemma4:e2b" # Standard auf das neue Modell
        self.security_mode = True
        self.language = "de"
        self.last_detected_lang = "de"
        self.tts_type = "kokoro-tts"
        self.piper_voice = "de_DE-thorsten-high"
        self.kokoro_voice = "gf_eva"
        self.on_status_change = None 
        
        self.load_config()
        self.log("--- JARVIS INITIALISIERUNG ---", "standard")
        self.stt_model = None
        self.init_tts()
        
        self.history = []
        self.processing_lock = threading.Lock()
        self.interrupted_by_user = False 
        self.interrupted_by_wakeword = False
        self.live_call_active = False

    def init_tts(self):
        config = {"tts_type": self.tts_type, "piper_voice": self.piper_voice, "kokoro_voice": self.kokoro_voice}
        self.tts = TTSEngine(config=config, use_gpu=True, stt_model=None)

    def log(self, message, level="debug", metadata=None):
        if message is None: return
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] [{level.upper()}] {str(message).strip()}\n"
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(entry); f.flush()
        except: pass
        if level == "standard": print(f"> {message}")

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    config = json.load(f)
                    for k, v in config.items():
                        if hasattr(self, k): setattr(self, k, v)
            except: pass

    def save_config(self):
        config = {attr: getattr(self, attr) for attr in ["ollama_model", "view_mode", "security_mode", "language", "tts_type", "piper_voice", "kokoro_voice"]}
        try:
            with open(CONFIG_FILE, "w") as f: json.dump(config, f, indent=4)
        except: pass

    def set_status(self, status):
        if self.on_status_change: self.on_status_change(status)

    def transcribe_audio(self, wav_path):
        try:
            if self.stt_model is None:
                self.stt_model = whisper.load_model(WHISPER_MODEL_NAME, device="cpu")
            result = self.stt_model.transcribe(wav_path)
            text = result["text"].strip()
            if text:
                self.last_detected_lang = result.get("language", "de")
                self.log(f"Du: {text}", "standard")
            return text
        except: return ""

    def stop_execution(self):
        self.interrupted_by_user = True
        self.interrupted_by_wakeword = True

    def run_ollama_agent(self, user_text):
        self.interrupted_by_user = False
        self.interrupted_by_wakeword = False
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
        self.log(f"Verarbeite: {user_text}", "debug")
        self.set_status("thinking")
        
        lang = self.language if self.language != "auto" else self.last_detected_lang
        prompt = (
            f"You are Jarvis, a highly autonomous AI agent on Linux. Mode: FINETUNED_EXPERT.\n\n"
            "CRITICAL RULES:\n"
            "1. JSON FORMAT: Output exactly ONE raw JSON block at the end. No markdown.\n"
            "   Format: { \"tool\": \"name\", \"kwargs\": { \"arg\": \"val\" } }\n"
            "2. COMPLETION: Say 'Task complete' and STOP when finished.\n"
            "3. BE BRIEF: Use max 1-2 sentences.\n"
        )

        # Historie verwalten (System Prompt immer an Pos 0)
        if not self.history:
            self.history.append({"role": "system", "content": prompt})
        else:
            self.history[0] = {"role": "system", "content": prompt}
        
        self.history.append({"role": "user", "content": user_text})
        spoken_history = []
        last_tool_sig = ""
        
        for step in range(15):
            if self.interrupted_by_user or self.interrupted_by_wakeword: break
            
            try:
                full_resp = ""
                # Timeout über Session steuern
                for chunk in ollama.chat(model=self.ollama_model, messages=self.history, stream=True, keep_alive=-1):
                    if self.interrupted_by_user or self.interrupted_by_wakeword: return
                    full_resp += chunk['message']['content']
                response_text = full_resp.strip()
            except Exception as e:
                self.log(f"Ollama Error: {e}", "standard"); break

            if not response_text or response_text == "None": break
            
            # JSON extrahieren
            json_str = None
            f_b = response_text.find('{'); l_b = response_text.rfind('}')
            if f_b != -1 and l_b != -1 and l_b > f_b: json_str = response_text[f_b:l_b+1]

            speech = response_text
            if json_str: speech = response_text[:f_b].strip()
            speech = re.sub(r"http[s]?://\S+", "", speech)
            speech = re.sub(r"<(thought|think)>.*?</\1>", "", speech, flags=re.S | re.I)
            speech = re.sub(r"```.*?```", "", speech, flags=re.S)
            speech = speech.replace("json", "").replace("**", "").replace("#", "").replace("`", "").strip()

            if speech and speech != "None" and re.search(r'[a-zA-ZäöüßÄÖÜ]', speech):
                if speech not in spoken_history:
                    self.set_status("speaking")
                    self.log(f"[Jarvis]: {speech}", "standard")
                    stop_ev = threading.Event()
                    if self.tts_type != "none": self.tts.speak(speech, interrupt_event=stop_ev)
                    self.set_status("thinking")
                    spoken_history.append(speech)
            
            self.history.append({"role": "assistant", "content": response_text})
            
            if json_str:
                try:
                    data = json.loads(json_str)
                    tool_name = data.get('tool'); tool_kwargs = data.get('kwargs', {})
                    if not tool_name or tool_name == "None": break

                    sig = f"{tool_name}_{json.dumps(tool_kwargs, sort_keys=True)}"
                    if sig == last_tool_sig: break
                    last_tool_sig = sig
                    
                    msg = f"TOOL: {tool_name}..."
                    if tool_name == "execute_command": msg = f"TOOL: 💻 Führe aus: `{tool_kwargs.get('command', '...')}`"
                    elif tool_name == "search_web": msg = f"TOOL: 🔍 Suche nach: '{tool_kwargs.get('query', '...')}'"
                    elif tool_name == "write_file": msg = f"TOOL: 📝 Schreibe: `{tool_kwargs.get('file_path', '...')}`"
                    
                    if msg and "None" not in msg: self.log(msg, "standard")
                    
                    res = parse_and_execute_tool(json.dumps(data))
                    self.history.append({"role": "system", "content": f"TOOL_RESULT: {res}\nCONTINUE until done."})
                except: break
            else: break
            
            if "task complete" in response_text.lower(): break
                
        self.set_status("idle")

    def update_config(self, d):
        for k, v in d.items():
            if hasattr(self, k): setattr(self, k, v)
        self.init_tts(); self.save_config()

    def unload_models(self):
        try: requests.post("http://localhost:11434/api/generate", json={"model": self.ollama_model, "keep_alive": 0}, timeout=2)
        except: pass
        if hasattr(self, 'tts') and self.tts: self.tts.unload_models()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        gc.collect()

    def run_voice_only(self):
        self.log("JARVIS ONLINE", "standard")
        while True:
            try:
                if not self.live_call_active:
                    self.set_status("idle")
                    if listen_for_wakeword():
                        self.set_status("listening")
                        data = record_until_silence()
                        save_wav("latest_input.wav", data)
                        text = self.transcribe_audio("latest_input.wav")
                        if text: self.run_ollama_agent(text)
                else: time.sleep(1)
            except: time.sleep(1)

if __name__ == "__main__":
    JarvisAssistant().run_voice_only()

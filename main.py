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

def robust_tool_extraction(s):
    """Extrahiert Tool-Informationen aus verschiedenen Formaten (JSON, Klartext-Tags oder Direkt-Befehle)."""
    try:
        s_upper = s.upper()
        
        # 0. WRITE_FILE (Markdown)
        write_match = re.search(r'WRITE_FILE:\s*([^\n]+)\n+```[a-zA-Z]*\n(.*?)```', s, re.DOTALL | re.IGNORECASE)
        if write_match:
            return {"tool": "write_file", "kwargs": {"file_path": write_match.group(1).strip().strip('`').strip('"'), "content": write_match.group(2).strip()}}

        # 1. SEARCH_WEB: <query>
        if "SEARCH_WEB:" in s_upper:
            query = s[s_upper.find("SEARCH_WEB:") + 11:].split('\n')[0].strip().strip('"')
            return {"tool": "search_web", "kwargs": {"query": query}}

        # 2. EXEC_CMD / FIREFOX / OPEN: <url/cmd>
        for tag in ["EXEC_CMD:", "FIREFOX:", "OPEN:"]:
            if tag in s_upper:
                cmd = s[s_upper.find(tag) + len(tag):].split('\n')[0].strip().strip('`').strip('"')
                if tag != "EXEC_CMD:" and "://" in cmd: cmd = f"firefox {cmd}"
                return {"tool": "execute_command", "kwargs": {"command": cmd}}

        # 3. Klassisches JSON
        f_b = s.find('{'); l_b = s.rfind('}')
        if f_b != -1 and l_b != -1:
            try:
                data = json.loads(s[f_b:l_b+1])
                if "tool" in data: return data
            except: pass
            
        # 4. JSON-Stil Regex Fallback
        tool_match = re.search(r'"tool":\s*"([^"]+)"', s)
        if tool_match:
            tool_name = tool_match.group(1)
            kwargs = {}
            for key in ["query", "command", "file_path"]:
                m = re.search(f'"{key}":\s*"([^"\\n]+)"', s)
                if m: kwargs[key] = m.group(1)
            return {"tool": tool_name, "kwargs": kwargs}

    except: pass
    return None

class JarvisAssistant:
    def __init__(self):
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(f"\n\n--- SESSION START ({time.ctime()}) ---\n")
                f.flush()
        except: pass

        self.view_mode = "standard"
        self.ollama_model = "gemma4:e2b"
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

    def init_tts(self):
        config = {"tts_type": self.tts_type, "piper_voice": self.piper_voice, "kokoro_voice": self.kokoro_voice}
        self.tts = TTSEngine(config=config, use_gpu=True, stt_model=None)

    def log(self, message, level="debug"):
        if not message: return
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
                # Nutze CUDA falls verfügbar für massiven Speed-Boost
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.stt_model = whisper.load_model(WHISPER_MODEL_NAME, device=device)
            
            result = self.stt_model.transcribe(wav_path, fp16=torch.cuda.is_available())
            text = result["text"].strip()
            
            # Whisper Halluzinations-Filter
            # 1. Zu kurz (meistens nur Rauschen)
            if len(text) < 3: return ""
            # 2. Bekannte Whisper-"Gedanken" (oft bei Stille)
            hallucinations = ["vielen dank", "untertitel", "zuschauen", "reutlingen", "you", "thanks for watching", "amara.org", "subtitle"]
            if any(h in text.lower() for h in hallucinations) and len(text) < 20:
                self.log(f"Whisper Halluzination ignoriert: {text}", "debug")
                return ""

            if text:
                self.last_detected_lang = result.get("language", "de")
                self.log(f"Du: {text}", "standard")
            return text
        except Exception as e:
            self.log(f"STT Error: {e}", "debug")
            return ""

    def stop_execution(self):
        self.interrupted_by_user = True
        self.interrupted_by_wakeword = True

    def run_ollama_agent(self, user_text, attached_files=None):
        self.interrupted_by_user = False
        self.interrupted_by_wakeword = False
        threading.Thread(target=self._run_agent_logic_safe, args=(user_text, attached_files), daemon=True).start()

    def _run_agent_logic_safe(self, user_text, attached_files=None):
        try:
            with self.processing_lock:
                self._run_agent_loop(user_text, attached_files)
        except Exception as e:
            self.log(f"CRITICAL ERROR: {e}\n{traceback.format_exc()}", "standard")
        finally:
            self.set_status("idle")

    def _run_agent_loop(self, user_text, attached_files=None):
        self.log(f"AGENT START: {user_text}", "debug")
        self.set_status("thinking")
        
        lang = self.language if self.language != "auto" else self.last_detected_lang
        now = time.strftime("%A, %d. %B %Y")
        prompt = (
            f"You are Jarvis, a highly autonomous AI expert on Linux. Mode: SYSTEM_OWNER.\n"
            f"TODAYS DATE: {now}.\n\n"
            "CRITICAL RULES:\n"
            "1. NO EXCUSES: Never say 'I can't open links' or 'I don't have access'. You HAVE tools!\n"
            "2. OPENING LINKS: If you have a URL, you MUST use 'EXEC_CMD: firefox <url>' immediately. NEVER just show the link to the user.\n"
            "3. TOOL FORMATS:\n"
            "   - SEARCH_WEB: <query>\n"
            "   - EXEC_CMD: <command>\n"
            "   - WRITE_FILE: <path>\\n```\\n<code>\\n```\n"
            "4. NO YAPPING: Max 1 short sentence per turn.\n"
            "5. COMPLETION: Say 'Task complete' only when truly finished.\n"
        )

        if not self.history:
            self.history.append({"role": "system", "content": prompt})
        else:
            self.history[0] = {"role": "system", "content": prompt}
        
        images = []
        if attached_files:
            for f in attached_files:
                if not os.path.exists(f): continue
                ext = f.split('.')[-1].lower()
                if ext in ['png', 'jpg', 'jpeg', 'bmp', 'webp', 'gif']:
                    images.append(f)
                elif ext in ['txt', 'md', 'html', 'json', 'csv', 'py', 'js', 'sh', 'c', 'cpp', 'rs', 'toml', 'yaml', 'xml', 'ini', 'cfg']:
                    try:
                        with open(f, "r", encoding="utf-8") as file:
                            content = file.read()
                            user_text += f"\n\n[Inhalt der angehängten Datei {os.path.basename(f)}]:\n```\n{content}\n```"
                    except Exception as e:
                        user_text += f"\n\n[Fehler beim Lesen von {os.path.basename(f)}: {e}]"
                elif ext in ['mp3', 'wav', 'ogg', 'm4a', 'flac']:
                    self.log(f"Transkribiere Audio: {os.path.basename(f)}...", "standard")
                    transcript = self.transcribe_audio(f)
                    if transcript:
                        user_text += f"\n\n[Audio-Transkription von {os.path.basename(f)}]:\n\"{transcript}\""
                else:
                    user_text += f"\n\n[Die Datei {os.path.basename(f)} wurde angehängt. Der absolute Pfad lautet: {os.path.abspath(f)}. Du kannst 'execute_command' nutzen, um Tools wie 'cat', 'pdfinfo' oder 'ffmpeg' auf diese Datei anzuwenden.]"

        user_msg = {"role": "user", "content": user_text}
        if images:
            user_msg["images"] = images
            
        self.history.append(user_msg)
        
        if len(self.history) > 10:
            self.history = [self.history[0]] + self.history[-5:]

        spoken_history = []
        last_tool_sig = ""
        
        for step in range(15):
            if self.interrupted_by_user or self.interrupted_by_wakeword: break
            try:
                full_resp = ""
                self.log(f"Schritt {step+1}: Warte auf Ollama...", "debug")
                for chunk in ollama.chat(model=self.ollama_model, messages=self.history, stream=True, keep_alive=300):
                    if self.interrupted_by_user or self.interrupted_by_wakeword: return
                    full_resp += chunk['message']['content']
                response_text = full_resp.strip()
            except Exception as e:
                self.log(f"Ollama Error: {e}", "standard"); break

            if not response_text: break
            
            data = robust_tool_extraction(response_text)
            
            # Finde Startpunkt für Tool-Tag
            tags = ["SEARCH_WEB:", "EXEC_CMD:", "WRITE_FILE:", "FIREFOX:", "OPEN:", "GET_SYSTEM_INFO", "TAKE_SCREENSHOT", "{"]
            tag_start = len(response_text)
            for t in tags:
                idx = response_text.upper().find(t)
                if idx != -1 and idx < tag_start: tag_start = idx
            
            speech = response_text[:tag_start].strip()

            # Falls kein Tool erkannt wurde, aber wir in Schritt 1 sind -> Zwinge Suche
            if not data and step == 0:
                fact_keywords = ["wer", "was", "wie", "wann", "wo", "alt", "geboren", "news", "nachrichten", "öffne", "open"]
                if any(k in user_text.lower() for k in fact_keywords):
                    self.history.append({"role": "assistant", "content": response_text})
                    self.history.append({"role": "system", "content": "FEHLER: Du hast nur geredet, aber kein Tool genutzt. Handle jetzt!"})
                    continue

            # Sprache säubern
            speech = re.sub(r"http[s]?://\S+", "", speech)
            speech = re.sub(r"<(thought|think)>.*?</\1>", "", speech, flags=re.S | re.I)
            if data: speech = re.sub(r"```.*?```", "", speech, flags=re.S)
            speech = speech.replace("json", "").replace("**", "").replace("#", "").replace("`", "").strip()

            if speech and speech != "None" and re.search(r'[a-zA-ZäöüßÄÖÜ]', speech):
                if speech not in spoken_history:
                    self.set_status("speaking")
                    self.log(f"[Jarvis]: {speech}", "standard")
                    stop_ev = threading.Event()
                    if self.tts_type != "none":
                        tts_thread = threading.Thread(target=self.tts.speak, args=(speech, stop_ev), daemon=True)
                        tts_thread.start(); tts_thread.join(timeout=15)
                    self.set_status("thinking")
                    spoken_history.append(speech)
            
            self.history.append({"role": "assistant", "content": response_text})
            
            if data:
                try:
                    tool_name = data.get('tool'); tool_kwargs = data.get('kwargs', {})
                    if not tool_name: break

                    sig = f"{tool_name}_{json.dumps(tool_kwargs, sort_keys=True)[:100]}"
                    if sig == last_tool_sig: break
                    last_tool_sig = sig
                    
                    msg = f"TOOL: {tool_name}"
                    if tool_name == "execute_command": msg = f"TOOL: 💻 Führe aus: `{tool_kwargs.get('command', '...')}`"
                    elif tool_name == "search_web": msg = f"TOOL: 🔍 Suche nach: '{tool_kwargs.get('query', '...')}'"
                    elif tool_name == "write_file": msg = f"TOOL: 📝 Speichere Datei: `{tool_kwargs.get('file_path', '...')}`"
                    self.log(msg, "standard")
                    
                    res = parse_and_execute_tool(json.dumps(data))
                    self.history.append({"role": "system", "content": f"TOOL_RESULT: {res}\nCONTINUE until done."})
                except Exception as e:
                    self.log(f"Tool-Error: {e}", "debug"); break
            else:
                if "task complete" in response_text.lower(): break
                if step > 0: break
                
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
                self.set_status("idle")
                if listen_for_wakeword():
                    self.set_status("listening")
                    data = record_until_silence()
                    save_wav("latest_input.wav", data)
                    text = self.transcribe_audio("latest_input.wav")
                    if text: self.run_ollama_agent(text)
            except: time.sleep(1)

if __name__ == "__main__":
    JarvisAssistant().run_voice_only()

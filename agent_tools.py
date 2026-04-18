import subprocess
from ddgs import DDGS
import json
import pexpect
import shutil
import sys
import ollama
import os
import time
import signal
from PyQt5.QtWidgets import QApplication, QInputDialog, QLineEdit, QComboBox, QVBoxLayout, QDialog, QPushButton, QLabel, QCheckBox
from multiprocessing import Process, Queue
from PyQt5.QtCore import Qt

active_process = None
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

def get_security_mode():
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                return json.load(f).get("security_mode", True)
    except: pass
    return True

def _gui_worker(queue, func_name, *args, **kwargs):
    app = QApplication.instance() or QApplication(sys.argv)
    if func_name == "get_gui_password": result = _get_gui_password_logic(*args, **kwargs)
    else: result = None
    queue.put(result)

def _get_gui_password_logic():
    dialog = QInputDialog()
    dialog.setWindowTitle("Jarvis Sicherheit")
    dialog.setLabelText("Passwort benötigt:")
    dialog.setTextEchoMode(QLineEdit.Password)
    dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowStaysOnTopHint)
    return dialog.textValue() if dialog.exec_() == QDialog.Accepted else ""

def get_gui_password():
    q = Queue(); p = Process(target=_gui_worker, args=(q, "get_gui_password"))
    p.start(); p.join(); return q.get()

def search_web(query: str, max_results: int = 3) -> str:
    print(f"[Tool Execution] Suche im Web nach: '{query}'")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results: return "FEHLER: Keine Suchergebnisse gefunden."
        formatted = ""
        for i, res in enumerate(results):
            formatted += f"[{i+1}] {res['title']}\nURL: {res.get('href', 'N/A')}\nInhalt: {res['body']}\n\n"
        return formatted
    except Exception as e: return f"FEHLER bei Websuche: {e}"

def write_file(file_path: str, content: str) -> str:
    print(f"[Tool Execution] Schreibe Datei: '{file_path}'")
    try:
        path = os.path.abspath(os.path.expanduser(file_path))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"ERFOLG: Datei geschrieben unter: {path}"
    except Exception as e: return f"FEHLER beim Schreiben: {e}"

def take_screenshot() -> str:
    print("[Tool Execution] Screenshot via grim...")
    try:
        path = os.path.abspath(os.path.expanduser(f"~/screenshot_{int(time.time())}.png"))
        subprocess.run(f"grim '{path}'", shell=True, check=True)
        return f"ERFOLG: Screenshot gespeichert unter: {path}"
    except Exception as e: return f"FEHLER: {e}"

def execute_command(command: str) -> str:
    global active_process
    if get_security_mode() and any(x in command for x in ["sudo", "pacman", "yay"]):
        return "FEHLER: Sicherheitsmodus aktiv. Befehl blockiert."

    if active_process and active_process.isalive():
        try: os.killpg(os.getpgid(active_process.pid), signal.SIGTERM); active_process.close(force=True)
        except: pass
    
    print(f"[Tool Execution] Führe aus: '{command}'")
    
    # GUI Apps / URLs
    gui_apps = ["firefox", "brave", "code", "vlc", "steam", "heroic", "nautilus", "dolphin", "xdg-open"]
    if any(x in command.lower() for x in gui_apps) or command.startswith("http"):
        subprocess.Popen(command, shell=True, preexec_fn=os.setsid)
        return f"ERFOLG: '{command}' wurde gestartet."

    interactive = any(x in command for x in ["sudo", "install", "-S", "update"])
    if any(x in command for x in ["which ", "ls ", "-Q", "grep "]): interactive = False

    if interactive:
        try:
            active_process = pexpect.spawn(command, encoding='utf-8', timeout=10, setpgid=True)
            return read_process_output()
        except Exception as e: return f"FEHLER: {e}"
    else:
        try:
            res = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=20)
            out = filter_noise(res.stdout.strip())
            err = filter_noise(res.stderr.strip())
            if out: return out
            if any(x in command for x in ["which", "ls", "grep"]): return "INFO: Nichts gefunden."
            return err if err else "ERFOLG: Befehl ohne Rückgabe ausgeführt."
        except Exception as e: return f"FEHLER: {e}"

def filter_noise(text: str) -> str:
    if not text: return ""
    noise = ["ALSA lib", "snd_pcm", "connect(2)", "attempt to connect", "SoX could not", "VAD loaded"]
    return "\n".join([l for l in text.split("\n") if not any(n in l for n in noise)]).strip()

def read_process_output() -> str:
    global active_process; full_output = ""
    prompts = ['(?i)password', '(?i)passwort', r'\[Y/n\]', r'\[y/N\]', pexpect.EOF, pexpect.TIMEOUT]
    while True:
        try:
            index = active_process.expect(prompts, timeout=5)
            full_output += active_process.before + (str(active_process.after) if index < 4 else "")
            if index <= 1: 
                pwd = get_gui_password()
                if pwd: active_process.sendline(pwd); time.sleep(0.5); continue
                else: active_process.sendline("\x03"); return full_output + "\nABBRUCH."
            elif index == 4: return filter_noise(full_output) + "\n[FERTIG]"
            elif index == 5: 
                if not active_process.isalive(): return filter_noise(full_output)
                return filter_output + "\n[TIMEOUT]"
            else: active_process.sendline("y"); continue
        except: return filter_noise(full_output)

def get_system_info() -> str:
    return json.dumps({"os": "Linux", "distro": "CachyOS", "user": os.environ.get("USER", "louis")})

def parse_and_execute_tool(json_string: str) -> str:
    try:
        data = json.loads(json_string); name = data.get("tool"); kw = data.get("kwargs", {})
        if name == "search_web": return search_web(**kw)
        if name == "execute_command": return execute_command(**kw)
        if name == "write_file": return write_file(**kw)
        if name == "take_screenshot": return take_screenshot()
        if name == "get_system_info": return get_system_info()
        return f"FEHLER: Unbekanntes Tool: {name}"
    except Exception as e: return f"FEHLER beim JSON-Parsing: {e}"

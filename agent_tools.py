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

# Globale Variable für persistente Shell-Sitzung
active_process = None
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
VOICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices")

def get_security_mode():
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                return json.load(f).get("security_mode", True)
    except: pass
    return True

from multiprocessing import Process, Queue

def _gui_worker(queue, func_name, *args, **kwargs):
    app = QApplication.instance()
    if app is None: app = QApplication(sys.argv)
    if func_name == "manage_jarvis_gui": result = _manage_jarvis_gui_logic(*args, **kwargs)
    elif func_name == "get_gui_password": result = _get_gui_password_logic(*args, **kwargs)
    else: result = None
    queue.put(result)

def manage_jarvis_gui(current_config):
    q = Queue(); p = Process(target=_gui_worker, args=(q, "manage_jarvis_gui", current_config))
    p.start(); p.join(); return q.get()

def _manage_jarvis_gui_logic(config):
    dialog = QDialog(); dialog.setWindowTitle("JARVIS System Control"); dialog.setMinimumWidth(450)
    dialog.setStyleSheet("QDialog { background-color: #0b0e14; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; } QLabel { color: #00d4ff; font-size: 14px; font-weight: bold; margin-top: 10px; } QComboBox, QCheckBox, QLineEdit { background-color: #1a1f29; color: #ffffff; border: 1px solid #00d4ff; border-radius: 4px; padding: 8px; font-size: 13px; } QPushButton { background-color: #00d4ff; color: #0b0e14; border-radius: 4px; padding: 12px; font-size: 14px; font-weight: bold; margin-top: 20px; }")
    layout = QVBoxLayout()
    layout.addWidget(QLabel("Ollama Model:")); model_combo = QComboBox()
    try:
        m = [m.model for m in ollama.list().models]
        model_combo.addItems(sorted(list(set(m))))
        if config.get("ollama_model") in m: model_combo.setCurrentText(config.get("ollama_model"))
    except: model_combo.addItem(config.get("ollama_model"))
    layout.addWidget(model_combo)
    layout.addWidget(QLabel("Language:")); lang_combo = QComboBox(); lang_combo.addItems(["de", "en", "auto"]); lang_combo.setCurrentText(config.get("language", "de")); layout.addWidget(lang_combo)
    layout.addWidget(QLabel("TTS Engine:")); tts_combo = QComboBox(); tts_combo.addItems(["qwen3-tts", "piper-tts", "none"]); tts_combo.setCurrentText(config.get("tts_type", "qwen3-tts")); layout.addWidget(tts_combo)
    security_checkbox = QCheckBox("Security Mode"); security_checkbox.setChecked(config.get("security_mode", True)); layout.addWidget(security_checkbox)
    save_btn = QPushButton("SAVE"); layout.addWidget(save_btn)
    new_config = {}
    def save():
        new_config.update({"ollama_model": model_combo.currentText(), "language": lang_combo.currentText(), "security_mode": security_checkbox.isChecked(), "tts_type": tts_combo.currentText()})
        dialog.accept()
    save_btn.clicked.connect(save); dialog.setLayout(layout); dialog.exec_()
    return json.dumps({"type": "config_update", "data": new_config}) if new_config else None

def _get_gui_password_logic():
    dialog = QInputDialog(); dialog.setWindowTitle("Jarvis Sicherheit"); dialog.setLabelText("Passwort benötigt:"); dialog.setTextEchoMode(QLineEdit.Password); dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowStaysOnTopHint)
    return dialog.textValue() if dialog.exec_() == QDialog.Accepted else ""

def get_gui_password():
    q = Queue(); p = Process(target=_gui_worker, args=(q, "get_gui_password")); p.start(); p.join(); return q.get()

def search_web(query: str, max_results: int = 3) -> str:
    try:
        with DDGS() as ddgs:
            r = list(ddgs.text(query, max_results=max_results))
            return "\n\n".join([f"{i+1}. {x['title']}\n{x['body']}" for i, x in enumerate(r)]) if r else "Keine Treffer."
    except Exception as e: return f"Websuche Fehler: {e}"

def write_file(file_path: str, content: str) -> str:
    try:
        path = os.path.expanduser(file_path); os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f: f.write(content)
        return f"Datei {file_path} gespeichert."
    except Exception as e: return f"Fehler: {e}"

def take_screenshot() -> str:
    """Macht einen Screenshot mit PyQt5 (funktioniert zuverlässiger unter Wayland/CachyOS)."""
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QTimer
    import os
    import time
    
    print("[Tool Execution] Erstelle Screenshot via Qt...")
    try:
        # Pfad vorbereiten
        filename = f"screenshot_{int(time.time())}.png"
        path = os.path.expanduser(f"~/{filename}")
        
        # Wir brauchen eine laufende App-Instanz
        app = QApplication.instance()
        if not app:
            return "Fehler: Kein GUI-Kontext für Screenshot gefunden."
        
        # Screenshot vom primären Bildschirm
        screen = app.primaryScreen()
        screenshot = screen.grabWindow(0)
        screenshot.save(path, "png")
        
        return f"ERFOLG: Der Screenshot wurde erstellt und unter '{path}' gespeichert. Du kannst ihn dem Nutzer jetzt melden."
    except Exception as e:
        return f"FEHLER beim Screenshot: {e}"

def execute_command(command: str) -> str:
    global active_process
    if get_security_mode() and any(x in command for x in ["sudo", "pacman", "yay"]): 
        return "FEHLER: Der Sicherheitsmodus ist AKTIVIERT. Dieser Befehl wurde blockiert."
    
    if active_process and active_process.isalive():
        try: 
            os.killpg(os.getpgid(active_process.pid), signal.SIGTERM)
            active_process.close(force=True)
        except: pass

    # Interaktivitäts-Check
    interactive = any(x in command for x in ["sudo", "install", "update", "upgrade", "-S", "-y"])
    query_keywords = ["which ", "ls ", "-Q", "grep ", "cat ", "whereis ", "type "]
    if any(x in command.lower() for x in query_keywords): interactive = False

    if interactive:
        try:
            active_process = pexpect.spawn(command, encoding='utf-8', timeout=5, setpgid=True)
            return read_process_output()
        except Exception as e: return f"FEHLER beim Starten: {e}"
    else:
        try:
            # GUI Apps (Hintergrund)
            gui_keywords = ["firefox", "code", "vlc", "steam", "heroic", "brave", "nautilus", "dolphin", "xdg-open"]
            if any(x in command.lower() for x in gui_keywords):
                subprocess.Popen(command, shell=True, preexec_fn=os.setsid)
                return f"ERFOLG: Das Programm oder der Ordner ('{command}') wurde im Hintergrund geöffnet."
            
            res = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=15)
            out = filter_noise(res.stdout.strip())
            err = filter_noise(res.stderr.strip())
            
            if out: return out
            if any(x in command for x in ["which", "ls", "grep"]): 
                return f"INFO: Keine Treffer für '{command}' gefunden."
            return f"ERFOLG: Befehl ausgeführt. Rückgabe: {err if err else 'Keine'}"
        except Exception as e: return f"FEHLER bei Ausführung: {e}"

def filter_noise(text):
    if not text: return ""
    noise = ["ALSA lib", "snd_pcm", "connect(2)", "attempt to connect", "SoX could not", "VAD loaded"]
    return "\n".join([l for l in text.split("\n") if not any(n in l for n in noise)]).strip()

def read_process_output():
    global active_process; out = ""
    prompts = ['(?i)password', '(?i)passwort', r'\[Y/n\]', r'\[y/N\]', pexpect.EOF, pexpect.TIMEOUT]
    while True:
        try:
            idx = active_process.expect(prompts, timeout=2)
            out += active_process.before + (str(active_process.after) if idx < 4 else "")
            if idx <= 1: # Pwd
                p = get_gui_password()
                if p: active_process.sendline(p); time.sleep(0.5); continue
                else: active_process.sendline("\x03"); return out + "\nAbbruch."
            elif idx == 4: return filter_noise(out) + "\nFertig."
            elif idx == 5: 
                if not active_process.isalive(): return filter_noise(out)
                if any(x in out.lower() for x in ["install", "update"]): continue
                return filter_noise(out) + "\nLäuft noch..."
            else: # Auto-y
                active_process.sendline("y"); out += "\n[y]"; continue
        except: return filter_noise(out)

def get_system_info():
    import platform
    try:
        dist = "Linux"
        if os.path.exists("/etc/os-release"):
            with open("/etc/os-release") as f:
                for l in f:
                    if l.startswith("PRETTY_NAME="): dist = l.split("=")[1].strip('" \n'); break
        return json.dumps({"os": platform.system(), "distro": dist, "arch": platform.machine(), "user": os.environ.get("USER", "louis")})
    except Exception as e: return f"Fehler: {e}"

def parse_and_execute_tool(json_string):
    try:
        data = json.loads(json_string); name = data.get("tool"); kw = data.get("kwargs", {})
        if name == "search_web": return search_web(**kw)
        if name == "execute_command": return execute_command(**kw)
        if name == "write_file": return write_file(**kw)
        if name == "take_screenshot": return take_screenshot()
        if name == "get_system_info": return get_system_info()
        return f"Unbekannt: {name}"
    except: return "JSON Fehler."

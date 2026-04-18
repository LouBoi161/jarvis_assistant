import subprocess
from ddgs import DDGS
import json
import pexpect
import shutil
import sys
import ollama
import os
from PyQt5.QtWidgets import QApplication, QInputDialog, QLineEdit, QComboBox, QVBoxLayout, QDialog, QPushButton, QLabel, QCheckBox

# Globale Variable für persistente Shell-Sitzung
active_process = None
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
VOICES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices")

def get_security_mode():
    """Liest den Sicherheitsmodus direkt aus der Config."""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                return json.load(f).get("security_mode", True)
    except:
        pass
    return True

from multiprocessing import Process, Queue

def _gui_worker(queue, func_name, *args, **kwargs):
    """Interner Worker, der in einem eigenen Prozess läuft, um Qt-Threadsicherheit zu garantieren."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    if func_name == "manage_jarvis_gui":
        result = _manage_jarvis_gui_logic(*args, **kwargs)
    elif func_name == "get_gui_password":
        result = _get_gui_password_logic(*args, **kwargs)
    else:
        result = None
    
    queue.put(result)

def manage_jarvis_gui(current_config):
    """Öffnet das GUI-Fenster in einem sicheren Unterprozess."""
    q = Queue()
    p = Process(target=_gui_worker, args=(q, "manage_jarvis_gui", current_config))
    p.start()
    p.join()
    return q.get()

def _manage_jarvis_gui_logic(config):
    """Die eigentliche Logik der Einstellungs-GUI."""
    dialog = QDialog()
    dialog.setWindowTitle("JARVIS System Control")
    dialog.setMinimumWidth(450)
    
    # Modernes Jarvis-Design (QSS)
    dialog.setStyleSheet("""
        QDialog {
            background-color: #0b0e14;
            color: #e0e0e0;
            font-family: 'Segoe UI', 'Roboto', sans-serif;
        }
        QLabel {
            color: #00d4ff;
            font-size: 14px;
            font-weight: bold;
            margin-top: 10px;
        }
        QComboBox, QCheckBox, QLineEdit {
            background-color: #1a1f29;
            color: #ffffff;
            border: 1px solid #00d4ff;
            border-radius: 4px;
            padding: 8px;
            font-size: 13px;
        }
        QPushButton {
            background-color: #00d4ff;
            color: #0b0e14;
            border-radius: 4px;
            padding: 12px;
            font-size: 14px;
            font-weight: bold;
            margin-top: 20px;
        }
    """)

    layout = QVBoxLayout()
    
    # Ollama Model
    layout.addWidget(QLabel("Ollama Model Selection:"))
    model_combo = QComboBox()
    try:
        local_models_resp = ollama.list()
        if hasattr(local_models_resp, 'models'):
            models = [m.model for m in local_models_resp.models]
        else:
            models = [m['name'] for m in local_models_resp.get('models', [])]
    except:
        models = [config.get("ollama_model")]
    
    model_combo.addItems(sorted(list(set(models))))
    if config.get("ollama_model") in models:
        model_combo.setCurrentText(config.get("ollama_model"))
    layout.addWidget(model_combo)
    
    # System View Mode
    layout.addWidget(QLabel("System View Mode:"))
    view_mode_combo = QComboBox()
    view_mode_combo.addItems(["standard", "debug"])
    view_mode_combo.setCurrentText(config.get("view_mode", "standard"))
    layout.addWidget(view_mode_combo)

    # Assistant Language
    layout.addWidget(QLabel("Assistant Language:"))
    lang_combo = QComboBox()
    lang_combo.addItems(["de", "en", "auto"])
    lang_combo.setCurrentText(config.get("language", "de"))
    layout.addWidget(lang_combo)

    # TTS Type
    layout.addWidget(QLabel("TTS Engine:"))
    tts_combo = QComboBox()
    tts_combo.addItems(["qwen3-tts", "piper-tts", "none"])
    tts_combo.setCurrentText(config.get("tts_type", "qwen3-tts"))
    layout.addWidget(tts_combo)

    # Piper Voice
    piper_label = QLabel("Piper Voice (name-lang-quality):")
    layout.addWidget(piper_label)
    piper_combo = QComboBox()
    
    # Scan piper_models directory for available voices
    piper_models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "piper_models")
    if not os.path.exists(piper_models_dir):
        os.makedirs(piper_models_dir)
    
    # Get all .onnx files and use their names for the list
    piper_voices = [f.replace(".onnx", "") for f in os.listdir(piper_models_dir) if f.endswith(".onnx")]
    
    if not piper_voices:
        piper_voices = ["de_DE-thorsten-high"] # Default fallback

    piper_combo.addItems(sorted(list(set(piper_voices))))
    piper_combo.setEditable(True)
    if config.get("piper_voice") in piper_voices:
        piper_combo.setCurrentText(config.get("piper_voice"))
    else:
        piper_combo.setCurrentText(config.get("piper_voice", "de_DE-thorsten-high"))
    layout.addWidget(piper_combo)

    # Qwen Voice
    qwen_label = QLabel("Qwen Voice (.wav in voices/):")
    layout.addWidget(qwen_label)
    qwen_combo = QComboBox()
    if not os.path.exists(VOICES_DIR):
        os.makedirs(VOICES_DIR)
    
    qwen_voices = [f for f in os.listdir(VOICES_DIR) if f.endswith(".wav")]
    if not qwen_voices:
        qwen_voices = ["default.wav"]
    
    qwen_combo.addItems(qwen_voices)
    if config.get("qwen_voice") in qwen_voices:
        qwen_combo.setCurrentText(config.get("qwen_voice"))
    layout.addWidget(qwen_combo)

    # Visibility logic
    def update_visibility():
        is_piper = tts_combo.currentText() == "piper-tts"
        is_qwen = tts_combo.currentText() == "qwen3-tts"
        piper_label.setVisible(is_piper)
        piper_combo.setVisible(is_piper)
        qwen_label.setVisible(is_qwen)
        qwen_combo.setVisible(is_qwen)

    tts_combo.currentTextChanged.connect(update_visibility)
    update_visibility()

    # Security
    layout.addWidget(QLabel("Security Settings:"))
    security_checkbox = QCheckBox("Security Mode (Commands Blocked)")
    security_checkbox.setChecked(config.get("security_mode", True))
    layout.addWidget(security_checkbox)
    
    save_btn = QPushButton("APPLY CONFIGURATION")
    layout.addWidget(save_btn)
    
    new_config = {}
    def save():
        new_config['ollama_model'] = model_combo.currentText()
        new_config['view_mode'] = view_mode_combo.currentText()
        new_config['language'] = lang_combo.currentText()
        new_config['security_mode'] = security_checkbox.isChecked()
        new_config['tts_type'] = tts_combo.currentText()
        new_config['piper_voice'] = piper_combo.currentText()
        new_config['qwen_voice'] = qwen_combo.currentText()
        dialog.accept()
        
    save_btn.clicked.connect(save)
    dialog.setLayout(layout)
    dialog.exec_()
    
    if new_config:
        return json.dumps({"type": "config_update", "data": new_config})
    return "No changes applied."

def get_gui_password():
    """Öffnet die Passwort-Abfrage in einem sicheren Unterprozess."""
    q = Queue()
    p = Process(target=_gui_worker, args=(q, "get_gui_password"))
    p.start()
    p.join()
    return q.get()

def _get_gui_password_logic():
    """Die eigentliche Logik der Passwort-Abfrage."""
    pwd, ok = QInputDialog.getText(
        None, 
        "Jarvis Sicherheit", 
        "Ein Programm (z.B. sudo/yay) fordert dein Passwort an:", 
        QLineEdit.Password
    )
    if ok and pwd:
        return pwd
    return ""

def search_web(query: str, max_results: int = 3) -> str:
    """Sucht im Web nach dem gegebenen Suchbegriff und gibt die Ergebnisse zurück."""
    print(f"[Tool Execution] Suche im Web nach: '{query}'")
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        
        if not results:
            return "Keine Ergebnisse gefunden."
            
        formatted_results = ""
        for i, res in enumerate(results):
            formatted_results += f"{i+1}. {res['title']}\nURL: {res.get('href', 'Keine URL')}\nBeschreibung: {res['body']}\n\n"
        return formatted_results
    except Exception as e:
        return f"Fehler bei der Websuche: {e}"

def execute_command(command: str) -> str:
    """Führt einen Linux-Systembefehl aus."""
    global active_process
    
    if get_security_mode():
        print(f"[Tool Execution] BLOCKIERT (Sicherheitsmodus): '{command}'")
        return "FEHLER: Der Sicherheitsmodus ist AKTIVIERT. Jarvis darf aktuell KEINE Systembefehle ausführen."

    # Beende vorherigen aktiven Prozess und seine gesamte Gruppe
    if active_process and active_process.isalive():
        print(f"[SYSTEM]: Beende laufenden Prozess {active_process.pid}...")
        try:
            import os
            import signal
            os.killpg(os.getpgid(active_process.pid), signal.SIGTERM)
            active_process.close(force=True)
        except:
            if active_process: active_process.terminate(force=True)
    
    print(f"[Tool Execution] Führe Befehl aus: '{command}'")
    
    # Schutz vor pacman locks
    if "pacman" in command or "yay" in command:
        if os.path.exists("/var/lib/pacman/db.lck"):
            return "FEHLER: Der Paketmanager ist blockiert (/var/lib/pacman/db.lck existiert). Ein anderer Prozess (oder ein abgestürzter Jarvis-Prozess) nutzt gerade pacman. Bitte warte kurz oder lösche die Sperrdatei."

    cmd_parts = command.split()
    
    # Prüfe ob es ein interaktiver Befehl ist
    interactive_commands = ["sudo", "pacman", "yay", "paru", "apt"]
    is_interactive = any(cmd in cmd_parts for cmd in interactive_commands)
    
    if is_interactive:
        try:
            # Starte in einer neuen Prozessgruppe für sauberes Beenden
            active_process = pexpect.spawn(command, encoding='utf-8', timeout=5, setpgid=True)
            return read_process_output()
        except Exception as e:
            return f"Fehler beim Starten des interaktiven Befehls: {e}"
    else:
        # Normaler Konsolenbefehl
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
            output = filter_noise(result.stdout.strip())
            error = filter_noise(result.stderr.strip())
            if output: return output
            if error: return f"Fehler-Output: {error}"
            return "Befehl wurde ohne Rückgabe ausgeführt."
        except Exception as e:
            return f"Fehler bei der Ausführung: {e}"

def filter_noise(text: str) -> str:
    """Filtert ALSA, SoX und andere irrelevante Warnungen aus dem Output."""
    if not text: return ""
    lines = text.split('\n')
    filtered_lines = []
    noise_patterns = [
        "ALSA lib", "snd_pcm", "connect(2) call to", "attempt to connect", "Cannot open device",
        "a52 is only for playback", "Invalid card", "SoX could not be found", "flash-attn is not installed",
        "Loading Silero VAD", "Using cache found in", "VAD loaded", "Unknown PCM"
    ]
    for line in lines:
        if not any(pattern in line for pattern in noise_patterns):
            filtered_lines.append(line)
    return '\n'.join(filtered_lines).strip()

def read_process_output() -> str:
    """Liest den Output des Terminals in einer Schleife, bis es pausiert oder beendet ist."""
    global active_process
    full_output = ""
    
    input_prompts = [
        '(?i)password', '(?i)passwort', 
        r'\[Y/n\]', r'\[y/N\]', r'\[y/n\]', r'\? \[Y/n\]',
        r'\(J/n\)', r'\(j/N\)', r'==> .*:', r':: .*:', r'\? \[y/N\]',
        pexpect.EOF, pexpect.TIMEOUT
    ]
    
    while True:
        try:
            index = active_process.expect(input_prompts, timeout=2)
            
            output_chunk = active_process.before + (str(active_process.after) if active_process.after not in [pexpect.EOF, pexpect.TIMEOUT] else "")
            full_output += output_chunk
            
            if index <= 1: # Passwort prompt
                if get_security_mode():
                    active_process.sendline("\x03")
                    return filter_noise(full_output) + "\n[SYSTEM: Passwort-Eingabe blockiert]"

                print("[SYSTEM]: Passwort-Anfrage erkannt...")
                pwd = get_gui_password()
                if pwd:
                    active_process.sendline(pwd)
                    time.sleep(0.5)
                    continue 
                else:
                    active_process.sendline("\x03")
                    return filter_noise(full_output) + "\n[SYSTEM: Abbruch durch Benutzer]"
                    
            elif index == len(input_prompts) - 2: # EOF
                return filter_noise(full_output) + "\n\n[BEFEHL ABGESCHLOSSEN]"
            
            elif index == len(input_prompts) - 1: # TIMEOUT
                if not active_process.isalive():
                    return filter_noise(full_output) + "\n\n[BEFEHL BEENDET]"
                # Bei pacman/yay warten wir länger
                if any(x in full_output.lower() for x in ["install", "update", "upgrade", "fetching"]):
                    continue
                return filter_noise(full_output) + "\n\n[SYSTEM: Befehl läuft noch im Hintergrund...]"
                
            else: # Auto-Confirm (y)
                active_process.sendline("y")
                full_output += "\n[SYSTEM: Auto-Bestätigung 'y' ausgeführt]\n"
                continue
                
        except Exception as e:
            return filter_noise(full_output) + f"\n[SYSTEM: Fehler beim Lesen: {e}]"

def get_system_info() -> str:
    """Gibt Informationen über das Betriebssystem und die Hardware zurück (sicher)."""
    import platform
    import os
    
    try:
        # Versuche detaillierte Linux-Distro Info zu bekommen
        distro_info = "Unbekannte Linux-Distribution"
        if os.path.exists("/etc/os-release"):
            with open("/etc/os-release", "r") as f:
                for line in f:
                    if line.startswith("PRETTY_NAME="):
                        distro_info = line.split("=")[1].strip().strip('"')
                        break
        
        info = {
            "os": platform.system(),
            "distro": distro_info,
            "kernel": platform.release(),
            "arch": platform.machine(),
            "cpu": platform.processor(),
            "shell": os.environ.get("SHELL", "Unbekannt"),
            "user": os.environ.get("USER", "louis")
        }
        return json.dumps(info, indent=4, ensure_ascii=False)
    except Exception as e:
        return f"Fehler beim Abrufen der System-Infos: {e}"

def parse_and_execute_tool(json_string: str) -> str:
    """Parst die JSON-Antwort von Gemma und führt das entsprechende Tool aus."""
    try:
        data = json.loads(json_string)
        tool_name = data.get("tool")
        kwargs = data.get("kwargs", {})
        
        if tool_name == "search_web":
            return search_web(**kwargs)
        elif tool_name == "execute_command":
            return execute_command(**kwargs)
        elif tool_name == "send_input":
            return send_input(**kwargs)
        elif tool_name == "manage_jarvis_gui":
            return manage_jarvis_gui(kwargs)
        elif tool_name == "get_system_info":
            return get_system_info()
        else:
            return f"Unbekanntes Tool: {tool_name}"
    except json.JSONDecodeError:
        return "Fehler: Die Tool-Anfrage war kein gültiges JSON."

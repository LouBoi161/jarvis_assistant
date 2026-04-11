import subprocess
from ddgs import DDGS
import json
import pexpect
import shutil
import sys
import ollama
from PyQt5.QtWidgets import QApplication, QInputDialog, QLineEdit, QComboBox, QVBoxLayout, QDialog, QPushButton, QLabel

# Globale Variable für persistente Shell-Sitzung
active_process = None

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

def manage_jarvis_gui(current_model, current_view_mode="standard"):
    """Öffnet das GUI-Fenster in einem sicheren Unterprozess."""
    q = Queue()
    p = Process(target=_gui_worker, args=(q, "manage_jarvis_gui", current_model, current_view_mode))
    p.start()
    p.join()
    return q.get()

def _manage_jarvis_gui_logic(current_model, current_view_mode="standard"):
    """Die eigentliche Logik der Einstellungs-GUI."""
    dialog = QDialog()
    dialog.setWindowTitle("JARVIS System Control")
    dialog.setMinimumWidth(400)
    
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
        QComboBox, QCheckBox {
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
    layout.addWidget(QLabel("Ollama Model Selection:"))
    model_combo = QComboBox()
    try:
        local_models_resp = ollama.list()
        if hasattr(local_models_resp, 'models'):
            models = [m.model for m in local_models_resp.models]
        else:
            models = [m['name'] for m in local_models_resp.get('models', [])]
    except:
        models = [current_model]
    
    model_combo.addItems(sorted(list(set(models))))
    if current_model in models:
        model_combo.setCurrentText(current_model)
    layout.addWidget(model_combo)
    
    layout.addWidget(QLabel("System View Mode:"))
    view_mode_combo = QComboBox()
    view_mode_combo.addItems(["standard", "debug"])
    view_mode_combo.setCurrentText(current_view_mode)
    layout.addWidget(view_mode_combo)
    
    save_btn = QPushButton("APPLY CONFIGURATION")
    layout.addWidget(save_btn)
    
    settings = {}
    def save():
        settings['model'] = model_combo.currentText()
        settings['view_mode'] = view_mode_combo.currentText()
        dialog.accept()
        
    save_btn.clicked.connect(save)
    dialog.setLayout(layout)
    dialog.exec_()
    
    if settings:
        return json.dumps({"type": "config_update", "data": settings})
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
    print(f"[Tool Execution] Führe Befehl aus: '{command}'")
    
    blocked_keywords = ["rm -rf /", "mkfs", "dd "]
    for keyword in blocked_keywords:
        if keyword in command:
            return f"Befehl blockiert: Enthält potenziell gefährliches Schlüsselwort '{keyword}'."
            
    cmd_parts = command.split()
    base_cmd = cmd_parts[0]
    
    # Beende vorherigen aktiven Prozess, falls einer noch läuft
    if active_process and active_process.isalive():
        active_process.terminate(force=True)
    
    # Prüfe ob es ein interaktiver Befehl ist (Installation, Updates, sudo)
    interactive_commands = ["sudo", "pacman", "yay", "paru", "apt"]
    is_interactive = any(cmd in cmd_parts for cmd in interactive_commands)
    
    if is_interactive:
        try:
            # Starte persistenten Shell-Prozess
            active_process = pexpect.spawn(command, encoding='utf-8', timeout=5)
            return read_process_output()
        except Exception as e:
            return f"Fehler beim Starten des interaktiven Befehls: {e}"
    else:
        # Prüfen, ob es ein grafisches Programm ist oder ein Konsolenbefehl
        gui_apps = ["firefox", "code", "vlc", "nautilus", "virt-manager", "ptyxis", "gnome-terminal"]
        is_gui = any(app in command.lower() for app in gui_apps)
        
        if is_gui:
            try:
                subprocess.Popen(cmd_parts, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return f"Programm erfolgreich gestartet: {command} (läuft im Hintergrund)"
            except Exception as e:
                return f"Fehler beim Starten des Programms: {e}"
        else:
            # Normaler Konsolenbefehl: Output abfangen und zurückgeben
            try:
                result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
                output = result.stdout.strip()
                error = result.stderr.strip()
                if output:
                    return output
                if error:
                    return f"Fehler-Output: {error}"
                return "Befehl wurde ohne Rückgabe ausgeführt."
            except Exception as e:
                return f"Fehler bei der Ausführung: {e}"

def send_input(text: str) -> str:
    """Sendet Text an den wartenden Installer/Prozess im Terminal (z.B. 'y', '1', 'a')."""
    global active_process
    print(f"[Tool Execution] Sende Eingabe an Terminal: '{text}'")
    if not active_process or not active_process.isalive():
        return "Fehler: Es läuft aktuell kein interaktiver Befehl, der auf Eingaben wartet."
    
    try:
        active_process.sendline(text)
        return read_process_output()
    except Exception as e:
        return f"Fehler beim Senden der Eingabe: {e}"

def read_process_output() -> str:
    """Liest den Output des Terminals, bis es pausiert (Passwortabfrage, User Input oder Ende)."""
    global active_process
    output = ""
    
    # Regelmäßige Ausdrücke für typische Paketmanager/Installer Prompts in Linux (yay, pacman, apt)
    # y/n, Y/n, yes/no, [1 2 3], :: etc.
    input_prompts = [
        '(?i)password', 
        '(?i)passwort', 
        r'\[Y/n\]', 
        r'\[y/N\]', 
        r'\[y/n\]', 
        r'\? \[Y/n\]',
        r'==> .*:',
        r':: .*:',
        r'\? \[y/N\]',
        pexpect.EOF
    ]
    
    try:
        # Wir warten auf typische Prompts (Regex oben).
        # Wir setzen ein langes Timeout für große Downloads (z.B. kernel headers oder große git repos).
        index = active_process.expect(input_prompts, timeout=600)
        
        if index == 0 or index == 1: # Passwort prompt gefunden
            output += active_process.before + str(active_process.after)
            print("[System]: Passwort-Anfrage erkannt. Öffne GUI-Fenster...")
            pwd = get_gui_password()
            if pwd:
                active_process.sendline(pwd)
                output += "\n[System: Passwort wurde eingegeben, lese weiter...]\n"
                return read_process_output()
            else:
                active_process.sendline("\x03") # Ctrl+C
                return output + "\n[System: Passwort-Eingabe wurde vom Benutzer abgebrochen.]"
                
        elif index == len(input_prompts) - 1: # EOF -> Programm beendet
            output += active_process.before
            return output + "\n\n[PROGRAMM BEENDET]"
            
        else: # Eines der Auswahl-Prompts (==>, ::, [Y/n]) wurde gefunden
            # AUTO-CONFIRM für den Agenten: Wir senden 'y' und lesen weiter
            current_prompt = str(active_process.after)
            output += active_process.before + current_prompt
            print(f"[System]: Auto-Bestätigung für Prompt: {current_prompt.strip()}")
            active_process.sendline("y")
            return output + "\n[System: Auto-Bestätigung gesendet, lese weiter...]\n" + read_process_output()
            
    except pexpect.TIMEOUT:
        output += active_process.before
        return output + "\n\n[System: Timeout (10 Min). Der Prozess antwortet nicht mehr oder erzeugt keine Ausgabe mehr.]"
    except Exception as e:
        return output + f"\n[Fehler beim Lesen des Outputs: {e}]"

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
            return manage_jarvis_gui(kwargs.get("current_model"))
        elif tool_name == "update_config_direct":
            return update_config_direct(kwargs.get("key"), kwargs.get("value"))
        else:
            return f"Unbekanntes Tool: {tool_name}"
    except json.JSONDecodeError:
        return "Fehler: Die Tool-Anfrage war kein gültiges JSON."

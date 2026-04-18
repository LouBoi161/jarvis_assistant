import sys
import os
import threading
import time
import json
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QFrame, QGraphicsDropShadowEffect,
                             QPushButton, QScrollArea, QSizePolicy)
from PyQt5.QtCore import Qt, pyqtSignal, QPropertyAnimation, QRect, QEasingCurve, QThread, QPoint, QSize
from PyQt5.QtGui import QColor, QFont, QIcon

# Importiere den bestehenden Assistant
from main import JarvisAssistant
from agent_tools import manage_jarvis_gui

class AssistantThread(QThread):
    """Thread, der den JarvisAssistant im Hintergrund ausführt."""
    status_changed = pyqtSignal(str) 
    text_received = pyqtSignal(str, str) # sender, text
    
    def __init__(self, assistant):
        super().__init__()
        self.assistant = assistant
        self.assistant.log = self.custom_log
        
    def custom_log(self, message, level="debug"):
        # Originales Log-Verhalten beibehalten
        if self.assistant.view_mode == "debug" or level == "standard":
            print(message)
            
        if level == "standard":
            # Extrahiere Sender und Text
            if "Du (" in message:
                sender = "User"
                text = message.split("):")[1].strip()
                self.text_received.emit(sender, text)
            elif "[" in message and "]" in message:
                sender = "Jarvis"
                text = message.split("]")[1].strip()
                self.text_received.emit(sender, text)
            else:
                self.text_received.emit("System", message)

    def run(self):
        self.assistant.run_voice_only() # Wir brauchen eine non-blocking Version von run()

class MessageWidget(QFrame):
    """Ein einzelnes Nachrichten-Element in der History."""
    def __init__(self, sender, text):
        super().__init__()
        self.layout = QVBoxLayout(self)
        self.setContentsMargins(5, 5, 5, 5)
        
        color = "#00d4ff" if sender == "Jarvis" else "#ffffff"
        if sender == "System": color = "#aaaaaa"
        
        self.sender_label = QLabel(sender.upper())
        self.sender_label.setStyleSheet(f"color: {color}; font-size: 10px; font-weight: bold;")
        
        self.text_label = QLabel(text)
        self.text_label.setWordWrap(True)
        self.text_label.setStyleSheet("color: #e0e0e0; font-size: 13px;")
        
        self.layout.addWidget(self.sender_label)
        self.layout.addWidget(self.text_label)
        self.setStyleSheet("background-color: rgba(255, 255, 255, 10); border-radius: 5px; margin-bottom: 5px;")

class JarvisGUI(QWidget):
    def __init__(self, assistant):
        super().__init__()
        self.assistant = assistant
        self.is_expanded = False
        self.init_ui()
        
    def init_ui(self):
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Main Layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        self.main_layout.setSpacing(0)
        
        # --- 1. HISTORY AREA (Zuerst unsichtbar) ---
        self.history_scroll = QScrollArea()
        self.history_scroll.setWidgetResizable(True)
        self.history_scroll.setFixedWidth(780)
        self.history_scroll.setMaximumHeight(0) # Startet eingeklappt
        self.history_scroll.setStyleSheet("""
            QScrollArea {
                background-color: rgba(11, 14, 20, 240);
                border: 2px solid #00d4ff;
                border-bottom: none;
                border-top-left-radius: 15px;
                border-top-right-radius: 15px;
            }
            QScrollBar:vertical {
                border: none;
                background: rgba(0,0,0,0);
                width: 5px;
            }
            QScrollBar::handle:vertical {
                background: #00d4ff;
                min-height: 20px;
                border-radius: 2px;
            }
        """)
        
        self.history_content = QWidget()
        self.history_layout = QVBoxLayout(self.history_content)
        self.history_layout.setAlignment(Qt.AlignTop)
        self.history_scroll.setWidget(self.history_content)
        self.main_layout.addWidget(self.history_scroll)
        
        # --- 2. MAIN BAR (Der schwebende Balken) ---
        self.bar_container = QFrame()
        self.bar_container.setObjectName("MainBar")
        self.bar_container.setFixedWidth(800)
        self.bar_container.setMinimumHeight(100)
        
        self.bar_style_normal = """
            QFrame#MainBar {
                background-color: rgba(11, 14, 20, 230);
                border: 2px solid #00d4ff;
                border-radius: 15px;
            }
        """
        self.bar_style_expanded = """
            QFrame#MainBar {
                background-color: rgba(11, 14, 20, 230);
                border: 2px solid #00d4ff;
                border-top: 1px solid rgba(0, 212, 255, 100);
                border-bottom-left-radius: 15px;
                border-bottom-right-radius: 15px;
            }
        """
        self.bar_container.setStyleSheet(self.bar_style_normal)
        
        # Schatten/Glow
        self.glow = QGraphicsDropShadowEffect()
        self.glow.setBlurRadius(25)
        self.glow.setColor(QColor(0, 212, 255, 100))
        self.glow.setOffset(0, 0)
        self.bar_container.setGraphicsEffect(self.glow)
        
        self.bar_layout = QVBoxLayout(self.bar_container)
        
        # Header (Status & Buttons)
        self.header_layout = QHBoxLayout()
        
        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet("color: #00d4ff; font-size: 14px;")
        
        self.status_label = QLabel("JARVIS IDLE")
        self.status_label.setStyleSheet("color: #00d4ff; font-size: 10px; font-weight: bold; letter-spacing: 2px;")
        
        self.header_layout.addWidget(self.status_dot)
        self.header_layout.addWidget(self.status_label)
        self.header_layout.addStretch()
        
        # Settings Button
        self.settings_btn = QPushButton("⚙")
        self.settings_btn.setFixedSize(30, 30)
        self.settings_btn.setCursor(Qt.PointingHandCursor)
        self.settings_btn.setStyleSheet("background: transparent; color: #00d4ff; font-size: 20px; border: none;")
        self.settings_btn.clicked.connect(self.open_settings)
        self.header_layout.addWidget(self.settings_btn)
        
        # History Toggle Button
        self.expand_btn = QPushButton("▼")
        self.expand_btn.setFixedSize(30, 30)
        self.expand_btn.setCursor(Qt.PointingHandCursor)
        self.expand_btn.setStyleSheet("background: transparent; color: #00d4ff; font-size: 16px; border: none;")
        self.expand_btn.clicked.connect(self.toggle_history)
        self.header_layout.addWidget(self.expand_btn)
        
        self.bar_layout.addLayout(self.header_layout)
        
        # Output Area
        self.output_label = QLabel("System initialisiert. Bereit für Befehle.")
        self.output_label.setWordWrap(True)
        self.output_label.setAlignment(Qt.AlignCenter)
        self.output_label.setStyleSheet("color: white; font-size: 16px; padding: 5px; font-weight: 300;")
        self.bar_layout.addWidget(self.output_label)
        
        # Input Area
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Befehl eingeben...")
        self.input_field.setStyleSheet("""
            QLineEdit {
                background-color: rgba(0,0,0,40);
                border: 1px solid rgba(0, 212, 255, 60);
                border-radius: 5px;
                color: #00d4ff;
                font-size: 14px;
                padding: 8px;
            }
        """)
        self.input_field.setAlignment(Qt.AlignCenter)
        self.input_field.returnPressed.connect(self.process_text_input)
        self.bar_layout.addWidget(self.input_field)
        
        self.main_layout.addWidget(self.bar_container)
        self.setLayout(self.main_layout)
        
        # Zentrieren und Positionieren
        screen = QApplication.primaryScreen().geometry()
        self.move(screen.width() // 2 - 400, screen.height() - 200)
        
        # Animations
        self.history_anim = QPropertyAnimation(self.history_scroll, b"maximumHeight")
        self.history_anim.setDuration(400)
        self.history_anim.setEasingCurve(QEasingCurve.OutQuint)

    def toggle_history(self):
        if not self.is_expanded:
            self.history_anim.setStartValue(0)
            self.history_anim.setEndValue(400)
            self.bar_container.setStyleSheet(self.bar_style_expanded)
            self.expand_btn.setText("▲")
            self.is_expanded = True
        else:
            self.history_anim.setStartValue(400)
            self.history_anim.setEndValue(0)
            self.bar_container.setStyleSheet(self.bar_style_normal)
            self.expand_btn.setText("▼")
            self.is_expanded = False
        self.history_anim.start()

    def update_status(self, status):
        status_map = {
            "idle": ("#00d4ff", "JARVIS IDLE"),
            "listening": ("#ff0064", "JARVIS LISTENING..."),
            "thinking": ("#ffffff", "JARVIS THINKING..."),
            "speaking": ("#00ff96", "JARVIS SPEAKING")
        }
        color_hex, text = status_map.get(status, status_map["idle"])
        self.status_dot.setStyleSheet(f"color: {color_hex}; font-size: 14px;")
        self.status_label.setText(text)
        self.glow.setColor(QColor(color_hex))

    def display_text(self, sender, text):
        # In den Haupt-Balken schreiben (wenn relevant)
        if sender in ["Jarvis", "User"]:
            self.output_label.setText(text)
        
        # In die History einfügen
        msg = MessageWidget(sender, text)
        self.history_layout.addWidget(msg)
        
        # Automatisch nach unten scrollen
        time.sleep(0.05)
        self.history_scroll.verticalScrollBar().setValue(self.history_scroll.verticalScrollBar().maximum())

    def process_text_input(self):
        text = self.input_field.text()
        if text:
            self.display_text("User", text)
            self.input_field.clear()
            if hasattr(self, 'at'):
                threading.Thread(target=lambda: self.at.assistant.run_ollama_agent(text), daemon=True).start()

    def open_settings(self):
        # Nutzt das bestehende GUI-Tool aus agent_tools
        config_data = {
            "ollama_model": self.assistant.ollama_model,
            "view_mode": self.assistant.view_mode,
            "security_mode": self.assistant.security_mode,
            "language": self.assistant.language,
            "tts_type": self.assistant.tts_type,
            "piper_voice": self.assistant.piper_voice,
            "qwen_voice": self.assistant.qwen_voice
        }
        new_config_json = manage_jarvis_gui(config_data)
        if new_config_json and "config_update" in new_config_json:
            update_data = json.loads(new_config_json)["data"]
            self.assistant.update_config(update_data)
            self.display_text("System", "Konfiguration aktualisiert.")

    def closeEvent(self, event):
        if hasattr(self, 'at'):
            self.at.terminate()
            self.at.wait()
        os.killpg(0, 9) # Radikaler Cleanup
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    assistant = JarvisAssistant()
    gui = JarvisGUI(assistant)
    
    at = AssistantThread(assistant)
    at.text_received.connect(gui.display_text)
    at.status_changed.connect(gui.update_status)
    assistant.on_status_change = lambda s: at.status_changed.emit(s)
    
    gui.at = at
    at.start()
    gui.show()
    sys.exit(app.exec_())

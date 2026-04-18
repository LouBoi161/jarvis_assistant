import sys
import os
import threading
import time
import json
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QFrame, QGraphicsDropShadowEffect,
                             QPushButton, QScrollArea, QStackedWidget, QSizePolicy,
                             QCheckBox, QComboBox, QSpacerItem)
from PyQt5.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve, QThread, QSize
from PyQt5.QtGui import QColor, QFont, QIcon, QPainter

# Importiere den bestehenden Assistant
from main import JarvisAssistant
import ollama

class AssistantThread(QThread):
    """Thread, der den JarvisAssistant im Hintergrund ausführt."""
    status_changed = pyqtSignal(str) 
    text_received = pyqtSignal(str, str) # sender, text
    
    def __init__(self, assistant):
        super().__init__()
        self.assistant = assistant
        self.assistant.log = self.custom_log
        
    def custom_log(self, message, level="debug"):
        if level == "standard":
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
        self.assistant.run_voice_only()

class ChatBubble(QFrame):
    """Eine moderne Chat-Bubble."""
    def __init__(self, sender, text):
        super().__init__()
        layout = QVBoxLayout(self)
        self.setContentsMargins(10, 10, 10, 10)
        
        is_jarvis = (sender == "Jarvis")
        align = Qt.AlignLeft if is_jarvis else Qt.AlignRight
        bg_color = "rgba(0, 212, 255, 30)" if is_jarvis else "rgba(255, 255, 255, 15)"
        border_color = "#00d4ff" if is_jarvis else "rgba(255, 255, 255, 40)"
        
        self.sender_label = QLabel(sender.upper())
        self.sender_label.setStyleSheet(f"color: {border_color}; font-size: 10px; font-weight: bold; background: transparent;")
        
        self.text_label = QLabel(text)
        self.text_label.setWordWrap(True)
        self.text_label.setStyleSheet("color: #e0e0e0; font-size: 14px; background: transparent; border: none;")
        self.text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        layout.addWidget(self.sender_label)
        layout.addWidget(self.text_label)
        
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_color};
                border: 1px solid {border_color};
                border-radius: 12px;
                margin: 5px;
            }}
        """)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

class JarvisGUI(QWidget):
    def __init__(self, assistant):
        super().__init__()
        self.assistant = assistant
        self.init_ui()
        
    def init_ui(self):
        self.setWindowTitle("JARVIS AI Assistant")
        self.resize(1000, 800)
        self.setMinimumSize(600, 500)
        
        # Dark Theme Palette
        self.setStyleSheet("""
            QWidget {
                background-color: #0b0e14;
                color: #e0e0e0;
                font-family: 'Segoe UI', 'Roboto', sans-serif;
            }
            QScrollBar:vertical {
                border: none;
                background: #0b0e14;
                width: 8px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #1a1f29;
                min-height: 20px;
                border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #00d4ff;
            }
        """)
        
        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # --- 1. SIDEBAR ---
        self.sidebar = QFrame()
        self.sidebar.setFixedWidth(70)
        self.sidebar.setStyleSheet("background-color: #07090d; border-right: 1px solid #1a1f29;")
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setAlignment(Qt.AlignTop | Qt.AlignCenter)
        
        self.btn_chat = QPushButton("💬")
        self.btn_settings = QPushButton("⚙")
        
        for btn in [self.btn_chat, self.btn_settings]:
            btn.setFixedSize(50, 50)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet("""
                QPushButton {
                    background: transparent;
                    color: #00d4ff;
                    font-size: 24px;
                    border-radius: 10px;
                    margin-bottom: 10px;
                }
                QPushButton:hover {
                    background: rgba(0, 212, 255, 20);
                }
                QPushButton:checked {
                    background: rgba(0, 212, 255, 40);
                    border: 1px solid #00d4ff;
                }
            """)
            btn.setCheckable(True)
        
        self.btn_chat.setChecked(True)
        self.btn_chat.clicked.connect(lambda: self.switch_page(0))
        self.btn_settings.clicked.connect(lambda: self.switch_page(1))
        
        sidebar_layout.addWidget(self.btn_chat)
        sidebar_layout.addWidget(self.btn_settings)
        self.main_layout.addWidget(self.sidebar)
        
        # --- 2. MAIN STACKED WIDGET ---
        self.stack = QStackedWidget()
        
        # PAGE 0: CHAT
        self.chat_page = QWidget()
        chat_layout = QVBoxLayout(self.chat_page)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        
        # Chat Header
        self.chat_header = QFrame()
        self.chat_header.setFixedHeight(60)
        self.chat_header.setStyleSheet("background-color: #0b0e14; border-bottom: 1px solid #1a1f29;")
        header_layout = QHBoxLayout(self.chat_header)
        
        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet("color: #00d4ff; font-size: 18px; margin-left: 10px;")
        self.status_text = QLabel("JARVIS ONLINE")
        self.status_text.setStyleSheet("font-weight: bold; letter-spacing: 1px; color: #00d4ff;")
        
        header_layout.addWidget(self.status_dot)
        header_layout.addWidget(self.status_text)
        header_layout.addStretch()
        chat_layout.addWidget(self.chat_header)
        
        # Chat Scroll Area
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setStyleSheet("border: none; background-color: transparent;")
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setAlignment(Qt.AlignTop)
        self.scroll_layout.setSpacing(10)
        self.scroll.setWidget(self.scroll_content)
        chat_layout.addWidget(self.scroll)
        
        # Input Area
        self.input_container = QFrame()
        self.input_container.setFixedHeight(80)
        self.input_container.setStyleSheet("background-color: #0b0e14; border-top: 1px solid #1a1f29;")
        input_layout = QHBoxLayout(self.input_container)
        
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Frag Jarvis etwas...")
        self.input_field.setStyleSheet("""
            QLineEdit {
                background-color: #1a1f29;
                border: 1px solid #333;
                border-radius: 20px;
                color: white;
                font-size: 15px;
                padding: 10px 20px;
            }
            QLineEdit:focus {
                border: 1px solid #00d4ff;
            }
        """)
        self.input_field.returnPressed.connect(self.process_text_input)
        
        self.send_btn = QPushButton("➤")
        self.send_btn.setFixedSize(40, 40)
        self.send_btn.setStyleSheet("""
            QPushButton {
                background-color: #00d4ff;
                color: #0b0e14;
                border-radius: 20px;
                font-size: 20px;
            }
            QPushButton:hover {
                background-color: #00b8e6;
            }
        """)
        self.send_btn.clicked.connect(self.process_text_input)
        
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_btn)
        chat_layout.addWidget(self.input_container)
        
        self.stack.addWidget(self.chat_page)
        
        # PAGE 1: SETTINGS
        self.settings_page = QWidget()
        settings_layout = QVBoxLayout(self.settings_page)
        settings_layout.setContentsMargins(40, 40, 40, 40)
        
        title = QLabel("System-Einstellungen")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #00d4ff; margin-bottom: 20px;")
        settings_layout.addWidget(title)
        
        # Form
        self.form_frame = QFrame()
        self.form_frame.setStyleSheet("background-color: #11151c; border-radius: 15px; padding: 20px;")
        form_layout = QVBoxLayout(self.form_frame)
        
        # Ollama Model
        form_layout.addWidget(QLabel("Ollama Model:"))
        self.model_combo = QComboBox()
        form_layout.addWidget(self.model_combo)
        
        # TTS Type
        form_layout.addWidget(QLabel("TTS Engine:"))
        self.tts_combo = QComboBox()
        self.tts_combo.addItems(["qwen3-tts", "piper-tts", "none"])
        form_layout.addWidget(self.tts_combo)
        
        # Language
        form_layout.addWidget(QLabel("Sprache:"))
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["de", "en", "auto"])
        form_layout.addWidget(self.lang_combo)
        
        # Security
        self.sec_check = QCheckBox("Sicherheitsmodus (Befehle blockieren)")
        form_layout.addWidget(self.sec_check)
        
        form_layout.addStretch()
        
        # Save Button
        self.save_btn = QPushButton("EINSTELLUNGEN ÜBERNEHMEN")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #00d4ff;
                color: #0b0e14;
                border-radius: 5px;
                padding: 15px;
                font-weight: bold;
            }
        """)
        self.save_btn.clicked.connect(self.save_settings)
        form_layout.addWidget(self.save_btn)
        
        settings_layout.addWidget(self.form_frame)
        settings_layout.addStretch()
        
        self.stack.addWidget(self.settings_page)
        
        self.main_layout.addWidget(self.stack)
        
        # Init Settings Data
        self.load_settings_into_ui()

    def switch_page(self, index):
        self.btn_chat.setChecked(index == 0)
        self.btn_settings.setChecked(index == 1)
        self.stack.setCurrentIndex(index)
        if index == 1:
            self.refresh_models()

    def load_settings_into_ui(self):
        self.tts_combo.setCurrentText(self.assistant.tts_type)
        self.lang_combo.setCurrentText(self.assistant.language)
        self.sec_check.setChecked(self.assistant.security_mode)
        self.refresh_models()

    def refresh_models(self):
        try:
            local_models_resp = ollama.list()
            if hasattr(local_models_resp, 'models'):
                models = [m.model for m in local_models_resp.models]
            else:
                models = [m['name'] for m in local_models_resp.get('models', [])]
            self.model_combo.clear()
            self.model_combo.addItems(sorted(list(set(models))))
            if self.assistant.ollama_model in models:
                self.model_combo.setCurrentText(self.assistant.ollama_model)
        except:
            pass

    def save_settings(self):
        new_data = {
            "ollama_model": self.model_combo.currentText(),
            "tts_type": self.tts_combo.currentText(),
            "language": self.lang_combo.currentText(),
            "security_mode": self.sec_check.isChecked()
        }
        self.assistant.update_config(new_data)
        self.display_text("System", "Einstellungen wurden gespeichert.")
        self.switch_page(0)

    def update_status(self, status):
        status_map = {
            "idle": ("#00d4ff", "JARVIS ONLINE"),
            "listening": ("#ff0064", "JARVIS HÖRT ZU..."),
            "thinking": ("#ffffff", "JARVIS DENKT NACH..."),
            "speaking": ("#00ff96", "JARVIS SPRICHT")
        }
        color_hex, text = status_map.get(status, status_map["idle"])
        self.status_dot.setStyleSheet(f"color: {color_hex}; font-size: 18px; margin-left: 10px;")
        self.status_text.setText(text)
        self.status_text.setStyleSheet(f"font-weight: bold; letter-spacing: 1px; color: {color_hex};")

    def display_text(self, sender, text):
        bubble = ChatBubble(sender, text)
        self.scroll_layout.addWidget(bubble)
        # Scroll to bottom
        threading.Timer(0.1, self.scroll_to_bottom).start()

    def scroll_to_bottom(self):
        self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum())

    def process_text_input(self):
        text = self.input_field.text().strip()
        if text:
            self.display_text("User", text)
            self.input_field.clear()
            if hasattr(self, 'at'):
                threading.Thread(target=lambda: self.at.assistant.run_ollama_agent(text), daemon=True).start()

    def closeEvent(self, event):
        if hasattr(self, 'at'):
            self.at.terminate()
            self.at.wait()
        os.killpg(0, 9)
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

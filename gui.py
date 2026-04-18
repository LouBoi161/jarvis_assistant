import sys
import os
import threading
import time
import json
import subprocess
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QFrame, QGraphicsDropShadowEffect,
                             QPushButton, QScrollArea, QStackedWidget, QSizePolicy,
                             QCheckBox, QComboBox, QSpacerItem, QSizeGrip)
from PyQt5.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve, QThread, QPoint, QSize
from PyQt5.QtGui import QColor, QFont, QIcon, QPainter, QCursor

# Importiere den bestehenden Assistant
from main import JarvisAssistant
import ollama

class AssistantThread(QThread):
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
    def __init__(self, sender, text):
        super().__init__()
        layout = QVBoxLayout(self)
        self.setContentsMargins(10, 10, 10, 10)
        
        is_jarvis = (sender == "Jarvis")
        bg_color = "rgba(0, 212, 255, 25)" if is_jarvis else "rgba(255, 255, 255, 10)"
        border_color = "#00d4ff" if is_jarvis else "rgba(255, 255, 255, 30)"
        
        self.sender_label = QLabel(sender.upper())
        self.sender_label.setStyleSheet(f"color: {border_color}; font-size: 10px; font-weight: bold; background: transparent; border: none;")
        
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

class CustomTitleBar(QFrame):
    """Eine moderne, dunkle Titelleiste."""
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setFixedHeight(40)
        self.setStyleSheet("background-color: #07090d; border-bottom: 1px solid #1a1f29;")
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 0, 5, 0)
        
        self.title = QLabel("JARVIS v2.0")
        self.title.setStyleSheet("color: #00d4ff; font-weight: bold; font-size: 12px; letter-spacing: 1px; border: none;")
        
        self.btn_min = QPushButton("─")
        self.btn_close = QPushButton("✕")
        
        for btn in [self.btn_min, self.btn_close]:
            btn.setFixedSize(30, 30)
            btn.setStyleSheet("""
                QPushButton {
                    background: transparent;
                    color: #555;
                    font-size: 14px;
                    border: none;
                }
                QPushButton:hover {
                    color: white;
                    background: rgba(255,255,255,10);
                }
            """)
        
        self.btn_close.setStyleSheet(self.btn_close.styleSheet() + "QPushButton:hover { background: #e81123; }")
        
        self.btn_min.clicked.connect(self.parent.showMinimized)
        self.btn_close.clicked.connect(self.parent.close)
        
        layout.addWidget(self.title)
        layout.addStretch()
        layout.addWidget(self.btn_min)
        layout.addWidget(self.btn_close)
        
        self.start_pos = None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pos = event.globalPos()

    def mouseMoveEvent(self, event):
        if self.start_pos:
            delta = event.globalPos() - self.start_pos
            self.parent.move(self.parent.x() + delta.x(), self.parent.y() + delta.y())
            self.start_pos = event.globalPos()

class JarvisGUI(QWidget):
    def __init__(self, assistant):
        super().__init__()
        self.assistant = assistant
        self.init_ui()
        
    def init_ui(self):
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.resize(1100, 850)
        
        # Stylesheet für das gesamte System
        self.setStyleSheet("""
            QWidget {
                background-color: #0b0e14;
                color: #e0e0e0;
                font-family: 'Segoe UI', sans-serif;
            }
            QLabel { border: none; }
            QComboBox {
                background-color: #1a1f29;
                border: 1px solid #333;
                border-radius: 5px;
                padding: 8px;
                color: white;
                min-width: 200px;
            }
            QComboBox:hover, QLineEdit:hover {
                border: 1px solid #00d4ff;
            }
            QComboBox::drop-down {
                border: none;
                width: 30px;
            }
            QComboBox QAbstractItemView {
                background-color: #1a1f29;
                border: 1px solid #00d4ff;
                selection-background-color: #00d4ff;
                selection-color: #0b0e14;
                outline: none;
            }
            QLineEdit {
                background-color: #1a1f29;
                border: 1px solid #333;
                border-radius: 5px;
                padding: 10px;
                color: white;
            }
            QCheckBox {
                spacing: 10px;
                font-size: 14px;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border: 1px solid #333;
                border-radius: 4px;
            }
            QCheckBox::indicator:checked {
                background-color: #00d4ff;
                border: 1px solid #00d4ff;
            }
            QScrollArea { border: none; background: transparent; }
        """)
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        
        # Titelleiste
        self.title_bar = CustomTitleBar(self)
        self.layout.addWidget(self.title_bar)
        
        # Content Layout (Sidebar + Stack)
        self.content_container = QHBoxLayout()
        self.content_container.setSpacing(0)
        
        # --- SIDEBAR ---
        self.sidebar = QFrame()
        self.sidebar.setFixedWidth(80)
        self.sidebar.setStyleSheet("background-color: #07090d; border-right: 1px solid #1a1f29;")
        sidebar_layout = QVBoxLayout(self.sidebar)
        sidebar_layout.setContentsMargins(0, 20, 0, 20)
        sidebar_layout.setAlignment(Qt.AlignTop | Qt.AlignCenter)
        
        self.btn_chat = QPushButton("💬")
        self.btn_settings = QPushButton("⚙")
        
        for btn in [self.btn_chat, self.btn_settings]:
            btn.setFixedSize(50, 50)
            btn.setCheckable(True)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet("""
                QPushButton {
                    background: transparent;
                    color: #444;
                    font-size: 24px;
                    border-radius: 12px;
                    margin-bottom: 15px;
                }
                QPushButton:hover { color: #00d4ff; background: rgba(0, 212, 255, 10); }
                QPushButton:checked { color: #0b0e14; background: #00d4ff; }
            """)
        
        self.btn_chat.setChecked(True)
        self.btn_chat.clicked.connect(lambda: self.switch_page(0))
        self.btn_settings.clicked.connect(lambda: self.switch_page(1))
        
        sidebar_layout.addWidget(self.btn_chat)
        sidebar_layout.addWidget(self.btn_settings)
        self.content_container.addWidget(self.sidebar)
        
        # --- STACKED WIDGET ---
        self.stack = QStackedWidget()
        
        # CHAT PAGE
        self.chat_page = QWidget()
        chat_l = QVBoxLayout(self.chat_page)
        chat_l.setContentsMargins(0, 0, 0, 0)
        
        # Status Bar
        self.status_bar = QFrame()
        self.status_bar.setFixedHeight(50)
        self.status_bar.setStyleSheet("background-color: #0b0e14; border-bottom: 1px solid #1a1f29;")
        sb_l = QHBoxLayout(self.status_bar)
        self.status_dot = QLabel("●")
        self.status_dot.setStyleSheet("color: #00d4ff; font-size: 16px; margin-left: 10px;")
        self.status_text = QLabel("JARVIS BEREIT")
        self.status_text.setStyleSheet("font-weight: bold; color: #00d4ff; font-size: 11px; letter-spacing: 1px;")
        sb_l.addWidget(self.status_dot)
        sb_l.addWidget(self.status_text)
        sb_l.addStretch()
        chat_l.addWidget(self.status_bar)
        
        # Chat History
        self.scroll = QScrollArea()
        self.scroll_content = QWidget()
        self.scroll_l = QVBoxLayout(self.scroll_content)
        self.scroll_l.setAlignment(Qt.AlignTop)
        self.scroll.setWidget(self.scroll_content)
        self.scroll.setWidgetResizable(True)
        chat_l.addWidget(self.scroll)
        
        # Input
        self.input_cont = QFrame()
        self.input_cont.setFixedHeight(100)
        self.input_cont.setStyleSheet("background: #0b0e14; border-top: 1px solid #1a1f29;")
        ic_l = QHBoxLayout(self.input_cont)
        self.input_f = QLineEdit()
        self.input_f.setPlaceholderText("Sprich mit Jarvis oder tippe hier...")
        self.input_f.returnPressed.connect(self.process_text_input)
        self.send_b = QPushButton("➤")
        self.send_b.setFixedSize(45, 45)
        self.send_b.setStyleSheet("background: #00d4ff; color: #0b0e14; border-radius: 22px; font-size: 18px;")
        self.send_b.clicked.connect(self.process_text_input)
        ic_l.addWidget(self.input_f)
        ic_l.addWidget(self.send_b)
        chat_l.addWidget(self.input_cont)
        
        self.stack.addWidget(self.chat_page)
        
        # SETTINGS PAGE
        self.settings_page = QWidget()
        set_l = QVBoxLayout(self.settings_page)
        set_l.setContentsMargins(50, 30, 50, 30)
        
        stitle = QLabel("KONFIGURATION")
        stitle.setStyleSheet("font-size: 22px; font-weight: bold; color: #00d4ff; margin-bottom: 20px;")
        set_l.addWidget(stitle)
        
        self.set_scroll = QScrollArea()
        self.set_scroll_content = QWidget()
        self.set_scroll_l = QVBoxLayout(self.set_scroll_content)
        self.set_scroll_l.setSpacing(20)
        
        # Settings Groups
        self.add_setting_group(self.set_scroll_l, "KI-MODELL", self.init_model_settings())
        self.add_setting_group(self.set_scroll_l, "SPRACHAUSGABE (TTS)", self.init_tts_settings())
        self.add_setting_group(self.set_scroll_l, "SYSTEM", self.init_system_settings())
        
        self.save_b = QPushButton("EINSTELLUNGEN SPEICHERN")
        self.save_b.setFixedHeight(55)
        self.save_b.setStyleSheet("background: #00d4ff; color: #0b0e14; font-weight: bold; border-radius: 8px;")
        self.save_b.clicked.connect(self.save_settings)
        self.set_scroll_l.addWidget(self.save_b)
        
        self.set_scroll.setWidget(self.set_scroll_content)
        self.set_scroll.setWidgetResizable(True)
        set_l.addWidget(self.set_scroll)
        
        self.stack.addWidget(self.settings_page)
        self.content_container.addWidget(self.stack)
        self.layout.addLayout(self.content_container)
        
        # Resize Grip
        self.sizegrip = QSizeGrip(self)
        self.layout.addWidget(self.sizegrip, 0, Qt.AlignBottom | Qt.AlignRight)
        
        self.load_settings_into_ui()

    def add_setting_group(self, parent_layout, title, widget):
        group = QFrame()
        group.setStyleSheet("background: #11151c; border-radius: 12px; padding: 15px; border: 1px solid #1a1f29;")
        l = QVBoxLayout(group)
        t = QLabel(title)
        t.setStyleSheet("color: #555; font-size: 10px; font-weight: bold; margin-bottom: 5px; border: none;")
        l.addWidget(t)
        l.addWidget(widget)
        parent_layout.addWidget(group)

    def init_model_settings(self):
        w = QWidget()
        l = QVBoxLayout(w)
        l.addWidget(QLabel("Ollama Modell:"))
        self.model_c = QComboBox()
        l.addWidget(self.model_c)
        return w

    def init_tts_settings(self):
        w = QWidget()
        l = QVBoxLayout(w)
        
        l.addWidget(QLabel("TTS Engine:"))
        self.tts_c = QComboBox()
        self.tts_c.addItems(["qwen3-tts", "piper-tts", "none"])
        l.addWidget(self.tts_c)
        
        self.piper_v_cont = QWidget()
        pv_l = QVBoxLayout(self.piper_v_cont)
        pv_l.addWidget(QLabel("Piper Stimme:"))
        self.piper_v_c = QComboBox()
        pv_l.addWidget(self.piper_v_c)
        l.addWidget(self.piper_v_cont)
        
        self.qwen_v_cont = QWidget()
        qv_l = QVBoxLayout(self.qwen_v_cont)
        qv_l.addWidget(QLabel("Qwen Klon (.wav):"))
        self.qwen_v_c = QComboBox()
        qv_l.addWidget(self.qwen_v_c)
        l.addWidget(self.qwen_v_cont)
        
        self.tts_c.currentTextChanged.connect(self.toggle_voice_fields)
        return w

    def init_system_settings(self):
        w = QWidget()
        l = QVBoxLayout(w)
        l.addWidget(QLabel("Sprache:"))
        self.lang_c = QComboBox()
        self.lang_c.addItems(["de", "en", "auto"])
        l.addWidget(self.lang_c)
        self.sec_c = QCheckBox("Sicherheitsmodus (Systemzugriff blockiert)")
        l.addWidget(self.sec_c)
        return w

    def toggle_voice_fields(self):
        self.piper_v_cont.setVisible(self.tts_c.currentText() == "piper-tts")
        self.qwen_v_cont.setVisible(self.tts_c.currentText() == "qwen3-tts")

    def switch_page(self, index):
        self.btn_chat.setChecked(index == 0)
        self.btn_settings.setChecked(index == 1)
        self.stack.setCurrentIndex(index)
        if index == 1: self.refresh_all_options()

    def load_settings_into_ui(self):
        self.tts_c.setCurrentText(self.assistant.tts_type)
        self.lang_c.setCurrentText(self.assistant.language)
        self.sec_c.setChecked(self.assistant.security_mode)
        self.toggle_voice_fields()
        self.refresh_all_options()

    def refresh_all_options(self):
        # Models
        try:
            m = [m.model for m in ollama.list().models]
            self.model_c.clear()
            self.model_c.addItems(sorted(list(set(m))))
            if self.assistant.ollama_model in m: self.model_c.setCurrentText(self.assistant.ollama_model)
        except: pass
        
        # Piper Voices
        p_dir = os.path.join(os.path.dirname(__file__), "piper_models")
        if os.path.exists(p_dir):
            v = [f.replace(".onnx", "") for f in os.listdir(p_dir) if f.endswith(".onnx")]
            self.piper_v_c.clear()
            self.piper_v_c.addItems(sorted(v))
            if self.assistant.piper_voice in v: self.piper_v_c.setCurrentText(self.assistant.piper_voice)
            
        # Qwen Voices
        q_dir = os.path.join(os.path.dirname(__file__), "voices")
        if os.path.exists(q_dir):
            v = [f for f in os.listdir(q_dir) if f.endswith(".wav")]
            self.qwen_v_c.clear()
            self.qwen_v_c.addItems(sorted(v))
            if self.assistant.qwen_voice in v: self.qwen_v_c.setCurrentText(self.assistant.qwen_voice)

    def save_settings(self):
        d = {
            "ollama_model": self.model_c.currentText(),
            "tts_type": self.tts_c.currentText(),
            "language": self.lang_c.currentText(),
            "security_mode": self.sec_c.isChecked(),
            "piper_voice": self.piper_v_c.currentText(),
            "qwen_voice": self.qwen_v_c.currentText()
        }
        self.assistant.update_config(d)
        self.display_text("System", "Konfiguration erfolgreich gespeichert.")
        self.switch_page(0)

    def update_status(self, status):
        m = {
            "idle": ("#00d4ff", "JARVIS BEREIT"),
            "listening": ("#ff0064", "JARVIS HÖRT ZU..."),
            "thinking": ("#ffffff", "JARVIS DENKT..."),
            "speaking": ("#00ff96", "JARVIS SPRICHT")
        }
        c, t = m.get(status, m["idle"])
        self.status_dot.setStyleSheet(f"color: {c}; font-size: 16px; margin-left: 10px;")
        self.status_text.setText(t)
        self.status_text.setStyleSheet(f"font-weight: bold; color: {c}; font-size: 11px;")

    def display_text(self, sender, text):
        b = ChatBubble(sender, text)
        self.scroll_l.addWidget(b)
        threading.Timer(0.1, lambda: self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum())).start()

    def process_text_input(self):
        t = self.input_f.text().strip()
        if t:
            self.display_text("User", t)
            self.input_f.clear()
            if hasattr(self, 'at'):
                threading.Thread(target=lambda: self.at.assistant.run_ollama_agent(t), daemon=True).start()

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

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
from PyQt5.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve, QThread, QPoint, QSize, QTimer
from PyQt5.QtGui import QColor, QFont, QIcon, QPainter, QCursor, QWindow

# Importiere den bestehenden Assistant
from main import JarvisAssistant
import ollama

class AssistantThread(QThread):
    status_changed = pyqtSignal(str) 
    text_received = pyqtSignal(str, str)
    
    def __init__(self, assistant):
        super().__init__()
        self.assistant = assistant
        self.assistant.log = self.custom_log
        
    def custom_log(self, message, level="debug"):
        if level == "standard":
            if "Du (" in message:
                sender = "User"
                parts = message.split("):")
                text = parts[1].strip() if len(parts) > 1 else ""
                if text: self.text_received.emit(sender, text)
            elif "[Jarvis]" in message:
                sender = "Jarvis"
                parts = message.split("]", 1)
                text = parts[1].strip().lstrip(":").strip() if len(parts) > 1 else ""
                if text: self.text_received.emit(sender, text)
            elif "TOOL:" in message:
                sender = "Tool"
                text = message.replace("TOOL:", "").strip()
                self.text_received.emit(sender, text)
            else:
                self.text_received.emit("System", message)

    def run(self):
        self.assistant.run_voice_only()

class ChatBubble(QFrame):
    def __init__(self, sender, text, is_temporary=False):
        super().__init__()
        self.sender = sender
        self.is_temporary = is_temporary
        layout = QVBoxLayout(self)
        self.setContentsMargins(15, 10, 15, 10)
        
        is_jarvis = (sender == "Jarvis")
        is_tool = (sender == "Tool")
        
        if is_tool:
            bg_color = "rgba(0, 212, 255, 15)"
            border_color = "rgba(0, 212, 255, 30)"
            text_color = "#00d4ff"
            font_style = "italic"
            font_size = "12px"
        elif is_jarvis:
            bg_color = "rgba(0, 212, 255, 20)"
            border_color = "#00d4ff"
            text_color = "#e0e0e0"
            font_style = "normal"
            font_size = "14px"
        else: # User
            bg_color = "rgba(255, 255, 255, 8)"
            border_color = "#444"
            text_color = "#e0e0e0"
            font_style = "normal"
            font_size = "14px"
        
        if not is_tool:
            self.sender_label = QLabel(sender.upper())
            self.sender_label.setStyleSheet(f"color: {border_color}; font-size: 10px; font-weight: bold; background: transparent; border: none;")
            layout.addWidget(self.sender_label)
        
        self.text_label = QLabel(text)
        self.text_label.setWordWrap(True)
        self.text_label.setStyleSheet(f"color: {text_color}; font-size: {font_size}; font-style: {font_style}; background: transparent; border: none;")
        self.text_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        layout.addWidget(self.text_label)
        self.setStyleSheet(f"background-color: {bg_color}; border: 1px solid {border_color}; border-radius: 12px; margin: 5px;")

class CustomTitleBar(QFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setFixedHeight(45)
        self.setStyleSheet("background-color: #07090d; border-bottom: 1px solid #1a1f29;")
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 0, 5, 0)
        self.title = QLabel("JARVIS v2.3")
        self.title.setStyleSheet("color: #00d4ff; font-weight: bold; font-size: 12px; letter-spacing: 2px; border: none;")
        self.btn_min = QPushButton("─")
        self.btn_max = QPushButton("◻")
        self.btn_close = QPushButton("✕")
        for btn in [self.btn_min, self.btn_max, self.btn_close]:
            btn.setFixedSize(40, 45)
            btn.setStyleSheet("QPushButton { background: transparent; color: #666; font-size: 16px; border: none; } QPushButton:hover { color: white; background: rgba(255,255,255,10); }")
        self.btn_close.setStyleSheet(self.btn_close.styleSheet() + "QPushButton:hover { background: #e81123; }")
        self.btn_min.clicked.connect(self.parent.showMinimized)
        self.btn_max.clicked.connect(self.toggle_maximize)
        self.btn_close.clicked.connect(self.parent.close)
        layout.addWidget(self.title); layout.addStretch(); layout.addWidget(self.btn_min); layout.addWidget(self.btn_max); layout.addWidget(self.btn_close)
    def toggle_maximize(self):
        if self.parent.isMaximized(): self.parent.showNormal(); self.btn_max.setText("◻")
        else: self.parent.showMaximized(); self.btn_max.setText("❐")
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton: self.window().windowHandle().startSystemMove()

class JarvisGUI(QWidget):
    def __init__(self, assistant):
        super().__init__()
        self.assistant = assistant
        self.temp_tool_bubbles = []
        self.init_ui()
    def init_ui(self):
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.resize(1100, 850)
        self.setStyleSheet("""
            QWidget { background-color: #0b0e14; color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }
            QComboBox { background-color: #1a1f29; border: 2px solid #333; border-radius: 8px; padding: 10px; color: white; font-size: 14px; combobox-popup: 0; }
            QComboBox:hover { border: 2px solid #00d4ff; }
            QComboBox QAbstractItemView { background-color: #1a1f29; border: 2px solid #333; border: 1px solid #00d4ff; selection-background-color: #00d4ff; selection-color: #0b0e14; color: #e0e0e0; outline: none; }
            QLineEdit { background-color: #1a1f29; border: 2px solid #333; border-radius: 8px; padding: 12px; color: white; font-size: 14px; }
            QLineEdit:focus { border: 2px solid #00d4ff; }
            QCheckBox { spacing: 12px; font-size: 14px; }
            QCheckBox::indicator { width: 22px; height: 22px; border: 2px solid #333; border-radius: 6px; }
            QCheckBox::indicator:checked { background-color: #00d4ff; border: 2px solid #00d4ff; }
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical { border: none; background: #0b0e14; width: 10px; }
            QScrollBar::handle:vertical { background: #1a1f29; border-radius: 5px; }
            QScrollBar::handle:vertical:hover { background: #00d4ff; }
        """)
        self.layout = QVBoxLayout(self); self.layout.setContentsMargins(0, 0, 0, 0); self.layout.setSpacing(0)
        self.title_bar = CustomTitleBar(self); self.layout.addWidget(self.title_bar)
        self.content_container = QHBoxLayout(); self.content_container.setSpacing(0)
        self.sidebar = QFrame(); self.sidebar.setFixedWidth(90); self.sidebar.setStyleSheet("background-color: #07090d; border-right: 1px solid #1a1f29;")
        sidebar_layout = QVBoxLayout(self.sidebar); sidebar_layout.setContentsMargins(10, 30, 10, 20); sidebar_layout.setAlignment(Qt.AlignTop | Qt.AlignCenter)
        self.btn_chat = QPushButton("💬"); self.btn_settings = QPushButton("⚙")
        for btn in [self.btn_chat, self.btn_settings]:
            btn.setFixedSize(70, 65); btn.setCheckable(True); btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet("QPushButton { background: #11151c; color: #555; font-size: 30px; border: 1px solid #222; border-radius: 15px; margin-bottom: 25px; } QPushButton:hover { color: #00d4ff; background: #1a1f29; border: 1px solid #00d4ff; } QPushButton:checked { color: #0b0e14; background: #00d4ff; border: 1px solid #00d4ff; }")
        self.btn_chat.setChecked(True); self.btn_chat.clicked.connect(lambda: self.switch_page(0))
        self.btn_settings.clicked.connect(lambda: self.switch_page(1))
        sidebar_layout.addWidget(self.btn_chat); sidebar_layout.addWidget(self.btn_settings); self.content_container.addWidget(self.sidebar)
        self.stack = QStackedWidget()
        self.chat_page = QWidget(); chat_l = QVBoxLayout(self.chat_page); chat_l.setContentsMargins(0, 0, 0, 0)
        self.status_bar = QFrame(); self.status_bar.setFixedHeight(55); self.status_bar.setStyleSheet("background-color: #0b0e14; border-bottom: 1px solid #1a1f29;")
        sb_l = QHBoxLayout(self.status_bar); self.status_dot = QLabel("●"); self.status_dot.setStyleSheet("color: #00d4ff; font-size: 20px; margin-left: 20px;")
        self.status_text = QLabel("JARVIS ONLINE"); self.status_text.setStyleSheet("font-weight: bold; color: #00d4ff; font-size: 13px; letter-spacing: 2px;")
        sb_l.addWidget(self.status_dot); sb_l.addWidget(self.status_text); sb_l.addStretch(); chat_l.addWidget(self.status_bar)
        self.scroll = QScrollArea(); self.scroll_content = QWidget(); self.scroll_l = QVBoxLayout(self.scroll_content); self.scroll_l.setAlignment(Qt.AlignTop); self.scroll_l.setContentsMargins(40, 20, 40, 20); self.scroll_l.setSpacing(20)
        self.scroll.setWidget(self.scroll_content); self.scroll.setWidgetResizable(True); chat_l.addWidget(self.scroll)
        self.input_cont = QFrame(); self.input_cont.setFixedHeight(120); self.input_cont.setStyleSheet("background: #0b0e14; border-top: 1px solid #1a1f29;")
        ic_l = QHBoxLayout(self.input_cont); ic_l.setContentsMargins(40, 0, 40, 0); self.input_f = QLineEdit(); self.input_f.setPlaceholderText("Frag Jarvis etwas..."); self.input_f.setFixedHeight(55); self.input_f.returnPressed.connect(self.process_text_input)
        self.send_b = QPushButton("➤"); self.send_b.setFixedSize(55, 55); self.send_b.setStyleSheet("background: #00d4ff; color: #0b0e14; border-radius: 27px; font-size: 24px;"); self.send_b.clicked.connect(self.process_text_input)
        ic_l.addWidget(self.input_f); ic_l.addWidget(self.send_b); chat_l.addWidget(self.input_cont); self.stack.addWidget(self.chat_page)
        self.settings_page = QWidget(); set_l = QVBoxLayout(self.settings_page); set_l.setContentsMargins(60, 40, 60, 40); stitle = QLabel("KONFIGURATION"); stitle.setStyleSheet("font-size: 28px; font-weight: bold; color: #00d4ff; margin-bottom: 30px;"); set_l.addWidget(stitle)
        self.set_scroll = QScrollArea(); self.set_scroll_content = QWidget(); self.set_scroll_l = QVBoxLayout(self.set_scroll_content); self.set_scroll_l.setSpacing(30)
        self.add_setting_group(self.set_scroll_l, "KI-LOGIK (OLLAMA)", self.init_model_settings())
        self.add_setting_group(self.set_scroll_l, "SPRACHAUSGABE (TTS)", self.init_tts_settings())
        self.add_setting_group(self.set_scroll_l, "SYSTEM-BERECHTIGUNGEN", self.init_system_settings())
        self.save_b = QPushButton("EINSTELLUNGEN ÜBERNEHMEN"); self.save_b.setFixedHeight(65); self.save_b.setCursor(Qt.PointingHandCursor); self.save_b.setStyleSheet("QPushButton { background: #00d4ff; color: #0b0e14; font-weight: bold; font-size: 15px; border-radius: 12px; margin-top: 20px; } QPushButton:hover { background: #00b8e6; }")
        self.save_b.clicked.connect(self.save_settings); self.set_scroll_l.addWidget(self.save_b); self.set_scroll.setWidget(self.set_scroll_content); self.set_scroll.setWidgetResizable(True); set_l.addWidget(self.set_scroll); self.stack.addWidget(self.settings_page)
        self.content_container.addWidget(self.stack); self.layout.addLayout(self.content_container); self.sizegrip = QSizeGrip(self); self.layout.addWidget(self.sizegrip, 0, Qt.AlignBottom | Qt.AlignRight); self.load_settings_into_ui()
    def add_setting_group(self, parent_layout, title, widget):
        group = QFrame(); group.setStyleSheet("background: #0f131a; border-radius: 15px; padding: 30px; border: 2px solid #1a1f29;"); l = QVBoxLayout(group); t = QLabel(title); t.setStyleSheet("color: #666; font-size: 11px; font-weight: bold; margin-bottom: 15px; border: none; letter-spacing: 1px;"); l.addWidget(t); l.addWidget(widget); parent_layout.addWidget(group)
    def init_model_settings(self):
        w = QWidget(); l = QVBoxLayout(w); l.setContentsMargins(0,0,0,0); l.addWidget(QLabel("Aktives LLM Modell:")); self.model_c = QComboBox(); l.addWidget(self.model_c); return w
    def init_tts_settings(self):
        w = QWidget(); l = QVBoxLayout(w); l.setContentsMargins(0,0,0,0); l.setSpacing(20); l.addWidget(QLabel("TTS Engine:")); self.tts_c = QComboBox(); self.tts_c.addItems(["qwen3-tts", "piper-tts", "none"]); l.addWidget(self.tts_c)
        self.piper_v_cont = QWidget(); pv_l = QVBoxLayout(self.piper_v_cont); pv_l.setContentsMargins(0,0,0,0); pv_l.addWidget(QLabel("Piper Stimme:")); self.piper_v_c = QComboBox(); pv_l.addWidget(self.piper_v_c); l.addWidget(self.piper_v_cont)
        self.qwen_v_cont = QWidget(); qv_l = QVBoxLayout(self.qwen_v_cont); qv_l.setContentsMargins(0,0,0,0); qv_l.addWidget(QLabel("Stimmen-Klon (.wav):")); self.qwen_v_c = QComboBox(); qv_l.addWidget(self.qwen_v_c); l.addWidget(self.qwen_v_cont)
        self.tts_c.currentTextChanged.connect(self.toggle_voice_fields); return w
    def init_system_settings(self):
        w = QWidget(); l = QVBoxLayout(w); l.setContentsMargins(0,0,0,0); l.setSpacing(20); l.addWidget(QLabel("Sprach-Präferenz:")); self.lang_c = QComboBox(); self.lang_c.addItems(["de", "en", "auto"]); l.addWidget(self.lang_c); self.sec_c = QCheckBox("Sicherheitsmodus (Befehlsausführung verhindern)"); l.addWidget(self.sec_c); return w
    def toggle_voice_fields(self): self.piper_v_cont.setVisible(self.tts_c.currentText() == "piper-tts"); self.qwen_v_cont.setVisible(self.tts_c.currentText() == "qwen3-tts")
    def switch_page(self, index):
        self.btn_chat.setChecked(index == 0); self.btn_settings.setChecked(index == 1); self.stack.setCurrentIndex(index)
        if index == 1: self.refresh_all_options()
    def load_settings_into_ui(self): self.tts_c.setCurrentText(self.assistant.tts_type); self.lang_c.setCurrentText(self.assistant.language); self.sec_c.setChecked(self.assistant.security_mode); self.toggle_voice_fields(); self.refresh_all_options()
    def refresh_all_options(self):
        try:
            m = [m.model for m in ollama.list().models]
            self.model_c.clear(); self.model_c.addItems(sorted(list(set(m))))
            if self.assistant.ollama_model in m: self.model_c.setCurrentText(self.assistant.ollama_model)
        except: pass
        p_dir = os.path.join(os.path.dirname(__file__), "piper_models")
        if os.path.exists(p_dir):
            v = [f.replace(".onnx", "") for f in os.listdir(p_dir) if f.endswith(".onnx")]
            self.piper_v_c.clear(); self.piper_v_c.addItems(sorted(v))
            if self.assistant.piper_voice in v: self.piper_v_c.setCurrentText(self.assistant.piper_voice)
        q_dir = os.path.join(os.path.dirname(__file__), "voices")
        if os.path.exists(q_dir):
            v = [f for f in os.listdir(q_dir) if f.endswith(".wav")]
            self.qwen_v_c.clear(); self.qwen_v_c.addItems(sorted(v))
            if self.assistant.qwen_voice in v: self.qwen_v_c.setCurrentText(self.assistant.qwen_voice)
    def save_settings(self):
        d = {"ollama_model": self.model_c.currentText(), "tts_type": self.tts_c.currentText(), "language": self.lang_c.currentText(), "security_mode": self.sec_c.isChecked(), "piper_voice": self.piper_v_c.currentText(), "qwen_voice": self.qwen_v_c.currentText()}
        self.assistant.update_config(d); self.display_text("System", "Konfiguration erfolgreich gespeichert."); self.switch_page(0)
    def update_status(self, status):
        m = {"idle": ("#00d4ff", "JARVIS BEREIT"), "listening": ("#ff0064", "JARVIS HÖRT ZU..."), "thinking": ("#ffffff", "JARVIS DENKT..."), "speaking": ("#00ff96", "JARVIS SPRICHT")}
        c, t = m.get(status, m["idle"])
        self.status_dot.setStyleSheet(f"color: {c}; font-size: 20px; margin-left: 20px;"); self.status_text.setText(t); self.status_text.setStyleSheet(f"font-weight: bold; color: {c}; font-size: 13px;")
    def display_text(self, sender, text):
        if sender == "Jarvis":
            # Lösche alle temporären Tool-Bubbles wenn Jarvis die endgültige Antwort gibt
            for bubble in self.temp_tool_bubbles:
                bubble.hide()
                self.scroll_l.removeWidget(bubble)
            self.temp_tool_bubbles.clear()
        
        is_temp = (sender == "Tool")
        bubble = ChatBubble(sender, text, is_temporary=is_temp)
        if is_temp: self.temp_tool_bubbles.append(bubble)
        
        self.scroll_l.addWidget(bubble)
        QTimer.singleShot(100, lambda: self.scroll.verticalScrollBar().setValue(self.scroll.verticalScrollBar().maximum()))

    def process_text_input(self):
        t = self.input_f.text().strip()
        if t: 
            self.display_text("User", t)
            self.input_f.clear()
            if hasattr(self, 'at'):
                threading.Thread(target=self.at.assistant.run_ollama_agent, args=(t,), daemon=True).start()
    def closeEvent(self, event):
        """Wird aufgerufen, wenn das Fenster geschlossen wird."""
        print("[SYSTEM]: Beende Jarvis und bereinige Speicher...")

        # 1. Thread stoppen
        if hasattr(self, 'at'):
            self.at.terminate()
            self.at.wait()

        # 2. Modelle aktiv entladen (VRAM fix)
        if hasattr(self, 'assistant'):
            self.assistant.unload_models()

        # 3. Alle hängenden Subprozesse in dieser Gruppe killen
        import os
        import signal
        try:
            os.killpg(0, signal.SIGKILL)
        except:
            pass

        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv); app.setStyle("Fusion")
    assistant = JarvisAssistant()
    gui = JarvisGUI(assistant)
    at = AssistantThread(assistant)
    at.text_received.connect(gui.display_text); at.status_changed.connect(gui.update_status)
    assistant.on_status_change = lambda s: at.status_changed.emit(s)
    gui.at = at; at.start(); gui.show(); sys.exit(app.exec_())

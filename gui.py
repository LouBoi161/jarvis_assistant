import sys
import os
import threading
import time
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QFrame, QGraphicsDropShadowEffect)
from PyQt5.QtCore import Qt, pyqtSignal, QPropertyAnimation, QRect, QEasingCurve, QThread, QSize
from PyQt5.QtGui import QColor, QFont, QPalette

# Importiere den bestehenden Assistant
from main import JarvisAssistant

class AssistantThread(QThread):
    """Thread, der den JarvisAssistant im Hintergrund ausführt, um die GUI nicht zu blockieren."""
    status_changed = pyqtSignal(str) # 'listening', 'thinking', 'speaking', 'idle'
    text_received = pyqtSignal(str, str) # sender, text
    
    def __init__(self, assistant):
        super().__init__()
        self.assistant = assistant
        # Wir müssen die log-Funktion des Assistants umleiten
        self.original_log = self.assistant.log
        self.assistant.log = self.custom_log
        
    def custom_log(self, message, level="debug"):
        self.original_log(message, level)
        if level == "standard":
            if "[" in message and "]" in message:
                sender = message.split("]")[0].replace("[", "")
                text = message.split("]")[1].strip()
                self.text_received.emit(sender, text)
            else:
                self.text_received.emit("System", message)

    def run(self):
        # Hier starten wir die Hauptschleife des Assistants
        # Da main.py für Terminal ausgelegt ist, müssen wir evtl. Teile anpassen
        # Für den Anfang lassen wir ihn einfach laufen
        self.assistant.run()

class JarvisGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        
    def init_ui(self):
        # Fenster-Eigenschaften: Rahmenlos, Immer im Vordergrund, Transparent
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        # Design & Layout
        self.layout = QVBoxLayout()
        self.layout.setAlignment(Qt.AlignCenter)
        
        # Der Haupt-Container (Glassmorphism Look)
        self.container = QFrame()
        self.container.setObjectName("MainContainer")
        self.container.setFixedWidth(800)
        self.container.setMinimumHeight(80)
        
        self.container.setStyleSheet("""
            QFrame#MainContainer {
                background-color: rgba(11, 14, 20, 220);
                border: 2px solid #00d4ff;
                border-radius: 15px;
            }
        """)
        
        # Schatten-Effekt (Glow)
        self.glow = QGraphicsDropShadowEffect()
        self.glow.setBlurRadius(25)
        self.glow.setColor(QColor(0, 212, 255, 150))
        self.glow.setOffset(0, 0)
        self.container.setGraphicsEffect(self.glow)
        
        self.content_layout = QVBoxLayout(self.container)
        
        # Status Label (Ganz oben, klein)
        self.status_label = QLabel("SYSTEM READY")
        self.status_label.setStyleSheet("color: #00d4ff; font-size: 10px; font-weight: bold; letter-spacing: 2px;")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.content_layout.addWidget(self.status_label)
        
        # Output Text (Die Antwort von Jarvis)
        self.output_label = QLabel("Say 'Hey Jarvis' to start...")
        self.output_label.setWordWrap(True)
        self.output_label.setAlignment(Qt.AlignCenter)
        self.output_label.setStyleSheet("color: white; font-size: 18px; padding: 10px;")
        self.content_layout.addWidget(self.output_label)
        
        # Input Field (Dezent am unteren Rand)
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type a command...")
        self.input_field.setStyleSheet("""
            QLineEdit {
                background-color: transparent;
                border: none;
                border-top: 1px solid rgba(0, 212, 255, 50);
                color: #00d4ff;
                font-size: 14px;
                padding: 5px;
            }
        """)
        self.input_field.setAlignment(Qt.AlignCenter)
        self.input_field.returnPressed.connect(self.process_text_input)
        self.content_layout.addWidget(self.input_field)
        
        self.layout.addWidget(self.container)
        self.setLayout(self.layout)
        
        # Position am unteren Bildschirmrand
        screen = QApplication.primaryScreen().geometry()
        self.move(screen.width() // 2 - 400, screen.height() - 150)
        
        # Animation für Glow
        self.glow_anim = QPropertyAnimation(self.glow, b"blurRadius")
        self.glow_anim.setDuration(1500)
        self.glow_anim.setStartValue(10)
        self.glow_anim.setEndValue(30)
        self.glow_anim.setLoopCount(-1)
        self.glow_anim.setEasingCurve(QEasingCurve.InOutQuad)
        self.glow_anim.start()

    def update_status(self, status):
        colors = {
            "idle": QColor(0, 212, 255, 150),
            "listening": QColor(255, 0, 100, 200), # Pink/Rot beim Hören
            "thinking": QColor(255, 255, 255, 200), # Weiß beim Denken
            "speaking": QColor(0, 255, 150, 200)  # Grün beim Sprechen
        }
        self.glow.setColor(colors.get(status, colors["idle"]))
        self.status_label.setText(status.upper())
        
    def display_text(self, sender, text):
        # Falls Jarvis spricht, Text anzeigen
        if "Jarvis" in sender or "sorc" in sender:
            self.output_label.setText(text)
            self.update_status("speaking")
            # Nach 10 Sekunden den Text wieder löschen (Idle)
            threading.Timer(10.0, lambda: self.output_label.setText("Jarvis is idle.")).start()
            threading.Timer(10.0, lambda: self.update_status("idle")).start()
        elif sender == "System":
             self.status_label.setText(text.upper()[:30])

    def process_text_input(self):
        text = self.input_field.text()
        if text:
            self.output_label.setText(f"User: {text}")
            self.input_field.clear()
            # Hier müssten wir den Text an den Assistant übergeben
            # Wir bauen das gleich in den AssistantThread ein
            if hasattr(self, 'at'):
                self.at.assistant.run_ollama_agent(text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Jarvis Assistant initialisieren
    assistant = JarvisAssistant()
    
    gui = JarvisGUI()
    
    # Assistant in Hintergrund-Thread starten
    at = AssistantThread(assistant)
    at.text_received.connect(gui.display_text)
    at.status_changed.connect(gui.update_status)
    
    # Hook den status_changed Signal an den Assistant
    assistant.on_status_change = lambda s: at.status_changed.emit(s)
    
    gui.at = at # Referenz speichern
    
    at.start()
    gui.show()
    
    sys.exit(app.exec_())

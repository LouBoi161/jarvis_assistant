# JARVIS - AI Voice Assistant for Linux (Stable v2.3)

A local, intelligent, and professional AI voice assistant for Linux, powered by Ollama, OpenAI Whisper, and modern TTS engines.

## Features
- **All-in-One GUI:** Integrated Chat and Settings interface.
- **Voice Interaction:** Wake word detection ("Hey Jarvis") and local STT/TTS.
- **Proactive Agent:** Executes shell commands, searches the web, and manages files.
- **VRAM Management:** Actively unloads models on exit to free up your GPU.

## System Requirements

| Component | Minimum (CPU-only) | Recommended (GPU) |
|-----------|--------------------|-------------------|
| **RAM** | 8 GB | 16 GB+ |
| **GPU** | Not required | NVIDIA GPU (8GB+ VRAM) |
| **Storage** | ~10 GB | ~15 GB (for models) |

## Installation

### 1. System Dependencies
```bash
# Arch Linux
sudo pacman -S portaudio ffmpeg uv scrot grim
```

### 2. Install Ollama
Make sure the service is running and pull the default model:
```bash
ollama pull gemma4:e4b
```

### 3. Setup (via uv)
```bash
# Clone
git clone https://github.com/LouBoi161/jarvis_assistant.git
cd jarvis_assistant

# Setup & Install dependencies
uv sync

# Download initial offline voices (Piper TTS)
uv run download_voices.py
```

## Usage

### 🚀 Start GUI (Recommended)
The easiest way to use JARVIS is via the floating chat interface:
```bash
uv run gui.py
```

### 🖥️ Start CLI (Terminal only)
If you prefer the terminal:
```bash
uv run main.py
```

## Shortcuts & Commands
- **Wake Word:** "Hey Jarvis"
- **Settings:** Click the ⚙️ icon in the GUI or type `/settings` in the terminal.
- **Exit:** Close the window or press `Ctrl+C`. JARVIS will automatically clean up your VRAM.

---
**Made with the help of AI.**
MIT License.

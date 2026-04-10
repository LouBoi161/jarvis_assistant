# JARVIS - AI Voice Assistant for Linux (WIP)

A local, intelligent, and professional AI voice assistant for Linux, powered by Ollama, OpenAI Whisper, and Qwen3-TTS.

**Status:** Work In Progress (WIP) - Not yet final, but fully functional.

## Features
- **Voice Interaction:** Wake word detection ("Hey Jarvis") and silence-based recording.
- **Local STT:** Uses OpenAI Whisper (running locally) for high-quality speech-to-text.
- **LLM Reasoning:** Integrates with **Ollama** (default model: `gemma4:e4b`) for intelligent responses and tool execution.
- **Natural TTS:** Uses **Qwen3-TTS (0.6B)** for high-quality, expressive voice output.
- **Tool Execution:** JARVIS can execute shell commands, search the web, and manage settings through its integrated agent system.
- **Emotion Support:** Supports emotional tags like `[glücklich]`, `[aufgeregt]`, `[nachdenklich]`, and `[freundlich]` for more expressive speech.

## Requirements & Installation

### 1. System Dependencies
JARVIS requires some system-level libraries for audio processing and speech recognition.

**Arch Linux:**
```bash
sudo pacman -S portaudio ffmpeg
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install portaudio19-dev ffmpeg
```

**Fedora:**
```bash
sudo dnf install portaudio-devel ffmpeg
```

### 2. Install Ollama
Ollama is used for the LLM reasoning (Gemma 4). If you don't have it yet, install it with:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```
Make sure the service is running (`ollama serve`) and pull the default model:
```bash
ollama pull gemma4:e4b
```

### 3. Python Setup
We recommend using a virtual environment and **uv** for lightning-fast dependency installation.

```bash
# Clone the repository
git clone https://github.com/LouBoi161/jarvis_assistant.git
cd jarvis_assistant

# Install uv (if you don't have it yet)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install Python requirements using uv
uv pip install -r requirements.txt
```

## Usage

1. **Start JARVIS:**
   ```bash
   python main.py
   ```
2. **Interaction:**
   - **Voice:** Say **"Hey Jarvis"** to wake him up. After the notification sound, speak your command.
   - **Text:** You can also type directly into the terminal while JARVIS is running.
3. **Commands:**
   - `/settings` - Open the GUI to switch Ollama models.
   - `/model <name>` - Quick-switch to another model (e.g., `/model llama3`).
   - `/clear` - Clear the current chat history.
   - `/exit` - Close the assistant.

## Custom Voice Cloning
JARVIS supports zero-shot voice cloning via Qwen3-TTS.
1. Place a short audio file (5-10 seconds) of the target voice in the root directory.
2. Rename it to `voice.wav`.
3. (Optional) Create a `voice.txt` containing the exact transcript of what is said in `voice.wav` for better quality.
4. Restart JARVIS. He will now speak with that voice.

## Roadmap
- [ ] Improve voice cloning reliability.
- [ ] Add more tool integrations.
- [ ] Optimize VRAM usage for simultaneous STT/LLM/TTS.

---

**Made with the help of AI.**

## License
MIT License. See [LICENSE](LICENSE) for details.

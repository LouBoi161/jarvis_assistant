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

## Requirements
- **OS:** Linux
- **Python:** 3.10+
- **Ollama:** Installed and running (`ollama serve`)
- **PortAudio:** Required for `pyaudio` (install via your package manager, e.g., `sudo pacman -S portaudio` or `sudo apt install libportaudio2`)
- **GPU (Optional):** Recommended for faster STT/TTS (though TTS runs on CPU by default).

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/jarvis_assistant.git
   cd jarvis_assistant
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Pull the Ollama model:**
   Make sure Ollama is running, then pull the default model:
   ```bash
   ollama pull gemma4:e4b
   ```

## Usage

1. **Voice Setup:**
   You can provide a `voice.wav` in the root directory for voice cloning. If no `voice.wav` is found, the assistant will use its default voice.

2. **Run JARVIS:**
   ```bash
   python main.py
   ```

3. **Interaction:**
   - Say **"Hey Jarvis"** to wake up the assistant.
   - Or type directly into the terminal.
   - Use commands like `/settings`, `/model <name>`, or `/clear` in the terminal.

## Roadmap
- [ ] Improve voice cloning reliability.
- [ ] Add more tool integrations.
- [ ] Optimize VRAM usage for simultaneous STT/LLM/TTS.

## License
MIT License. See [LICENSE](LICENSE) for details.

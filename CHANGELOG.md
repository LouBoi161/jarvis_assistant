# Changelog - JARVIS Assistant

Alle wichtigen Änderungen an diesem Projekt werden in dieser Datei festgehalten.

## [2.4.0] - 2026-04-20
### Hinzugefügt
- **Intelligentes Gehirn (System Prompt v2):** Kompletter Rewrite des Prompts für "Chain-of-Thought" Denken.
- **Recherche-Zwang:** Jarvis prüft nun proaktiv via Websearch Paketnamen und Befehlssyntax, anstatt zu raten.
- **Fehler-Resilienz:** Automatische Bewertung von Tool-Outputs. Bei Fehlern (z.B. "Command not found") sucht Jarvis selbstständig nach Alternativen.
- **Robust Tool Extraction:** Unterstützung für Kurzform-Tags (`EXEC_CMD:`, `SEARCH_WEB:`) für bessere Stabilität bei kleinen LLMs.

## [2.3.0] - 2026-04-20
### Hinzugefügt
- **UI/UX Upgrade:** Pulsierende Lade-Animation (Typing-Indikator) hinzugefügt.
- **Status-Synchronisation:** Neues `is_busy` Flag verhindert das Flackern/Verschwinden des Ladestatus.
- **Interaktions-Kontrolle:** Senden-Button wird während der Verarbeitung zum Stopp-Button; Eingabefeld wird gesperrt.
- **Projekt-Struktur:** Umstellung von `requirements.txt` auf `pyproject.toml` für `uv sync` Workflow.

## [2.2.0] - 2026-04-20
### Hinzugefügt
- **Always-On LLM:** Modell bleibt dauerhaft im Speicher (`keep_alive=-1`) für sofortige Antwortzeiten.
- **Pre-loading:** LLM wird direkt beim Programmstart im Hintergrund geladen.
- **VRAM Optimierung:** Automatisches Entladen alter Modelle bei Modellwechsel in den Einstellungen.

## [2.1.0] - 2026-04-20
### Hinzugefügt
- **TTS Engine Auswahl:** Support für Qwen3-TTS, Piper-TTS und "None".
- **Voice Folder:** Stimmen für Qwen werden nun dynamisch aus dem `voices/` Ordner geladen.
- **Detailliertes Feedback:** GUI zeigt nun exakte Befehle und Suchanfragen in den Info-Bubbles an.

## [2.0.0] - 2026-04-20
### Hinzugefügt
- **GUI Release:** Erste stabile Version mit PyQt5 Oberfläche.
- **Logging:** Einführung der `latest_session.log` für einfacheres Debugging.

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Qwen/Qwen3-TTS-12Hz-0.6B"

try:
    print(f"Versuche {model_id} zu laden...")
    # Wir laden nur die Config/Tokenizer um zu sehen ob das Repo existiert und die Klassen passen
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print("Erfolgreich geladen!")
except Exception as e:
    print(f"Fehler beim Laden: {e}")

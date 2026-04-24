import os
import urllib.request

def download_initial_voices():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    piper_dir = os.path.join(script_dir, "piper_models")
    os.makedirs(piper_dir, exist_ok=True)
    
    # 4 Beste Stimmen: Deutsch (Männlich/Weiblich), Englisch (Männlich/Weiblich)
    voices = [
        "de_DE-thorsten-high",      # DE Männlich
        "de_DE-kerstin-low",        # DE Weiblich
        "en_US-ryan-high",          # EN Männlich
        "en_US-amy-medium"          # EN Weiblich
    ]
    
    base_url = "https://huggingface.co/rhasspy/piper-voices/resolve/main"
    
    print("--- Lade initiale Piper-Stimmen herunter ---")
    for voice in voices:
        model_path = os.path.join(piper_dir, f"{voice}.onnx")
        config_path = os.path.join(piper_dir, f"{voice}.onnx.json")
        
        if not os.path.exists(model_path) or not os.path.exists(config_path):
            print(f"Lade Stimme herunter: {voice}...")
            parts = voice.split('-')
            if len(parts) >= 3:
                lang_country = parts[0]
                voice_name = parts[1]
                quality = parts[2]
                lang_code = lang_country.split('_')[0]
                repo_path = f"{lang_code}/{lang_country}/{voice_name}/{quality}/{voice}.onnx"
                
                try:
                    urllib.request.urlretrieve(f"{base_url}/{repo_path}", model_path)
                    urllib.request.urlretrieve(f"{base_url}/{repo_path}.json", config_path)
                    print(f"✓ {voice} erfolgreich heruntergeladen.")
                except Exception as e:
                    print(f"x Fehler bei {voice}: {e}")
        else:
            print(f"✓ {voice} ist bereits vorhanden.")
            
    print("--- Download abgeschlossen! ---")

if __name__ == "__main__":
    download_initial_voices()
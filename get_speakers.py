import torch
import transformers.pytorch_utils
if not hasattr(transformers.pytorch_utils, 'isin_mps_friendly'):
    def isin_mps_friendly(elements, tensor):
        return torch.isin(elements, tensor)
    transformers.pytorch_utils.isin_mps_friendly = isin_mps_friendly

from TTS.api import TTS

tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2')
print("Verfügbare Sprecher:", tts.speakers)
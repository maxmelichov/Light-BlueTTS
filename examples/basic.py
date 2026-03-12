"""
Usage:
    wget https://huggingface.co/thewh1teagle/phonikud-onnx/resolve/main/phonikud-1.0.int8.onnx
    wget https://github.com/maxmelichov/Light-BlueTTS/releases/download/model-files-v1.0/onnx_models.tar.gz
    wget https://github.com/maxmelichov/Light-BlueTTS/releases/download/model-files-v1.0/voices.tar.gz
    tar -xf onnx_models.tar.gz
    tar -xf voices.tar.gz
    uv pip install phonikud phonikud-onnx
    uv run examples/basic.py
"""

import soundfile as sf
from phonikud_onnx import Phonikud
from phonikud import phonemize
from lightblue_onnx import LightBlueTTS

nikud_model = Phonikud("phonikud-1.0.int8.onnx")
tts = LightBlueTTS("onnx_models", style_json="voices/female1.json")

text = "שימו לב נוסעים יקרים, הרכבת תכנס לתחנת תל אביב מרכז בעוד מספר דקות, אנא התרחקו מקצה הרציף והמתינו מאחורי הקו הצהוב, תודה."
vocalized = nikud_model.add_diacritics(text)
phonemes = phonemize(vocalized)
print(f"Phonemes: {phonemes}")

samples, sr = tts.create(phonemes)

sf.write("audio.wav", samples, sr)
print("Saved audio.wav")

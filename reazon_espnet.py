from reazonspeech.espnet.asr import load_model, transcribe, audio_from_path
import json
from dataclasses import asdict

audio = audio_from_path(
    "/mnt/work-qnap/llmc/J-CHAT/audio/podcast_test/00000-of-00001/cuts.000000/dcdd979f47cb788aeb8ef58033d37fff.wav"
)
model = load_model()
ret = transcribe(model, audio)

with open("data/text/result.json", "w", encoding="utf-8") as f:
    json.dump(asdict(ret), f, ensure_ascii=False, indent=2)

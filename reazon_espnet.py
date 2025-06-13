from reazonspeech.espnet.asr import load_model, transcribe, audio_from_path
import json
from dataclasses import asdict

# 音声ファイル読み込みと書き起こし
audio = audio_from_path(
    "/mnt/work-qnap/llmc/J-CHAT/audio/podcast_test/00000-of-00001/cuts.000000/dcdd979f47cb788aeb8ef58033d37fff.wav"
)
model = load_model()
ret = transcribe(model, audio)
# print(ret)

# asdict で変換
result = asdict(ret)

# segmentのstart/end変換と丸め
for seg in result.get("segments", []):
    seg["start"] = round(seg.pop("start_seconds"), 3)
    seg["end"] = round(seg.pop("end_seconds"), 3)

# 保存
with open("data/text/result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

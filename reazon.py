from reazonspeech.nemo.asr import load_model, transcribe, audio_from_path
import json
from dataclasses import asdict
from pathlib import Path

# ディレクトリのパス
input_dir = Path(
    "/mnt/work-qnap/llmc/J-CHAT/audio/podcast_test/00000-of-00001/cuts.000000"
)
output_dir = Path("data/text_nemo")
output_dir.mkdir(parents=True, exist_ok=True)

# モデルの読み込み（最初に一度だけ）
model = load_model()

# .wav ファイルをすべて処理
for wav_path in sorted(input_dir.glob("*.wav")):
    print(f"Processing: {wav_path.name}")
    audio = audio_from_path(str(wav_path))
    ret = transcribe(model, audio)
    result = asdict(ret)

    # segment の start_seconds/end_seconds を start/end に変換＆丸め
    for seg in result.get("segments", []):
        seg["start"] = round(seg.pop("start_seconds"), 3)
        seg["end"] = round(seg.pop("end_seconds"), 3)

    # JSON ファイル名は .wav 名に _nemo.json を付けたもの
    json_path = output_dir / f"{wav_path.stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

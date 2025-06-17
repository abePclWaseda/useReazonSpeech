from reazonspeech.nemo.asr import load_model, transcribe, audio_from_path
import json
from dataclasses import asdict
from pathlib import Path

# 入力のルートディレクトリ
root_dir = Path("/mnt/work-qnap/llmc/J-CHAT/audio/podcast_train")
# 出力先ルートディレクトリ
output_root = Path("data/J-CHAT/text/podcast_train")
output_root.mkdir(parents=True, exist_ok=True)

# モデルの読み込み（1回だけ）
model = load_model(device="cuda:2")

# すべてのシャードを再帰的に探索
for shard_dir in sorted(root_dir.glob("*/cuts.*")):
    for wav_path in sorted(shard_dir.glob("*.wav")):
        print(f"Processing: {wav_path}")

        # 音声読み込みと文字起こし
        audio = audio_from_path(str(wav_path))
        ret = transcribe(model, audio)
        result = asdict(ret)

        # segment の時刻を start/end に変換し、秒数を丸める
        for seg in result.get("segments", []):
            seg["start"] = round(seg.pop("start_seconds"), 3)
            seg["end"] = round(seg.pop("end_seconds"), 3)

        # 出力ディレクトリを元に作成
        rel_path = wav_path.relative_to(root_dir)
        output_dir = output_root / rel_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # ファイル名: xxx.json として保存
        json_path = output_dir / f"{wav_path.stem}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

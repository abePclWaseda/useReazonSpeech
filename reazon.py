from reazonspeech.nemo.asr import load_model, transcribe, audio_from_path
import json
from dataclasses import asdict
from pathlib import Path

# 推論
audio = audio_from_path(
    "/mnt/work-qnap/llmc/J-CHAT/audio/podcast_test/00000-of-00001/cuts.000000/dcdd979f47cb788aeb8ef58033d37fff.wav"
)
model = load_model()
ret = transcribe(model, audio)
# asdict で変換
result = asdict(ret)

# segmentのstart/end変換と丸め
for seg in result.get("segments", []):
    seg["start"] = round(seg.pop("start_seconds"), 3)
    seg["end"] = round(seg.pop("end_seconds"), 3)

# 保存
with open("data/text/dcdd979f47cb788aeb8ef58033d37fff_nemo.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

# # 保存
# output_path = Path(
#     "/mnt/kiso-qnap3/yuabe/m1/moshi-finetune/data/CallHome/text/0696_full_ret.json"
# )
# output_path.write_text(
#     json.dumps(ret_dict, ensure_ascii=False, indent=2), encoding="utf-8"
# )

# print("✅ Saved:", output_path)

# フォーマット整形（1行ずつ書き出す）
# out_path = Path("data/text/dcdd979f47cb788aeb8ef58033d37fff_nemo.json")
# out_path.parent.mkdir(parents=True, exist_ok=True)
# with out_path.open("w", encoding="utf-8") as f:
#     f.write("[\n")
#     for i, sw in enumerate(ret.subwords):
#         obj = {
#             # "speaker": getattr(sw, "speaker", "A"),
#             "text": sw.token,
#             # "start": sw.start,
#             # "end": sw.end,
#         }
#         json_str = json.dumps(obj, ensure_ascii=False)
#         f.write(f"  {json_str}")
#         if i != len(ret.subwords) - 1:
#             f.write(",\n")
#         else:
#             f.write("\n")
#     f.write("]\n")

# print("✅ Saved:", out_path)

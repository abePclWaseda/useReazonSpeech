# from reazonspeech.espnet.asr import load_model, transcribe, audio_from_path

# audio = audio_from_path("dcdd979f47cb788aeb8ef58033d37fff.wav")
# model = load_model()
# ret = transcribe(model, audio)
# print(ret)

# import json
# from dataclasses import asdict
# from pathlib import Path

# segments_json = [asdict(seg) for seg in ret.segments]

# out_path = Path("segments.json")
# out_path.write_text(
#     json.dumps(segments_json, ensure_ascii=False, indent=2), encoding="utf-8"
# )
# print("✅ saved to", out_path)

# -----------------------------------------------------------------------------
from reazonspeech.nemo.asr import load_model, transcribe, audio_from_path

audio = audio_from_path("/mnt/kiso-qnap3/yuabe/m1/moshi-finetune/data/CallHome/audio/0696.mp3")
model = load_model()
ret = transcribe(model, audio)

import json
from dataclasses import asdict
from pathlib import Path

sp_tokens = [asdict(sw) for sw in ret.subwords]

out = Path("/mnt/kiso-qnap3/yuabe/m1/moshi-finetune/data/CallHome/text/0696.json")
out.write_text(json.dumps(sp_tokens, ensure_ascii=False, indent=2), encoding="utf-8")
print("✅ Saved:", out)

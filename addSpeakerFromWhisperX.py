import json


def merge_subwords_from_chars(subwords, char_words):
    """
    subwords の token を構成する char_words を順に対応づけて
    speaker, start, end を付ける
    """
    result = []
    word_idx = 0  # char_words の位置

    for sw in subwords:
        token = sw["token"]
        token_chars = list(token)

        matched_words = []
        while word_idx < len(char_words) and len(matched_words) < len(token_chars):
            cw = char_words[word_idx]
            matched_words.append(cw)
            word_idx += 1

            # 現在までにマッチした文字列
            current_str = "".join(w["word"] for w in matched_words)
            if current_str == token:
                break

        # 安全性チェック（本来は常に一致するはず）
        if "".join(w["word"] for w in matched_words) != token:
            raise ValueError(f"Token '{token}' に対応する文字列が見つかりませんでした")

        result.append(
            {
                "speaker": matched_words[0]["speaker"],
                "word": token,
                "start": matched_words[0]["start"],
                "end": matched_words[-1]["end"],
            }
        )

    return result


# JSONファイルの読み込み
with open(
    "/mnt/kiso-qnap3/yuabe/m1/useReazonSpeech/data/text/dcdd979f47cb788aeb8ef58033d37fff_nemo.json",
    "r",
    encoding="utf-8",
) as f:
    json_data = json.load(f)

with open(
    "/mnt/kiso-qnap3/yuabe/m1/useWhisperX/data/text/dcdd979f47cb788aeb8ef58033d37fff_reazon_nemo.json",
    "r",
    encoding="utf-8",
) as f:
    word_timings = json.load(f)

# マージ実行
merged_subwords = merge_subwords_from_chars(json_data["subwords"], word_timings)

# 出力ファイルに JSON 配列形式で書き出し
with open("data/text/merged_subwords.json", "w", encoding="utf-8") as f:
    f.write("[\n")
    for i, item in enumerate(merged_subwords):
        line = json.dumps(item, ensure_ascii=False)
        if i < len(merged_subwords) - 1:
            f.write("   " + f"{line},\n")
        else:
            f.write("   " + f"{line}\n")
    f.write("]\n")

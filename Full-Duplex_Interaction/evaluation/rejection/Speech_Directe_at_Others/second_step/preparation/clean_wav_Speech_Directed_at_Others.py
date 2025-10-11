#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from pydub import AudioSegment


def process_one_pair(wav_path: Path):
    json_path = wav_path.with_suffix("_sentence.json")
    if not json_path.exists():
        print(f"[WARN] 找不到 JSON：{json_path.name}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    segs = meta.get("speech_segments", [])
    if len(segs) < 2:
        print(f"[WARN] {wav_path.name} 段数不足2，跳过")
        return

    # === step 1: 确定删除区间（第0段 + 它之后的静音间隔） ===
    first_seg = segs[0]
    next_seg = segs[1]
    del_start = int(round(first_seg["xmin"] * 1000))
    del_end = int(round(next_seg["xmin"] * 1000))  # 到下一段开始处
    removed_ms = del_end - del_start

    # === step 2: 原始第2段（index=2）为静音段（删除后会变成新的 index=1） ===
    seg_silence = segs[2] if len(segs) >= 3 else None

    # === step 3: 读取音频 ===
    audio = AudioSegment.from_wav(str(wav_path))
    # 删除第一段+静音间隔
    new_audio = audio[:del_start] + audio[del_end:]

    # === step 4: 替换静音段 ===
    if seg_silence:
        silence_start_ms = int(round(seg_silence["xmin"] * 1000)) - removed_ms
        silence_end_ms = int(round(seg_silence["xmax"] * 1000)) - removed_ms
        silence_start_ms = max(silence_start_ms, 0)
        silence_end_ms = min(silence_end_ms, len(new_audio))
        if silence_end_ms > silence_start_ms:
            silence_seg = AudioSegment.silent(
                duration=silence_end_ms - silence_start_ms,
                frame_rate=new_audio.frame_rate
            ).set_channels(new_audio.channels)
            new_audio = new_audio[:silence_start_ms] + silence_seg + new_audio[silence_end_ms:]

    # === step 5: 更新 JSON ===
    new_segments = []
    for i, s in enumerate(segs):
        # 跳过第一段
        if i == 0:
            continue
        new_xmin = s["xmin"] - removed_ms / 1000.0
        new_xmax = s["xmax"] - removed_ms / 1000.0
        text = s["text"]
        # 原第3段置为空
        if i == 2:
            text = ""
        new_segments.append({
            "xmin": max(0, new_xmin),
            "xmax": max(0, new_xmax),
            "text": text
        })

    new_meta = {
        "final_duration": len(new_audio) / 1000.0,
        "speech_segments": new_segments
    }

    # === step 6: 保存 ===
    out_wav = wav_path.parent / f"clean_{wav_path.name}"
    #out_json = wav_path.parent / f"clean_{wav_path.stem}.json"
    new_audio.export(str(out_wav), format="wav")
    #with open(out_json, "w", encoding="utf-8") as f:
        #json.dump(new_meta, f, ensure_ascii=False, indent=2)

    print(f"[OK] 处理完成：{wav_path.name} -> {out_wav.name}")


def process_folder(folder: str):
    p = Path(folder)
    for wav_file in sorted(p.glob("*.wav")):
        process_one_pair(wav_file)


if __name__ == "__main__":
    input_dir = "/home/work_nfs17/gjli/full-duplex-data/en_dev/talk_to_others_process_osum"  # ← 修改为你的输入目录
    process_folder(input_dir)

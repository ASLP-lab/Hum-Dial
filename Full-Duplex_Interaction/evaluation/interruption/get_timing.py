#!/usr/bin/env python3
"""
==============================================

Processes audio files and calculates latency metrics.
Writes all results to a single JSON file with average latency calculations.
"""

from __future__ import annotations
import argparse, json, importlib, os
from pathlib import Path
from typing import List, Tuple, Dict, Any
import torch, torchaudio
import re

# ---------------- Parameters ----------------
SR = 16_000
USER_MERGE_GAP = 0.6  
MODEL_MERGE_GAP = 0.5  
AUDIO_EXT = "wav"
OUT_FILENAME = "Follow-up_Questions_latency_results.json" #不同子任务的输出文件
result_dir = "./dev/json_group" #Follow-up_Questions_latency_results.json等文件的输出路径

# ---------------- Silero‑VAD ---------------
vad_module = importlib.import_module("silero_vad")
if hasattr(vad_module, "VoiceActivityDetector"):
    from silero_vad import VoiceActivityDetector

    _VAD = VoiceActivityDetector(sample_rate=SR)

    def _vad_ts(wav):
        return [(t["start"], t["end"]) for t in _VAD.get_speech_ts(wav)]

else:
    from silero_vad import get_speech_timestamps

    _M, _ = torch.hub.load(
        "snakers4/silero-vad", model="silero_vad", trust_repo=True, onnx=False
    )

    def _vad_ts(w):
        return [
            (t["start"], t["end"])
            for t in get_speech_timestamps(w, _M, sampling_rate=SR)
        ]


# -------------- Helpers --------------------
def load_wav(p: Path):
    wav, sr = torchaudio.load(p)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav.squeeze(0)


def _merge(seg: List[Tuple[float, float]], gap_thr: float):
    if not seg:
        return []
    seg = sorted(seg)
    merged = [seg[0]]
    for s, e in seg[1:]:
        ps, pe = merged[-1]
        if s - pe <= gap_thr:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def seg_sec(wav, gap_thr):
    return _merge([(s / SR, e / SR) for s, e in _vad_ts(wav)], gap_thr)


# -------------- Interval utilities ---------
def overlaps(user, model):
    raw = []
    i = j = 0
    while i < len(user) and j < len(model):
        u_s, u_e = user[i]
        m_s, m_e = model[j]
        s = max(u_s, m_s)
        e = min(u_e, m_e)
        if e > s:
            raw.append((s, e))
        if u_e < m_e:
            i += 1
        else:
            j += 1
    # keep shortest interval per rounded end‑time (ms precision)
    best = {}
    for s, e in raw:
        key = int(round(e * 1000))
        if key not in best or (e - s) < (best[key][1] - best[key][0]):
            best[key] = (s, e)
    return [
        [round(s, 3), round(e, 3)] for s, e in sorted(best.values(), key=lambda x: x[1])
    ]


def response_gaps(user, model):
    model_starts = [s for s, _ in model]
    tmp = {}
    for u_s, u_e in user:
        nxt = next((s for s in model_starts if s > u_e), None)
        if nxt is None:
            continue
        key = int(round(nxt * 1000))
        candidate = [round(u_e, 3), round(nxt, 3)]
        # keep shorter gap (later user_end)
        if key not in tmp or candidate[0] > tmp[key][0]:
            tmp[key] = candidate
    return [iv for _, iv in sorted(tmp.items(), key=lambda kv: kv[1][1])]


# -------------- Processing functions ----------
def process_file_pair(user_wav: Path, model_wav: Path) -> Dict[str, Any]:
    """
    Process a pair of user and model audio files.
    """
    user_seg = seg_sec(load_wav(user_wav), USER_MERGE_GAP)
    model_seg = seg_sec(load_wav(model_wav), MODEL_MERGE_GAP)
    
    return {
        "latency_stop_list": overlaps(user_seg, model_seg),
        "latency_resp_list": response_gaps(user_seg, model_seg),
    }


def calculate_average_latency(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate average latency from all results.
    """
    total_stop = 0.0
    count_stop = 0
    total_resp = 0.0
    count_resp = 0
    
    for result in results:
        # Calculate average stop latency
        for interval in result["latency_stop_list"]:
            total_stop += interval[1] - interval[0]
            count_stop += 1
        
        # Calculate average response latency
        for interval in result["latency_resp_list"]:
            total_resp += interval[1] - interval[0]
            count_resp += 1
    
    return {
        "avg_latency_stop": total_stop / count_stop if count_stop > 0 else 0.0,
        "avg_latency_resp": total_resp / count_resp if count_resp > 0 else 0.0,
    }


# -------------- Main processing ----------
def main():
    ap = argparse.ArgumentParser(
        description="Process audio files and calculate latency metrics"
    )
    ap.add_argument("--root_dir", default="./dev/Follow-up_Questions") #输入：存放wav音频数据的文件夹
    args = ap.parse_args()

    root_dir = Path(args.root_dir)
    all_results = []
    
    # Collect all user audio files
    user_files = []

    for file in root_dir.iterdir():
        if file.suffix == f".{AUDIO_EXT}":
            file_name = file.stem
            # 匹配 xxxx_xxxx 或 xxxx_xxxx_add 格式
            if re.match(r'^\d{4}_\d{4}(_add)?$', file_name):
                user_files.append(file)

    
    # Process each user file
    for user_file in user_files:
        file_prefix = user_file.stem
        model_file = user_file.with_name(f"{file_prefix}_output.{AUDIO_EXT}")
        
        if not model_file.exists():
            print(f"⚠️ Missing model file for {user_file}, skipping")
            continue
        
        print(f"Processing: {user_file.name} and {model_file.name}")
        result = process_file_pair(user_file, model_file)
        
        # Add to results with key
        all_results.append({
            "key": f"repeat_{file_prefix}",
            "latency_stop_list": result["latency_stop_list"],
            "latency_resp_list": result["latency_resp_list"]
        })
    
    # Calculate average latency
    avg_latency = calculate_average_latency(all_results)
    
    # Create final output structure
    final_output = {
        "results": all_results,
        "average_latency": avg_latency
    }
    
    # Save to file
    output_path = Path(result_dir) / OUT_FILENAME
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(final_output, fp, ensure_ascii=False, indent=2)
    
    print(f"✔ All results saved to: {output_path}")
    print(f"Average stop latency: {avg_latency['avg_latency_stop']:.3f}s")
    print(f"Average response latency: {avg_latency['avg_latency_resp']:.3f}s")


if __name__ == "__main__":
    main()
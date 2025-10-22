from __future__ import annotations

import argparse
import asyncio
import json
import queue
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import socketio
import torch
import torchaudio.functional as AF
from silero_vad import get_speech_timestamps, load_silero_vad  # 添加VAD功能

### Configuration ###
FRAME_MS = 30
SEND_SR = 16_000
RECV_SR = 24_000
TX_SAMP = int(SEND_SR * FRAME_MS / 1000)
RX_SAMP = int(RECV_SR * FRAME_MS / 1000)
RX_BYTES = RX_SAMP * 2
#####################

# 加载Silero VAD模型
VAD_MODEL = load_silero_vad()

def _mono(sig: np.ndarray) -> np.ndarray:
    return sig if sig.ndim == 1 else sig.mean(axis=1)

def _resample(sig: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return sig
    wav = torch.from_numpy(sig.astype(np.float32) / 32768).unsqueeze(0)
    wav_rs = AF.resample(wav, orig_sr, target_sr)
    return (wav_rs.squeeze().numpy() * 32768).astype(np.int16)

def _chunk(sig: np.ndarray, frame_len: int) -> List[np.ndarray]:
    pad = (-len(sig)) % frame_len
    if pad:
        sig = np.concatenate([sig, np.zeros(pad, dtype=sig.dtype)])
    return [sig[i : i + frame_len] for i in range(0, len(sig), frame_len)]

def _compact_json(obj):
    return json.dumps(obj, separators=(",", ":"))

def is_silent(audio_path: Path, threshold: float = 0.01) -> bool:
    """
    检查音频文件是否全为静音
    
    Args:
        audio_path (Path): 音频文件路径
        threshold (float): 音量阈值，低于此值视为静音
        
    Returns:
        bool: 如果全为静音返回True，否则返回False
    """
    try:
        # 读取音频文件
        wav, sr = sf.read(audio_path, dtype="float32")
        
        # 转换为单声道
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        
        # 计算最大振幅
        max_amplitude = np.max(np.abs(wav))
        
        # 如果最大振幅低于阈值，则视为静音
        return max_amplitude < threshold
        
    except Exception as e:
        print(f"检查静音时出错: {e}")
        return False

class FreezeOmniClient:
    def __init__(self, server_ip: str, inp: Path, out: Path):
        self.server_ip = server_ip
        self.inp = inp
        self.out = out
        self.audio_q = queue.Queue()
        self.pending = bytearray()
        self.muted = False

        self.sio = socketio.Client(
            ssl_verify=False,
            reconnection=True,
            reconnection_attempts=0,
            reconnection_delay=2,
            reconnection_delay_max=30,
            randomization_factor=0.2,
        )

        self.sio.on("connect", self._on_connect)
        self.sio.on("disconnect", self._on_disconnect)
        self.sio.on("audio", self._on_audio)
        self.sio.on("stop_tts", self._on_stop_tts)
        self.sio.on("too_many_users", self._on_too_many)

    # ---------------- Socket.IO callbacks ----------------
    def _on_connect(self):
        print(f"[SIO] ✅ Connected: {self.inp}", flush=True)
        asyncio.run(self._stream())

    def _on_disconnect(self):
        print("[SIO] 🔌 Disconnected", flush=True)

    def _on_audio(self, data: bytes):
        self.audio_q.put(data)
        self.muted = False

    def _on_stop_tts(self):
        print("[SIO] ⏹️  stop_tts → mute", flush=True)
        self.pending.clear()
        self.muted = True

    def _on_too_many(self, *_, **__):
        print("[SIO] ❌ Too many users", file=sys.stderr)
        self.sio.disconnect()

    # ---------------- Main coroutine ----------------
    async def _stream(self):
        wav, sr = sf.read(self.inp, dtype="int16")
        wav = _mono(wav)
        wav = _resample(wav, sr, SEND_SR)
        tx_frames = _chunk(wav, TX_SAMP)
        total_frames = len(tx_frames)
        frames_written = 0

        with sf.SoundFile(
            self.out, "w", samplerate=RECV_SR, channels=1, subtype="PCM_16"
        ) as fout:
            self.sio.emit("recording-started")
            frame_dur = FRAME_MS / 1000.0

            for frame in tx_frames:
                self.sio.emit(
                    "audio",
                    _compact_json(
                        {"audio": list(frame.tobytes()), "sample_rate": SEND_SR}
                    ),
                )

                while not self.audio_q.empty():
                    self.pending.extend(self.audio_q.get())

                if self.muted:
                    chunk = b""
                else:
                    chunk = self.pending[:RX_BYTES]
                    self.pending = self.pending[RX_BYTES:]

                if len(chunk) < RX_BYTES:
                    chunk += b"\x00" * (RX_BYTES - len(chunk))
                fout.write(np.frombuffer(chunk, dtype=np.int16))
                frames_written += 1

                await asyncio.sleep(frame_dur)

            self.sio.emit("recording-stopped")
            flush_until = time.time() + 1.0
            while time.time() < flush_until and frames_written < total_frames:
                while not self.audio_q.empty():
                    self.pending.extend(self.audio_q.get())
                chunk = b"" if self.muted else self.pending[:RX_BYTES]
                self.pending = self.pending[RX_BYTES:]
                if len(chunk) < RX_BYTES:
                    chunk += b"\x00" * (RX_BYTES - len(chunk))
                fout.write(np.frombuffer(chunk, dtype=np.int16))
                frames_written += 1
                await asyncio.sleep(frame_dur)

            while frames_written < total_frames:
                fout.write(np.zeros(RX_SAMP, dtype=np.int16))
                frames_written += 1

        self.sio.disconnect()
        print(
            f"[DONE] input = {self.inp} | {len(wav) / SEND_SR:.2f}s → output = {sf.info(self.out).duration:.2f}s"
        )

    # ---------------- Public API ----------------
    def run(self):
        url = f"http://{self.server_ip}"
        self.sio.connect(url, transports=["websocket"], wait_timeout=20)
        self.sio.wait()
        if self.sio.connected:
            self.sio.disconnect()


# -----------------------------------------------------------------------------

def process_file(server_ip: str, inp: Path, out: Path, max_retries: int = 3) -> bool:
    """
    处理单个文件，如果输出为静音则重试
    
    Args:
        server_ip (str): 服务器IP
        inp (Path): 输入文件路径
        out (Path): 输出文件路径
        max_retries (int): 最大重试次数
        
    Returns:
        bool: 处理成功返回True，否则返回False
    """
    for attempt in range(max_retries):
        print(f"处理 {inp} (尝试 {attempt+1}/{max_retries})")
        
        # 处理文件
        client = FreezeOmniClient(server_ip, inp, out)
        client.run()
        
        # 检查输出是否为静音
        if is_silent(out):
            print(f"⚠️ 输出为静音: {out}")
            if attempt < max_retries - 1:
                print("重试处理...")
                continue
            else:
                print("❌ 达到最大重试次数，放弃处理")
                return False
        else:
            print(f"✅ 处理成功: {out}")
            return True
    
    return False

def main():
    ap = argparse.ArgumentParser(
        description="OSUM batch client: process wav files from folder or list file"
    )
    #请尽量选择--input_list，更加方便
    ap.add_argument("--server_ip", required=True, help="Server IP or hostname")
    ap.add_argument("--input_dir", help="Folder containing wav files")
    ap.add_argument("--input_list", help="Text file with wav paths, one per line")
    args = ap.parse_args()

    # 收集需要处理的 wav 文件
    wav_files: List[Path] = []
    if args.input_list:
        with open(args.input_list, "r", encoding="utf-8") as f:
            wav_files = [Path(line.strip()) for line in f if line.strip()]
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        wav_files = sorted(input_dir.glob("*.wav"))
    else:
        print("[ERROR] 请提供 --input_dir 或 --input_list")
        sys.exit(1)
    
    # 处理每个文件
    for inp in wav_files:
        # 直接在输入文件同目录下生成输出文件，文件名添加 _output 后缀
        out = inp.with_name(f"{inp.stem}_output{inp.suffix}")
        
        print(f"[RUN] {inp} → {out}")
        
        try:
            success = process_file(args.server_ip, inp, out)
            if success:
                print(f"✅ 处理成功: {out}")
            else:
                print(f"❌ 处理失败: {inp}")
        except Exception as e:
            print(f"❌ 处理失败 {inp}: {e}")


if __name__ == "__main__":
    main()
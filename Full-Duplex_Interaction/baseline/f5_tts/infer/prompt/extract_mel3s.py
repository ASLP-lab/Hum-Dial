import sys
import time
from datetime import datetime
import torch
import torchaudio
import librosa
import logging
import os
import yaml
import numpy as np
import onnxruntime
print(f"当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("/home/node52_tmpdata/hkxie/osum_xlgeng_zy")
from f5_tts.model.mel_processing import mel_spectrogram_torch_aslp
import torchaudio.compliance.kaldi as kaldi

def _extract_spk_embedding(speech):
    # cosyvoice spk pretrain extracted_embedding 
    option = onnxruntime.SessionOptions()
    option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    option.intra_op_num_threads = 1
    campplus_session = onnxruntime.InferenceSession("/home/node52_tmpdata/hkxie/osum_xlgeng_zy/f5_tts/model/campplus.onnx", 
                                                    sess_options=option, 
                                                    providers=["CPUExecutionProvider"])
    feat = kaldi.fbank(speech,
                        num_mel_bins=80,
                        dither=0,
                        sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    embedding = campplus_session.run(None,
                    {campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
    embedding = torch.tensor([embedding]).cpu().detach()#.to(self.device) #[bs,192]
    return embedding


def _prepare_prompt(prompt_wav: str, prompt_dir: str):
    """加载或提取 prompt 中的说话人嵌入与谱图条件。"""
    base = os.path.splitext(os.path.basename(prompt_wav))[0]
    emb_np = os.path.join(prompt_dir, f"{base}_spk_emb.npy")
    mel_np = os.path.join(prompt_dir, f"{base}_mel_spec.npy")
    wav, sr = torchaudio.load(os.path.join(prompt_dir, prompt_wav))
    wav = wav.mean(dim=0, keepdim=True) if wav.size(0) > 1 else wav
    if sr != 16000:
        wav = torchaudio.transforms.Resample(sr, 16000)(wav)

    # --- 新增：音频峰值归一化 ---
    # 1. 计算音频波形的最大绝对值，加一个极小值(epsilon)防止因静音片段导致除以零
    max_abs_val = torch.max(torch.abs(wav)) + 1e-8
    # 2. 将整个波形除以该最大值，使其范围缩放到 [-1, 1]
    wav = wav / max_abs_val
    # --- 归一化结束 ---
    spk_emb = _extract_spk_embedding(wav)
    mel_spec = mel_spectrogram_torch_aslp(
        y=wav,
        n_fft=1024,
        num_mels=80,
        sampling_rate=16000,
        hop_size=160,
        win_size=640,
        fmin=0,
        fmax=8000,
        center=False
    )
    mel_spec = mel_spec.permute(0, 2, 1) # [80,seq_t] -> [seq_t,80]
    # cond = self.extract_cond(mel_spec).to(self.fm_device)
    cond = mel_spec#.to(self.fm_device)
    b, t, d = mel_spec.shape
    target_len = 300  # 3 秒的帧数
    # 如果 mel_spec 的长度不等于 target_len，则进行裁剪或填充   
    if t > target_len:
        start = np.random.randint(0, t - target_len + 1)
        mel_spec = mel_spec[:,start:start + target_len,:]
    elif t < target_len:
        pad_len = target_len - t
        pad = np.zeros((1, pad_len, d), dtype=mel_spec.dtype)
        mel_spec = np.concatenate([mel_spec, pad], axis=0)
    print(f"mel_spec.shape={mel_spec.shape}, spk_emb.shape={spk_emb.shape}")
    np.save(emb_np, spk_emb.cpu().numpy())
    np.save(mel_np, mel_spec.detach().cpu().numpy())
    
root_dir = "/home/node52_tmpdata/hkxie/osum_xlgeng_zy/f5_tts/infer/prompt/拟人"

_prepare_prompt("NEUTRAL.wav", "/home/node52_tmpdata/hkxie/osum_xlgeng_zy/f5_tts/infer/prompt/拟人/")

for dir in os.listdir(root_dir):
    if dir is None or not os.path.isdir(os.path.join(root_dir, dir)):
        continue
    for file_wav in os.listdir(os.path.join(root_dir,dir)):
        if not file_wav.endswith(".wav"):
            continue
        _prepare_prompt(file_wav, os.path.join(root_dir, dir))
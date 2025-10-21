import torch
import sys
import librosa
import numpy as np
import math
from scipy.interpolate import interp1d
from scipy.io.wavfile import write
from pathlib import Path
import os

# 获取当前脚本的绝对路径
current_file_path = Path(__file__).resolve()

# 设置项目根目录（假设脚本位于项目子目录中）
project_root = current_file_path.parent.parent  # 根据实际目录结构调整

# 添加项目根目录到系统路径
sys.path.append(str(project_root))

# 导入模块
from dspgan.extractors import process_wav, __extract_mel
from dspgan.config import preConfiged16K
from dspgan.mel_processing import mel_spectrogram_torch_nhv2
import dspgan.utils as utils

# 设置设备
device = 'cpu'

# 加载配置文件
config_path = project_root / "dspgan" / "16to48.json"
hps = utils.get_hparams_from_file(str(config_path))
# print(f"config_path={config_path}")
# 加载模型
ckpts_dir = project_root / "dspgan" / "ckpts"
sm_mel2f0 = torch.jit.load(str(ckpts_dir / "mel2f0.pt")).to(device)
sm_net_g = torch.jit.load(str(ckpts_dir / "model_export.pt")).to(device)
sm_net_g_3k = torch.jit.load(str(ckpts_dir / "G3k_600000_export.pt")).to(device)

def encode_mel(input_path):
    wav = process_wav(input_path, preConfiged16K)
    mel, _, _ = __extract_mel(wav, preConfiged16K)
    return mel

def interp1d_new(pitch):
    pitch = pitch.reshape(-1)
    nonzero_ids = np.where(pitch > 0)[0]
    if len(nonzero_ids) > 1:
        interp_fn = interp1d(
            nonzero_ids,
            pitch[nonzero_ids],
            fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
            bounds_error=False,
        )
        pitch = interp_fn(np.arange(0, len(pitch)))

    return pitch.reshape(-1, 1)

def decode_mel(pred_mel, out_wav_path):
    #mel映射到lab的mel形式
    # pred_mel = melsexchange(pred_mel)
    # pred_mel = torch.from_numpy(pred_mel)
    import pdb;pdb.set_trace()
    if pred_mel.ndim==3: #[bs,dim,t]
        pred_mel = pred_mel.squeeze(0)
    
    conditions = pred_mel #[dim,seq_t]
    # print() 
    with torch.no_grad():
        conditions = conditions*4 #/ 4
        # conditions = torch.FloatTensor(conditions).unsqueeze(0)
        # conditions = conditions.to(device)
        conditions = conditions.unsqueeze(0)
        lf0_data = sm_mel2f0(conditions)
        lf0_data = lf0_data.detach().cpu().numpy().squeeze()
        lf0, uv = lf0_data[:, 0:1], lf0_data[:, 1:]
        lf0[uv < 0.5] = 0
        lf0 = lf0.squeeze()
        
    mel = pred_mel
    if mel.shape[0] != 80:
        mel = mel.transpose((1, 0))
    uv = np.zeros(lf0.shape, dtype=np.float32)
    uv[lf0 > 0] = 1
    LF0 = np.where(lf0 > 0., np.exp(lf0), 0.)
    LF0 = interp1d_new(LF0)
    LF0 = LF0.reshape([1, -1])
    uv = uv.reshape([1, -1])

    with torch.no_grad():
        mel = mel.float().to(device)
        LF0 = torch.from_numpy(LF0).float().to(device)
        uv = torch.from_numpy(uv).float().to(device)
        mel = torch.unsqueeze(mel, 0)
        LF0 = torch.unsqueeze(LF0, 0)
        uv = torch.unsqueeze(uv, 0)
        if LF0.shape[2] > mel.shape[2] and LF0.shape[2] - mel.shape[2] < 10:
            LF0 = LF0[:, :, :mel.shape[2]]
            uv = uv[:, :, :mel.shape[2]]
        if LF0.shape[2] < mel.shape[2]:
            mel = mel[:, :, :LF0.shape[2]]

        prior_audio_6k, harmonic_noise, uv_upsample = sm_net_g_3k.infer(mel, LF0, uv)
        nhv_mel, res_mel = mel_spectrogram_torch_nhv2(
            mel,
            prior_audio_6k.squeeze(1),
            hps.data.mel_filter_length,
            hps.data.n_mel_channels,
            hps.data.mel_sampling_rate,
            hps.data.mel_hop_length,
            hps.data.mel_win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax,
            hps.data.min_db,
            hps.data.max_abs_value,
            hps.data.min_level_db,
            hps.data.ref_level_db
        )
        prior_audio = sm_net_g(nhv_mel, uv_upsample)
        audio = prior_audio.squeeze().data.cpu().float().numpy()

    audio = audio * hps.data.max_wav_value
    audio = audio.astype(np.int16)
    write(out_wav_path, hps.data.sampling_rate, audio)

def decode_mel_streaming(pred_mel):
    #mel映射到lab的mel形式
    # pred_mel = melsexchange(pred_mel)
    # pred_mel = torch.from_numpy(pred_mel)
    # import pdb;pdb.set_trace()
    if pred_mel.ndim==3: #[bs,dim,t]
        pred_mel = pred_mel.squeeze(0)
    
    conditions = pred_mel #[dim,seq_t]
    # print() 
    with torch.no_grad():
        conditions = conditions*4 #/ 4
        # conditions = torch.FloatTensor(conditions).unsqueeze(0)
        # conditions = conditions.to(device)
        conditions = conditions.unsqueeze(0)
        lf0_data = sm_mel2f0(conditions)
        lf0_data = lf0_data.detach().cpu().numpy().squeeze()
        lf0, uv = lf0_data[:, 0:1], lf0_data[:, 1:]
        lf0[uv < 0.5] = 0
        lf0 = lf0.squeeze()
        
    mel = pred_mel
    if mel.shape[0] != 80:
        mel = mel.transpose((1, 0))
    uv = np.zeros(lf0.shape, dtype=np.float32)
    uv[lf0 > 0] = 1
    LF0 = np.where(lf0 > 0., np.exp(lf0), 0.)
    LF0 = interp1d_new(LF0)
    LF0 = LF0.reshape([1, -1])
    uv = uv.reshape([1, -1])

    with torch.no_grad():
        mel = mel.float().to(device)
        LF0 = torch.from_numpy(LF0).float().to(device)
        uv = torch.from_numpy(uv).float().to(device)
        mel = torch.unsqueeze(mel, 0)
        LF0 = torch.unsqueeze(LF0, 0)
        uv = torch.unsqueeze(uv, 0)
        if LF0.shape[2] > mel.shape[2] and LF0.shape[2] - mel.shape[2] < 10:
            LF0 = LF0[:, :, :mel.shape[2]]
            uv = uv[:, :, :mel.shape[2]]
        if LF0.shape[2] < mel.shape[2]:
            mel = mel[:, :, :LF0.shape[2]]

        prior_audio_6k, harmonic_noise, uv_upsample = sm_net_g_3k.infer(mel, LF0, uv)
        nhv_mel, res_mel = mel_spectrogram_torch_nhv2(
            mel,
            prior_audio_6k.squeeze(1),
            hps.data.mel_filter_length,
            hps.data.n_mel_channels,
            hps.data.mel_sampling_rate,
            hps.data.mel_hop_length,
            hps.data.mel_win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax,
            hps.data.min_db,
            hps.data.max_abs_value,
            hps.data.min_level_db,
            hps.data.ref_level_db
        )
        prior_audio = sm_net_g(nhv_mel, uv_upsample)
        audio = prior_audio
        # audio = prior_audio.squeeze().data.cpu().float().numpy()

    # audio = audio * hps.data.max_wav_value
    # audio = audio.astype(np.int16)
    return audio


if __name__ == "__main__":
    mel = encode_mel("/home/work_nfs14/code/hkxie/TTS/F5-TTS/src/third_party/dspgan/test_gt.wav")
    decode_mel(torch.from_numpy(mel).to(device), "/home/work_nfs14/code/hkxie/TTS/F5-TTS/src/third_party/dspgan/gen_wav0215.wav")
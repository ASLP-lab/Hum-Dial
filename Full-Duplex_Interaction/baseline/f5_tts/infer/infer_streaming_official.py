import argparse
import torch
import numpy as np
import os
import sys
import yaml
import json
import random
from pathlib import Path
import torchaudio
import torchaudio.compliance.kaldi as kaldi
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
# 将父目录添加到模块搜索路径
sys.path.append(project_root)
print("Python 模块搜索路径:")
for path in sys.path:
    print(f" - {path}")
from omegaconf import OmegaConf
from f5_tts.model.ecapa_tdnn import ECAPA_TDNN
from f5_tts.model import DiT
# from f5_tts.model.backbones.dit_mask_patch import DiT
from f5_tts.model.mel_processing import mel_spectrogram_torch_aslp
from f5_tts.infer.utils_infer import load_model
from f5_tts.infer.utils_infer import (
    mel_spec_type,
    target_rms,
    cross_fade_duration,
    nfe_step,
    cfg_strength,
    sway_sampling_coef,
    speed,
    fix_duration,
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)
import torch
sys.path.append("/home/node57_data/hkxie/workspace/streaming/src")
from third_party.hifigan.models import Generator
import onnxruntime

# 设置默认路径和配置
DEFAULT_CONFIG_PATH = "/home/node60_tmpdata/hkxie/osum_dit/src/f5_tts/configs/fm_10ms_contrasive_ecapa.yaml"
DEFAULT_VOCODER_CONFIG = "/home/node57_data/hkxie/4O/F5-TTS/src/third_party/hifigan/config_streamfm10ms.json"
DEFAULT_VOCODER_CKPT = "/home/node57_data/hkxie/4O/F5-TTS/src/third_party/hifigan/ckpt_hifigan/g_00400000"
# DEFAULT_CKPT_FILE = "/home/node57_data/hkxie/4O/F5-TTS/ckpts/F5TTS_fm_10ms_dspgancosyvoice1/model_300000.pt"
DEFAULT_CKPT_FILE = "/home/node60_tmpdata/hkxie/osum_dit/ckpts/fm_10ms_ecapa_contrasive_hifigancosyvoice1/model_200000.pt"
DEFAULT_CKPT_FILE = "/home/node60_tmpdata/hkxie/osum_dit/ckpts/fm_10ms_ecapa_nocfg_hifigancosyvoice1/model_last.pt"
DEFAULT_CKPT_FILE = "/home/node60_tmpdata/hkxie/osum_dit/ckpts/fm_10ms_ecapa_mix_emilia_hifigancosyvoice1/model_700000.pt"
option = onnxruntime.SessionOptions()
option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
option.intra_op_num_threads = 1
campplus_session = onnxruntime.InferenceSession("/home/node57_data/hkxie/4O/F5-TTS/src/f5_tts/model/campplus.onnx", sess_options=option, providers=["CPUExecutionProvider"])

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

# cosyvoice spk pretrain extracted_embedding 
def _extract_spk_embedding(speech): #campplus_model str查看一下
    feat = kaldi.fbank(speech,
                        num_mel_bins=80,
                        dither=0,
                        sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    embedding = campplus_session.run(None,{campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
    embedding = torch.tensor([embedding]).cpu().detach()#.to(self.device) #[bs,192]
    return embedding

# 加载配置和模型
def load_configuration(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print(f"Loading checkpoint from: {filepath}")
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Checkpoint loaded.")
    return checkpoint_dict

def initialize_model(config, device):
    model_cls = DiT
    model_cfg = OmegaConf.load(config.get("model_cfg", DEFAULT_CONFIG_PATH)).model.arch
    ckpt_file = config.get("ckpt_file", DEFAULT_CKPT_FILE)
    vocab_file = config.get("vocab_file", "")
    print(f"传入loadmodel得device={device}")
    ema_model = load_model(model_cls, model_cfg, ckpt_file, mel_spec_type="hifigan", vocab_file=vocab_file, device=device)
    return ema_model

def initialize_vocoder(device):
    with open(DEFAULT_VOCODER_CONFIG) as f:
        vocoder_config = json.load(f)
    h = AttrDict(vocoder_config)
    generator = Generator(h).to(device)
    state_dict_g = load_checkpoint(DEFAULT_VOCODER_CKPT, device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()
    return generator

# 推理函数
def run_inference(wav_path, token_path,output_path ,ema_model, generator, device):
    # 加载 token
    # token_path = "/home/node60_tmpdata/hkxie/data/token/10wh/part_1/293829757092_qPOkz_92_6229.hubert_code.npy"
    # token_path = "/home/node60_tmpdata/hkxie/data/token/10wh/part_6/187228175172_Fdo9u_VAD14_4.hubert_code.npy"
    token = np.load(token_path)
    token = torch.from_numpy(token).unsqueeze(0)  # 转换为 torch.Tensor [batch, token_seq]

    # import pdb;pdb.set_trace()
    # import torch.profiler
    # wav_path = "/home/work_nfs14/code/hkxie/TTS/MaskGCT/newtest/prompt_wavs/hq/hq_6.wav"
    # wav_path = "/home/node57_data/hkxie/4O/F5-TTS/src/f5_tts/infer/prompt/xielei-15.wav"
    # wav_path = "/home/node57_data/hkxie/4O/streaming_fm/20250406/test2prompt_gxl.wav"
    # 加载音频并生成 mel 频谱
    audio, sample_rate = torchaudio.load(wav_path)
    # make sure mono input
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    # resample if necessary
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        audio = resampler(audio)

    max_amplitude = audio.abs().max()
    if max_amplitude > 0:
        audio = audio / (max_amplitude + 1e-6)

    spk_emb = _extract_spk_embedding(audio).to(device) #可能需要截短？，部分audio可能时长太长
    mel_spec = mel_spectrogram_torch_aslp(y=audio, n_fft=1024, num_mels=80, sampling_rate=16000, hop_size=160, win_size=640, fmin=0, fmax=8000, center=False)
    mel_spec = mel_spec.permute(0, 2, 1) # [80,seq_t] -> [seq_t,80]

    # cond = extract_cond(mel_spec).to(device)
    ref_audio_len = [int(len(token[0]) * 4)]  # 音频长度与 token 长度的比例
    # output_wav_path = os.path.join("/home/node57_data/hkxie/4O/F5-TTS/src/f5_tts/infer/s3token1_fm_streaming20250304_ep20w", f"{os.path.basename(token_path)[:-4]}.wav")
    output_wav_path = os.path.join(output_path, "streaming", f"{os.path.basename(token_path)[:-4]}.wav")
    output_wav_nostreaming_path = os.path.join(output_path, "no_streaming" ,f"{os.path.basename(token_path)[:-4]}.wav")
    # 推理过程
    import time

    # 推理过程：流式推理
    start_time_streaming = time.time()  # 记录流式推理开始时间
    with torch.inference_mode():
        generated = ema_model.sample_streaming(
            cond=mel_spec,
            spk_emb=spk_emb,
            text=token,
            duration=ref_audio_len,
            steps=10,
            cfg_strength=cfg_strength,
            vocoder=generator,
            sway_sampling_coef=sway_sampling_coef,
            output_wav_path=output_wav_path
        )
    end_time_streaming = time.time()  # 记录流式推理结束时间

    streaming_inference_time = end_time_streaming - start_time_streaming  # 计算流式推理耗时
    print(f"Streaming inference time: {streaming_inference_time:.4f} seconds")
    torch.cuda.synchronize(device)
    # 推理过程：非流式推理
    start_time_non_streaming = time.time()  # 记录非流式推理开始时间
    with torch.inference_mode():
        generated, _ = ema_model.sample(
            cond=mel_spec,
            spk_emb=spk_emb,
            text=token,
            duration=ref_audio_len,
            steps=10,
            cfg_strength=cfg_strength,
            vocoder=generator,
            sway_sampling_coef=sway_sampling_coef,
            output_wav_path=output_wav_nostreaming_path
        )
    torch.cuda.synchronize(device)
    end_time_non_streaming = time.time()  # 记录非流式推理结束时间

    non_streaming_inference_time = end_time_non_streaming - start_time_non_streaming  # 计算非流式推理耗时
    print(f"Non-streaming inference time: {non_streaming_inference_time:.4f} seconds")

# /home/node57_data/hkxie/4O/F5-TTS/src/f5_tts/infer/cosyvoice2_token_test
# 遍历文件夹生成 token 路径并进行推理
def process_folder(wav_folder, token_dir ,ema_model, generator, output_path ,device):
    wav_files = [f for f in os.listdir(wav_folder) if f.endswith('.wav')]  # 获取文件夹中的所有 wav 文件

    import pdb;pdb.set_trace()
    for wav_file in wav_files:
        wav_path = os.path.join(wav_folder, wav_file)
        # import pdb;pdb.set_trace()
        # 使用 wav 文件的 basename 生成 token 文件路径
        basename = os.path.splitext(wav_file)[0]
        token_path = os.path.join(token_dir, f"{basename}.npy")
        
        # 确保 token 文件存在
        if os.path.exists(token_path):
            print(f"Processing: {wav_path} with token: {token_path}")
            run_inference(wav_path, token_path, output_path, ema_model, generator, device)
        else:
            print(f"Token file not found for {wav_path},{token_path} skipping...")

# 命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description="Run F5-TTS Inference")
    parser.add_argument('--wav_path', type=str, required=False, default="/home/node57_data/hkxie/4O/streaming_fm/testset/s3token1", help="Path to input WAV file")
    parser.add_argument('--token_path', type=str, required=False, help="Path to token file")
    parser.add_argument('--config_path', type=str, default=DEFAULT_CONFIG_PATH, help="Path to config YAML")
    parser.add_argument('--output_path', type=str, default='./output_test/', help="Path to output")
    parser.add_argument('--device', type=str, default='0',required=False, help="device")
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 设置设备
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cuda:0")
    
    # 加载配置文件
    config = load_configuration(args.config_path)
    
    # 初始化模型和 vocoder
    ema_model = initialize_model(config, device)
    generator = initialize_vocoder(device)
    # import pdb;pdb.set_trace()
    # 执行文件夹中的所有文件的推理
    process_folder(args.wav_path, args.token_path, ema_model, generator, args.output_path, device)

if __name__ == "__main__":
    main()

'''
python infer_streaming_official.py --wav_path /home/node57_data/hkxie/4O/F5-TTS/src/f5_tts/infer/cosyvocie1_test_token --token_path ./

python infer_streaming_official.py \
    --wav_path /home/node57_data/hkxie/4O/streaming_fm/testset/origin_wav \
    --token_path /home/node57_data/hkxie/4O/streaming_fm/testset/s3token1 \
    --output_path /home/node57_data/hkxie/4O/streaming_fm/testset/output_200k

python infer_streaming_official.py \
    --wav_path /home/node57_data/hkxie/4O/F5-TTS/src/f5_tts/infer/0414_token \
    --token_path /home/node57_data/hkxie/4O/F5-TTS/src/f5_tts/infer/0414_token \
    --output_path /home/node57_data/hkxie/4O/streaming_fm/20250417

python infer_streaming_official.py \
    --wav_path /home/node57_data/hkxie/4O/F5-TTS/src/f5_tts/infer/cosyvoice2_token_test \
    --token_path /home/node57_data/hkxie/4O/F5-TTS/src/f5_tts/infer/cosyvoice2_token_test \
    --output_path /home/node60_tmpdata/hkxie/covomix/F5-TTS/testout/20250528_infer

'''

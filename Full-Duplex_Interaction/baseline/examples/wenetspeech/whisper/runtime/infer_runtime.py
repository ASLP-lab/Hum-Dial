import base64
import json
import time

import gradio as gr


import sys


sys.path.insert(0, '../../../../')
from gxl_ai_utils.utils import utils_file
from patches import modelling_qwen2_infer_patch # 打patch
try:
    from wenet.utils.init_tokenizer import init_tokenizer
    from gxl_ai_utils.config.gxl_config import GxlNode
    from wenet.utils.init_model import init_model
    import logging
    import librosa
    import torch
    import torchaudio
except ImportError:
    pass
is_npu = True

try:
    import torch_npu
except ImportError:
    is_npu = False
    print("torch_npu is not available. if you want to use npu, please install it.")


gpu_id = 1


def init_model_my():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s %(message)s')
    checkpoint_path = "/mnt/sfs/asr/ckpt/qwen2_multi_task_4_6gpus_gxl_adapter/full_train_new_pattern_from_epoch9_now_epoch0/step_4999.pt"
    config_path = "../conf/config_llm_huawei_instruct_3B_cosyvoice1-token.yaml"
    args = GxlNode({
        "checkpoint": checkpoint_path,
    })
    configs = utils_file.load_dict_from_yaml(config_path)
    model, configs = init_model(args, configs)
    if is_npu:
        device = torch.device(f'npu:{gpu_id}')
    else:
        device =torch.device(f'cuda:{gpu_id}')
    model = model.to(device)
    tokenizer = init_tokenizer(configs)
    print(model)
    return model, tokenizer, device


model, tokenizer, device = init_model_my()
# model = model.to(model.llama_model.dtype)


def do_resample(input_wav_path, output_wav_path):
    """"""
    waveform, sample_rate = torchaudio.load(input_wav_path)
    # 检查音频的维度
    num_channels = waveform.shape[0]
    # 如果音频是多通道的，则进行通道平均
    if num_channels > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    waveform = torchaudio.transforms.Resample(
        orig_freq=sample_rate, new_freq=16000)(waveform)
    utils_file.makedir_for_file(output_wav_path)
    torchaudio.save(output_wav_path, waveform, 16000)

def get_feat_from_wav_path(input_wav_path):
    """
    获取音频的特征
    Args:
        input_wav_path: str

    Returns:
        feat: tensor, shape=(1, T, 80)
        feat_lens: tensor, shape=(1,)
    """
    timestamp_ms = int(time.time() * 1000)
    now_file_tmp_path_resample = f'~/.cache/.temp/{timestamp_ms}_resample.wav'
    do_resample(input_wav_path, now_file_tmp_path_resample)
    input_wav_path = now_file_tmp_path_resample
    waveform, sample_rate = torchaudio.load(input_wav_path)
    waveform = waveform.squeeze(0)  # (channel=1, sample) -> (sample,)
    window = torch.hann_window(400)
    stft = torch.stft(waveform,
                      400,
                      160,
                      window=window,
                      return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = torch.from_numpy(
        librosa.filters.mel(sr=sample_rate,
                            n_fft=400,
                            n_mels=80))
    mel_spec = filters @ magnitudes

    # NOTE(xcsong): https://github.com/openai/whisper/discussions/269
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    feat = log_spec.transpose(0, 1)
    feat_lens = torch.tensor([feat.shape[0]], dtype=torch.int64).to(device)
    feat = feat.unsqueeze(0).to(device)
    return feat, feat_lens


def infer_tts(input_text):
    prompt = "恳请将如下文本转换为其对应的语音token，力求生成最为流畅、自然的语音。"
    res_text = model.generate_tts(device=device, text=input_text, prompt=prompt)[0]
    return res_text

def infer_s2s(input_wav_path):
    prompt = "先根据语音输入，直接以文字形式进行回答或对话，接着再生成语音token。"
    feat, feat_lens = get_feat_from_wav_path(input_wav_path)
    res_text = model.generate_s2s(wavs=feat, wavs_len=feat_lens, prompt=prompt)[0]
    return res_text

def infer_s2t_chat(input_wav_path, do_sample=True, topk=5, topp=0.9, temperature=1.0,):
    feat, feat_lens = get_feat_from_wav_path(input_wav_path)
    res_text = model.generate4chat(wavs=feat, wavs_len=feat_lens, do_sample=do_sample, top_k=topk, top_p=topp, temperature=temperature)[0]
    print(f'输出结果：{res_text}')
    return res_text

if __name__=="__main__":
    infer_s2t_chat("input_data/guotumianji.wav")
    while True:
        do_sample = input("是否使用采样？(y/n)")
        if do_sample.lower() == 'y':
            do_sample = True
        else:
            do_sample = False
        topk = int(input("topk值："))
        topp = float(input("topp值："))
        temperature = float(input("temperature值："))
        infer_s2t_chat("input_data/guotumianji.wav", do_sample=do_sample, topk=topk, topp=topp, temperature=temperature)
import sys

import torch
import torch_npu
import torchaudio
from torch import dtype

sys.path.insert(0,'../../../../')
from gxl_ai_utils.utils import utils_file
from wenet.utils.init_tokenizer import init_tokenizer
from wenet.text.base_tokenizer import BaseTokenizer
from gxl_ai_utils.config.gxl_config import GxlNode
from wenet.utils.init_model import init_model
import logging
import librosa
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')
config_path = "/mnt/sfs/asr/code/osum_xlgeng_zy/examples/wenetspeech/whisper/conf/config_llm_huawei_instruct-version_cosyvoice1-token.yaml"
checkpoint_path="/mnt/sfs/asr/code/osum_xlgeng_zy/examples/wenetspeech/whisper/exp/epoch_27_LLMinstruct_cosyvoice1_10Wtts_2Khqtts_3Ks2s_5Ws2t/step_24999.pt"
args = GxlNode({
    "checkpoint": checkpoint_path,
})
configs = utils_file.load_dict_from_yaml(config_path)
model, configs = init_model(args, configs)
gpu_id = 7
device = torch.device(f'npu:{gpu_id}')
model = model.to(device)
model.eval()
tokenizer = init_tokenizer(configs)
print(model)
input_wav_path = "/mnt/sfs/asr/update_data/align_cn_noize/roobo_100097437.wav"

def do_decode_for_token(input_wav_path, answer_text=None, speech_token=None):
    input_prompt = "以聊天的方式对以下音频获取相应的回答文本，之后把该回答文本转换为对应的语音token，旨在生成流畅自然的语音。"
    waveform, sample_rate = torchaudio.load(input_wav_path)
    waveform = waveform.mean(dim=0, keepdim=False)
    # print(f'wavform shape: {waveform.shape}')
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
    print(f'feat shape: {feat.shape}')
    print(f'feat_lens: {feat_lens}')
    # label = torch.tensor([label], dtype=torch.int64).to(device)
    if answer_text == None:
        res_text = model.generate_s2s(prompt=input_prompt, wavs=feat, wavs_len=feat_lens)
    elif speech_token == None:
        res_text = model.infer_sample4speech2text_token_teacher_force( prompt=input_prompt, wavs=feat, wavs_len=feat_lens, answer_text=answer_text)
    else:
        res_text = model.infer_sample4speech2text_token_teacher_force2( prompt=input_prompt, wavs=feat, wavs_len=feat_lens,  answer_text=answer_text, speech_token=speech_token)
    return res_text
import json
if __name__ == "__main__":
    import time
    input_wav_path = "/mnt/sfs/asr/update_data/align_cn_noize/roobo_100097437.wav"

    # # 第一次推理
    # input_prompt = "将这段音频的语音内容详细记录为文字稿。"
    # start_time = time.time()  # 开始计时
    # res_text = do_decode(input_wav_path, input_prompt)
    # end_time = time.time()  # 结束计时
    # print(f"推理结果: {res_text}")
    # print(f"第一次推理消耗时间: {end_time - start_time:.2f} 秒")

    # # 第二次推理
    # input_prompt = "将音频转录为文字，同时在每个英文单词和相应中文字符的前后添加时间戳，时间戳格式为<>，且时间单位需精确到0.1秒。"
    # start_time = time.time()  # 开始计时
    # res_text = do_decode(input_wav_path, input_prompt)
    # end_time = time.time()  # 结束计时
    # print(f"推理结果: {res_text}")
    # print(f"第二次推理消耗时间: {end_time - start_time:.2f} 秒")
    while True:
        # input_prompt = input("请输入要转换的音频的提示信息：")
        input_txt = input("请输入语音路径：")
        if "exit" in input_txt:
            continue
        answer_text = input("请输入回答的标准答案,none表示不使用token：")
        speech_token = input("请输入对应的语音token, none表示不使用token：")
        start_time = time.time()  # 开始计时
        if answer_text == "none":
            res = do_decode_for_token(input_txt)
        elif speech_token == "none": 
            res = do_decode_for_token(input_txt, answer_text=answer_text)
        else:
            true_speech_token = json.loads(speech_token)
            true_speech_token = [int(i) for i in true_speech_token]
            res = do_decode_for_token(input_txt, answer_text=answer_text, speech_token=true_speech_token)
        end_time = time.time()  # 结束计时
        print(f"消耗时间: {end_time - start_time:.2f} 秒")
        print(f"推理结果: {res}")

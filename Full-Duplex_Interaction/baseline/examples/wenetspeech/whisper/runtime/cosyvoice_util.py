import sys
import time

from gxl_ai_utils.utils import utils_file


sys.path.insert(0, '../../tts/third_party/Matcha-TTS')
sys.path.append('../../tts')
sys.path.append('.')
from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.file_utils import load_wav
import torchaudio

import json

gpu_id = 0
cosyvoice = CosyVoice('/mnt/sfs/asr/ckpt/cosyvoice1/CosyVoice-300M-25Hz/CosyVoice-300M-25Hz', gpu_id=gpu_id)
prompt_speech_16k = load_wav("/mnt/sfs/asr/code/zy/osum_xlgeng_zy/examples/wenetspeech/tts/assert/hq_1.wav", 16000)
prompt_path="/mnt/sfs/asr/code/zy/osum_xlgeng_zy/examples/wenetspeech/whisper/runtime/input_data/0315132.wav" # 拟人
# prompt_path="/mnt/sfs/asr/code/zy/osum_xlgeng_zy/examples/wenetspeech/whisper/runtime/input_data/xie.wav"
prompt_speech_22k_local = load_wav(prompt_path, 22050)
import os
os.makedirs('data/output_data', exist_ok=True)
def token_list2wav(token_list):
    timestamp = time.time()*1000
    wav_path = f'./data/output_data/test2cosyvoice1-25hz_gxl_{timestamp}.wav'
    token_list = [int(i) for i in token_list]
    j = cosyvoice.inference_zero_shot_gz_22k(
        '收到好友从远方寄来的生日礼物。',
        '希望你以后能够做的比我还好呦。', prompt_speech_22k_local, stream=False, token_list=token_list)
    utils_file.makedir_for_file(wav_path)
    torchaudio.save(wav_path, j['tts_speech'],cosyvoice.sample_rate)
    return wav_path

def token_list2wav_gbm(file_path, token_list):
    timestamp = time.time() * 1000
    wav_path = f'{file_path}/test2cosyvoice1-25hz_gxl_{timestamp}.wav'
    token_list = [int(i) for i in token_list]
    j = cosyvoice.inference_zero_shot_gz_22k(
        '收到好友从远方寄来的生日礼物。',
        '希望你以后能够做的比我还好呦。', prompt_speech_22k_local, stream=False, token_list=token_list)
    utils_file.makedir_for_file(wav_path)
    torchaudio.save(wav_path, j['tts_speech'], cosyvoice.sample_rate)
    return wav_path

def token_list2wav_wsy(token_list, prompt_speech):
    timestamp = time.time()*1000
    wav_path = f'./data/output_data/test2cosyvoice1-25hz_gxl_{timestamp}.wav'
    token_list = [int(i) for i in token_list]
    j = cosyvoice.inference_zero_shot_gz_22k(
        '收到好友从远方寄来的生日礼物。',
        '希望你以后能够做的比我还好呦。', prompt_speech, stream=False, token_list=token_list)
    utils_file.makedir_for_file(wav_path)
    torchaudio.save(wav_path, j['tts_speech'],cosyvoice.sample_rate)
    return wav_path

def token_list2wav2(token_list, output_file_path):
    token_list = [int(i) for i in token_list]
    j = cosyvoice.inference_zero_shot_gxl(
        '收到好友从远方寄来的生日礼物。',
        '希望你以后能够做的比我还好呦。', prompt_speech_16k, stream=False, token_list=token_list)
    import os
    # os.makedirs('data/output_data', exist_ok=True)
    wav_path = output_file_path
    utils_file.makedir_for_file(wav_path)
    torchaudio.save(wav_path, j['tts_speech'],cosyvoice.sample_rate)
    return wav_path

def token_list2wav3(token_list, output_file_path, prompt_path):
    prompt_speech_16k_local = load_wav(prompt_path, 16000)

    token_list = [int(i) for i in token_list]
    j = cosyvoice.inference_zero_shot_gxl(
        '收到好友从远方寄来的生日礼物。',
        '希望你以后能够做的比我还好呦。', prompt_speech_16k_local, stream=False, token_list=token_list)
    import os
    # os.makedirs('data/output_data', exist_ok=True)
    wav_path = output_file_path
    utils_file.makedir_for_file(wav_path)
    torchaudio.save(wav_path, j['tts_speech'],cosyvoice.sample_rate)
    return wav_path

def token_list2wav3_22k(token_list, output_file_path, prompt_path):
    prompt_speech_22k_local = load_wav(prompt_path, 22050)

    token_list = [int(i) for i in token_list]
    j = cosyvoice.inference_zero_shot_gz_22k(
        '收到好友从远方寄来的生日礼物。',
        '希望你以后能够做的比我还好呦。', prompt_speech_22k_local, stream=False, token_list=token_list)
    import os
    # os.makedirs('data/output_data', exist_ok=True)
    wav_path = output_file_path
    utils_file.makedir_for_file(wav_path)
    torchaudio.save(wav_path, j['tts_speech'],cosyvoice.sample_rate)
    return wav_path
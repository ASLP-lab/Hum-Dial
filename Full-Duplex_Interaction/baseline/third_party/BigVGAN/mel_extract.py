import torch
import bigvgan
import librosa
import os
from meldataset import get_mel_spectrogram
os.environ["OMP_NUM_THREADS"] = "1"
import glob
import json
import logging
import sys
import traceback
# from multiprocessing.pool import Pool
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
from tqdm import tqdm
import textgrid
import json
import warnings
import copy
from tqdm import tqdm
import struct
import pyworld as pw
from scipy.io.wavfile import read, write
import parser
import argparse
import numpy as np
import re
import subprocess
from env import AttrDict
from multiprocessing.pool import Pool

# 设置设备为 CPU
device = torch.device("cuda:7")
# device = torch.device("cpu")

def load_hparams_from_json(path) -> AttrDict:
    with open(path) as f:
        data = f.read()
    return AttrDict(json.loads(data))

config_file = "/home/node57_data/hkxie/4O/F5-TTS/src/third_party/BigVGAN/configs/config.json"
h = load_hparams_from_json(config_file)

def process_utterance(wav_path):
    try:
        wav, sr = librosa.load(wav_path, sr=h.sampling_rate, mono=True) # wav is np.ndarray with shape [T_time] and values in [-1, 1]
        wav = torch.FloatTensor(wav).unsqueeze(0) # wav is FloatTensor with shape [B(1), T_time]
        mel = get_mel_spectrogram(wav, h).to(device) # mel is FloatTensor with shape [B(1), C_mel, T_frame]
        mel = mel.squeeze(0).cpu() # [C_mel, T_frame]
        
    except Exception as e:
        logging.error(f"Error loading file {wav_path}: {e}")
        print("error file",wav_path)
        return None, None
        
    if len(wav) == 0:
        print(f"Warning: {wav_path} contains empty audio.")
        logging.warning(f"Warning: {wav_path} contains empty audio.")
        return None, None  # 空音频直接返回
        
    return wav, mel


def data_preprocessing_item(tg_fn,n_fft,hop_size,win_size,fmin,fmax,sample_rate):
    # print("begin executing data_preprocessing_item")
    wav_fn = tg_fn
    wav, mel = process_utterance(wav_fn)

    return mel


def err_call_back(err):
    print(f'error: {str(err)}')

def process_file(args):
    tg_fn, n_fft, hop_size, win_size, fmin, fmax, sample_rate, save_dir = args
    try:
        file_name = os.path.basename(tg_fn).split(".")[0]
        mel_file_path = os.path.join(save_dir, file_name + '.mel.npy')
        
        # if os.path.exists(mel_file_path):
        #     print(f"{file_name} 已处理，跳过")
        #     return None
        
        mel = data_preprocessing_item(tg_fn, n_fft, hop_size, win_size, fmin, fmax, sample_rate)
        return (mel_file_path, mel)
    except Exception as e:
        err_call_back(e)
        return None
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed_path', type=str, required=True, help='path to the processed data')
    parser.add_argument('--target_path', type=str, required=True, help='path to save the mel-spectrograms')
    args = parser.parse_args()

    sample_rate = 16000
    hop_size = 200
    win_size = 800
    fmin = 0
    fmax = 8000
    n_fft = 1024

    processed_path = args.processed_path
    save_dir = args.target_path

    # 确保保存目录存在
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    """从 wav.scp 文件中获取所有 .wav 文件路径"""
    all_wav_files = []
    with open(processed_path, 'r') as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                _, wav_path = parts
                all_wav_files.append(wav_path)

    # missing_mel_wavs = [
    #     wav_fn for wav_fn in all_wav_files
    #     if not os.path.exists(
    #         os.path.join(
    #             save_dir,
    #             os.path.basename(wav_fn).rsplit(".", 1)[0] + '.mel.npy'
    #         )
    #     )
    # ]
    
    # all_textgrid_fns = missing_mel_wavs
    all_textgrid_fns = all_wav_files
    print(f"all_textgrid_fns={len(all_textgrid_fns)}")
    
    # 动态确定进程数
    num_processes = min(cpu_count(), 28)
    num_processes = 24
    # 创建进程池
    with Pool(processes=num_processes) as p:
        results_buffer = []

        # # 使用 starmap 来传递多个参数给 process_file
        # gen_function = ((tg_fn, n_fft, hop_size, win_size, fmin, fmax, sample_rate, save_dir) 
        #                 for tg_fn in all_textgrid_fns)

        tasks = [
            (tg_fn, n_fft, hop_size, win_size, fmin, fmax, sample_rate, save_dir) 
            for tg_fn in all_textgrid_fns
        ]
        
        for result in tqdm(p.imap_unordered(process_file, tasks), total=len(tasks)):
            if result is not None:
                mel_file_path, mel = result
                if mel_file_path is not None or mel is not None: 
                    results_buffer.append((mel_file_path, mel))
                    
                    # 当缓冲区达到批处理大小时，批量写入磁盘
                    if len(results_buffer) >= 1000:
                        for mel_file_path, mel in results_buffer:
                            np.save(mel_file_path, mel)
                        results_buffer.clear()

        # 写入剩余未写入的结果
        if results_buffer:
            for mel_file_path, mel in results_buffer:
                np.save(mel_file_path, mel.cpu())

    print("All files have been processed.")

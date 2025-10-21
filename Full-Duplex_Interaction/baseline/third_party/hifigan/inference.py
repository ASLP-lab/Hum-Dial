from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from mel_processing import mel_spectrogram_torch_aslp
from models import Generator
from pesq import pesq
from pystoi import stoi
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
# from pypesq import pesq as py_pesq
import pandas as pd
from tqdm import tqdm
import numpy as np
# from metric_utils import mean_std
from scipy.io.wavfile import read

def load_wav(full_path):
    """
    读取音频文件，并确保返回单通道音频数据。
    :param full_path: 音频文件路径
    :return: 单通道音频数据和采样率
    """
    sampling_rate, data = read(full_path)
    import pdb;pdb.set_trace()
    # 检查音频是否为双通道
    if len(data.shape) > 1 and data.shape[1] == 2:  # 双通道音频
        # 将双通道转换为单通道（取平均值）
        data = np.mean(data, axis=1)
    elif len(data.shape) > 1 and data.shape[0] == 2:  # 双通道音频（另一种可能的形状）
        # 将双通道转换为单通道（取平均值）
        data = np.mean(data, axis=0)

    return data, sampling_rate
h = None
device = "cuda:0"


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def get_mel(x):
    # return mel_spectrogram(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)
    return mel_spectrogram_torch_aslp(x, h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax)

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]


def inference(a):
    global device
    generator = Generator(h).to(device)

    state_dict_g = load_checkpoint(a.checkpoint_file, device)
    generator.load_state_dict(state_dict_g['generator'])

    #filelist = os.listdir(a.input_wavs_dir)
    filelist = sorted(glob.glob(os.path.join(a.input_wavs_dir, "*.wav")))

    os.makedirs(a.output_dir, exist_ok=True)

    generator.eval()
    generator.remove_weight_norm()
    
    data = {"filename": [], "pesq_nb":[], "estoi": [], "stoi":[]}
    sr = 8000

    si_snr = ScaleInvariantSignalNoiseRatio()
    with torch.no_grad():
        for i, wavfile in enumerate(tqdm(filelist)):
            filename = os.path.basename(wavfile)
            wav, sr = load_wav(wavfile)
            wav = wav / MAX_WAV_VALUE
            wav = torch.FloatTensor(wav).to(device)
            # import pdb;pdb.setg_trace()
            x = get_mel(wav.unsqueeze(0))
            y_g_hat = generator(x)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')
            #print("mel.shape",x.shape, "input wav.shape",wav.shape, "output_wav.shape", audio.shape)

            output_file = os.path.join(a.output_dir, os.path.splitext(filename)[0] + '_generated.wav')
            write(output_file, h.loss_sampling_rate, audio)
            #print(output_file)
            
            wav = wav.squeeze().cpu().detach().numpy()
            min_len = min(len(wav),len(audio))
    #         #data["pesq"].append(pesq(sr, wav[:min_len],audio[:min_len], 'wb'))
    #         data["pesq_nb"].append(pesq(sr, wav[:min_len], audio[:min_len], 'nb'))
    #         estoi_score = stoi(wav[:min_len], audio[:min_len], sr,  extended=True)
    #         stoi_score = stoi(wav[:min_len], audio[:min_len], sr,  extended=False)
    #         data["filename"].append(filename)
    #         data["estoi"].append(estoi_score)
    #         data["stoi"].append(stoi_score)
    # df = pd.DataFrame(data)
    # df.to_csv(os.path.join(a.output_dir, "_results.csv"), index=False)
    # print("PESQ_nb: {:.2f} ± {:.2f}".format(*mean_std(df["pesq_nb"].to_numpy())))
    # print("ESTOI: {:.2f} ± {:.2f}".format(*mean_std(df["estoi"].to_numpy())))
    # print("STOI: {:.2f} ± {:.2f}".format(*mean_std(df["stoi"].to_numpy())))
    

def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_wavs_dir', default='test_files')
    parser.add_argument('--output_dir', default='generated_files')
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        data = f.read() 

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)
    print("config",h)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print("device",device)
    inference(a)


if __name__ == '__main__':
    main()


import math
import os
import random
import torch
import torch.utils.data
import numpy as np
from librosa.util import normalize
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
import librosa
import torchaudio
from mel_processing import mel_spectrogram_torch_aslp

MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

def load_audio(filepath, sr):
    audio, _ = librosa.load(filepath, sr=sr)
    return torch.FloatTensor(audio).unsqueeze(0)  # (1, T)

def _amp_to_db(x, min_level_db):
    min_level = np.exp(min_level_db / 20 * np.log(10))
    min_level = torch.ones_like(x) * min_level
    return 20 * torch.log10(torch.maximum(min_level, x))


def _normalize(S, max_abs_value, min_db):
    return torch.clamp((2 * max_abs_value) * ((S - min_db) / (-min_db)) - max_abs_value, -max_abs_value, max_abs_value)

mel_basis = {}
hann_window = {}


# def mel_spectrogram_torch_aslp(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
#     if torch.min(y) < -1.:
#         # print('min value is ', torch.min(y))
#         pass
#     if torch.max(y) > 1.:
#         # print('max value is ', torch.max(y))
#         pass

#     global mel_basis, hann_window
#     dtype_device = str(y.dtype) + '_' + str(y.device)
#     fmax_dtype_device = str(fmax) + '_' + dtype_device
#     wnsize_dtype_device = str(win_size) + '_' + dtype_device
#     if fmax_dtype_device not in mel_basis:
#         mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
#         mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)
#     if wnsize_dtype_device not in hann_window:
#         hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

#     y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
#                                 mode='reflect')
#     y = y.squeeze(1)

#     spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
#                       center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)

#     spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)

#     spec = torch.matmul(mel_basis[fmax_dtype_device], spec)

#     spec = _amp_to_db(spec, -115) - 20
#     spec = _normalize(spec, 1, -115)
#     return spec

def mel_spectrogram(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    if torch.min(y) < -1.:
        # print('min value is ', torch.min(y))
        pass
    if torch.max(y) > 1.:
        # print('max value is ', torch.max(y))
        pass
    global mel_basis, hann_window
    #torch.Size([8, 16048])
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)
    # import pdb; pdb.set_trace()
    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1) #torch.Size([8, 16872])
    # print("y.device=",y.device)
    try:
        spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                        center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
        # print(f"spec.shape={spec.shape},y.shape={y.shape}")
    except RuntimeError as e:
        print(f"Error during mel_spectrogram computation: {e}")
        
    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)
    # print(f"spec.shape={spec.shape},y.shape={y.shape}")
    return spec


def get_dataset_filelist(a):
    with open(a.input_training_file, 'r', encoding='utf-8') as fi:
        training_files = [x for x in fi.read().split('\n') if len(x) > 0]

    with open(a.input_validation_file, 'r', encoding='utf-8') as fi:
        validation_files = [x for x in fi.read().split('\n') if len(x) > 0]
        
    return training_files, validation_files


class MelDataset(torch.utils.data.Dataset):
    def __init__(self, training_files, segment_size, n_fft, num_mels,
                 hop_size, win_size, sampling_rate,  fmin, fmax,loss_num_mels,loss_n_fft,loss_hop_size,loss_win_size,loss_sampling_rate,loss_fmax, split=True, shuffle=True, n_cache_reuse=1,
                 device=None, fmax_loss=None, fine_tuning=True, base_mels_path=None ):
        self.audio_files = training_files
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audio_files)
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.split = split
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.fmax_loss = fmax_loss
        self.cached_wav = None
        self.n_cache_reuse = n_cache_reuse
        self._cache_ref_count = 0
        self.device = device
        self.fine_tuning = fine_tuning
        self.base_mels_path = base_mels_path
        self.loss_num_mels = loss_num_mels
        self.loss_n_fft = loss_n_fft
        self.loss_hop_size = loss_hop_size
        self.loss_win_size = loss_win_size
        self.loss_sampling_rate = loss_sampling_rate
        self.loss_fmax = loss_fmax  
        self.resampler = torchaudio.transforms.Resample(orig_freq=self.loss_sampling_rate, new_freq=self.sampling_rate).to("cpu")
        
    def __getitem__(self, index):
        filename = self.audio_files[index]
        if self._cache_ref_count == 0:
            audio, sampling_rate = load_wav(filename)
            audio = audio / MAX_WAV_VALUE #numpy.ndarray
            # print(audio.device)
            if audio.size == 0:
                print("Audio is empty or all zeros, returning None.")
                return None
            if not self.fine_tuning:
                audio = normalize(audio) * 0.95
            self.cached_wav = audio
            audio = torch.FloatTensor(audio) #cpu
            # print(audio.device)
            if sampling_rate != 24000:
                print("存在原始wav采样率不为24k的文件")
            # if sampling_rate != self.sampling_rate: #输入 16k 采样率，输出 24k 采样率
            #     resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=self.sampling_rate)
            #     audio_mel = resampler(audio)
                # print("#输入 16k 采样率，输出 24k 采样率")
                # raise ValueError("{} SR doesn't match target {} SR".format(
                #     sampling_rate, self.sampling_rate))
            self._cache_ref_count = self.n_cache_reuse
        else:
            audio = self.cached_wav
            self._cache_ref_count -= 1
            # print(audio.device)
        
        # audio_mel = audio_mel.unsqueeze(0)
        audio = audio.unsqueeze(0) # (1, T)

        if not self.fine_tuning:
            if self.split:
                if audio.size(1) >= self.segment_size:
                    max_audio_start = audio.size(1) - self.segment_size
                    audio_start = random.randint(0, max_audio_start)
                    audio = audio[:, audio_start:audio_start+self.segment_size]
                else:
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')
            if sampling_rate != self.sampling_rate: #输入 16k 采样率，输出 24k 采样率
                # print()
                # audio = torch.zeros(2, 48000)
                # resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=self.sampling_rate)
                audio_mel = self.resampler(audio)
                # print(audio.device,audio_mel.device)
                # print(audio.shape,audio_mel.shape)
            # mel = mel_spectrogram(audio_mel, self.n_fft, self.num_mels,
            #                       self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
            #                       center=False)
            mel = mel_spectrogram_torch_aslp(audio_mel, self.n_fft, self.num_mels,
                                  self.sampling_rate, self.hop_size, self.win_size, self.fmin, self.fmax,
                                  center=False) #16k
        else:
            if filename.split(".")[-1] == "flac":
                mel = np.load(filename.replace("/wav/","/mel/")+'.npy')
            else:
                mel = np.load(filename.replace("/wav/","/mel/").replace(".wav",".npy"))
            mel = torch.from_numpy(mel)

            if len(mel.shape) < 3:
                mel = mel.unsqueeze(0)

            if self.split:
                frames_per_seg = math.ceil(self.segment_size / self.hop_size)

                if audio.size(1) >= self.segment_size:
                    mel_start = random.randint(0, mel.size(2) - frames_per_seg - 1)
                    mel = mel[:, :, mel_start:mel_start + frames_per_seg]
                    audio = audio[:, mel_start * self.hop_size:(mel_start + frames_per_seg) * self.hop_size]
                else:
                    mel = torch.nn.functional.pad(mel, (0, frames_per_seg - mel.size(2)), 'constant')
                    audio = torch.nn.functional.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        mel_loss = mel_spectrogram(audio, self.loss_n_fft, self.loss_num_mels,
                                  self.loss_sampling_rate, self.loss_hop_size, self.loss_win_size, self.fmin, self.loss_fmax,
                                  center=False) #换成24k mel loss
 
        return (mel.squeeze(), audio.squeeze(0), filename, mel_loss.squeeze())

    def __len__(self):
        return len(self.audio_files)
    
if __name__ == "__main__":
    # 配置文件中的参数
    config = {
        "n_fft": 1024,
        "num_mels": 80,
        "sampling_rate": 16000,
        "hop_size": 160,
        "win_size": 640,
        "fmin": 0,
        "fmax": 8000
    }

    wav_audio = "/home/work_nfs14/code/hkxie/ASR/429143256866_Qm3Eu_240_5739.wav"
    audio = load_audio(wav_audio, config["sampling_rate"])
    audio = audio / MAX_WAV_VALUE
    audio = normalize(audio) * 0.95
    audio = torch.FloatTensor(audio) #cpu
    mel_np1 = np.load("/home/work_nfs14/code/hkxie/TTS/hifi-gan/429143256866_Qm3Eu_240_5739.mel.npy")
    mel_np = torch.FloatTensor(mel_np1).unsqueeze(0) #cpu
    print("mel_np.shape=",mel_np.shape,mel_np)
    # 计算 Mel 频谱
    
    mel = mel_spectrogram(audio, config["n_fft"], config["num_mels"],
                          config["sampling_rate"], config["hop_size"], config["win_size"],
                          config["fmin"], config["fmax"], center=False)

    mel_aslp = mel_spectrogram_torch_aslp(audio, config["n_fft"], config["num_mels"],
                                          config["sampling_rate"], config["hop_size"], config["win_size"],
                                          config["fmin"], config["fmax"], center=False)

    # 打印结果
    # print("Mel Spectrogram (librosa):", mel.shape,mel) #[-12,3]
    print("Mel Spectrogram (torchaudio):", mel_aslp.shape,mel_aslp)#[-1，1]
    from scipy.io.wavfile import read
    import torchaudio
    import pdb;pdb.set_trace()
    sampling_rate, a = read(wav_audio)
    audio = a / MAX_WAV_VALUE
    audio = torch.FloatTensor(audio) #cpu
    b = torchaudio.load(wav_audio)
    
    print(audio,'\n',b)
import json
from importlib.resources import files
import os,sys,io
import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import librosa
from datasets import Dataset as Dataset_
from datasets import load_from_disk
from torch import nn
from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from f5_tts.model.ecapa_tdnn import ECAPA_TDNN
from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import default
import numpy as np
import random
from f5_tts.model.mel_processing import mel_spectrogram_torch_aslp
import onnxruntime


# sys.path.append("/home/node57_data/hkxie/workspace/streamingfm/src/f5_tts/model/lance_test")

# from serialize import TorchShmSerializedList, get_rank
# from utils import load_co, load_li,load_lance, filter_keys
# from aslp.tools.lance_pack import LanceWriter, LanceReader
# from aslp.data.npydata import IntData,FloatData
# torch.multiprocessing.set_start_method('spawn', force=True)
# torch.multiprocessing.current_process().authkey = b'my_authkey'

# option = onnxruntime.SessionOptions()
# option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
# option.intra_op_num_threads = 1
# campplus_session = onnxruntime.InferenceSession("/home/node57_data/hkxie/4O/F5-TTS/src/f5_tts/model/campplus.onnx", sess_options=option, providers=["CPUExecutionProvider"])
# print(f"embedding_model顺利加载")


class OnlineDatasetLance(Dataset):
    def __init__(
        self,
        file_lst,
        lance_conn_path,
        data_path,  # 直接从 `1whtraindataset.txt` 读取数据
        target_sample_rate=16000,
        hop_length=200,
        n_mel_channels=80,
        n_fft=1024,
        win_length=800,
        mel_spec_type="vocos",
        preprocessed_mel=False,
        mel_spec_module: nn.Module | None = None,
    ):
        self.file_lst = file_lst
        self.lance_connections: list[LanceReader] = [
            LanceReader(path, target_cls=cls)
            for path, cls in lance_conn_path
        ]
        self.len = 38105568
        # self.data = self.load_data(data_path)  # 读取数据文件并解析
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_spec_type = mel_spec_type
        self.preprocessed_mel = preprocessed_mel

        if not preprocessed_mel:
            # 保存参数，不直接调用函数
            self.mel_params = {
                "n_fft": n_fft,
                "num_mels": n_mel_channels,
                "sampling_rate": target_sample_rate,
                "hop_size": hop_length,
                "win_size": win_length,
                "fmin": 0,
                "fmax": 8000,
                "center": False
            }

    def load_data(self, file_path):
        """ 从 txt 文件加载数据 """
        data = []
        
        # dataset_path = os.path.join("/home/node60_tmpdata/hkxie/data/token/10wh/",file_path)
        with open(os.path.join("/home/node60_tmpdata/hkxie/osum_dit/data/utt_wav_duration.txt"), 'r', encoding='utf-8') as f:

            for line_num, line in enumerate(tqdm(f, desc="正在加载数据"), start=1):
                parts = line.strip().split(" ")
                if len(parts) < 3:
                    print(f"[WARNING] 第 {line_num} 行字段不足3列，跳过: {line.strip()}")
                    continue

                utt, text_token_path, duration = parts
                data.append(utt)
            return data

    @staticmethod
    def init_data(fileid_list_path):
        li = fileid_list_path + ".li"
        co = fileid_list_path + ".co"
        conns = load_co(co)
        if get_rank() > 0:
            return TorchShmSerializedList([]), conns

        lance_file_lst = load_li(li)
        
        random.seed(42)
        random.shuffle(lance_file_lst)
        lance_file_lst = TorchShmSerializedList(lance_file_lst)
        return lance_file_lst, conns

    def _extract_spk_embedding(self,speech): #campplus_model str查看一下
        feat = kaldi.fbank(speech,
                            num_mel_bins=80,
                            dither=0,
                            sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = campplus_session.run(None,{campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
        embedding = torch.tensor([embedding]).cpu().detach()#.to(self.device) #[bs,192]
        return embedding

    def get_frame_len(self, index):
        """ 计算帧长度 """
        # print(f"self.data[index][duration]={self.data[index]}")
        # duration = min(float(self.file_lst[index].strip().split("|")[-1]), 30.0)
        duration = float(self.file_lst[index].strip().split("|")[-1])
        return duration * self.target_sample_rate / self.hop_length

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # print(f"[DEBUG] __getitem__ called with index: {index}")
        while True:
            utt, row_id, offset,audio_row_id,audio_offset,duration = self.file_lst[index].strip().split("|") 
            duration = float(duration)
            if 1.0 <= duration <= 55.0:
                # print("[DEBUG] Duration within valid range, proceeding.")
                break            

            index = (index + 1) % self.len
            # print(f"[DEBUG] Skipping to next index: {index} due to invalid duration.")


        audio = self.lance_connections[int(audio_offset)].get_datas_by_rowids([int(audio_row_id)])[0].mp3_binary
        audio, _ = librosa.load(io.BytesIO(audio),sr=16000,mono=True)
        audio = torch.FloatTensor(audio.reshape(1, -1))
        
        # # ✅ 步骤 1：归一化（范围归一至 [-1, 1]）
        # max_amplitude = audio.abs().max()
        # if max_amplitude > 0:
        #     audio = audio / (max_amplitude + 1e-6)

        # --- 新增：音频峰值归一化 ---
        # 1. 计算音频波形的最大绝对值，加一个极小值(epsilon)防止因静音片段导致除以零
        max_abs_val = torch.max(torch.abs(audio)) + 1e-8
        # 2. 将整个波形除以该最大值，使其范围缩放到 [-1, 1]
        audio = audio / max_abs_val
        # --- 归一化结束 ---

        # 转换 mel spectrogram
        try:
            mel_spec = mel_spectrogram_torch_aslp(y=audio, **self.mel_params)
            mel_spec = mel_spec.squeeze(0)
            # print(f"[DEBUG] Mel spectrogram shape: {mel_spec.shape}")
        except Exception as e:
            print(f"[ERROR] Failed to generate mel spectrogram: {e}")
            raise e

        # 音频长度标准化为 3 秒
        target_length = int(self.target_sample_rate * 3)  # 3秒对应的帧数

        if audio.shape[1] > target_length:
            # 随机裁剪3秒
            max_offset = audio.shape[1] - target_length
            start = random.randint(0, max_offset)
            audio = audio[:, start:start + target_length]
        elif audio.shape[1] < target_length:
            # 重复补足
            repeat_times = (target_length + audio.shape[1] - 1) // audio.shape[1]  # 向上取整
            audio = audio.repeat(1, repeat_times)[:, :target_length]

        # 说话人嵌入
        try:
            spk_emb = self._extract_spk_embedding(audio)
            # print(f"[DEBUG] Speaker embedding shape: {spk_emb.shape}")
        except Exception as e:
            print(f"[ERROR] Failed to extract speaker embedding: {e}")
            raise e


        # 读取文本 token
        try:
            token = self.lance_connections[int(offset)].get_datas_by_rowids([int(row_id)])[0].data1
            
            text_token = torch.tensor(token, dtype=torch.long)
            # print(f"[DEBUG] Loaded text token: len={len(text_token)}, path={text_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load text token at {text_path}: {e}")
            raise e

        token_seq_len = len(text_token)
        mel_seq_len = mel_spec.shape[1]
        upsample_rate = 4
        target_frames = 400

        if token_seq_len > target_frames:
            # print(f"token_seq_len={token_seq_len} > target_frames{target_frames}")
            start_idx = random.randint(0, token_seq_len - target_frames)
            text_token = text_token[start_idx : start_idx + target_frames]
            mel_spec = mel_spec[:, start_idx*upsample_rate : (start_idx + target_frames)*upsample_rate]
            duration = float(target_frames / 25)
            # print(f"[DEBUG] Truncated to target_frames: {target_frames}, start_idx: {start_idx}")
            # print(f"[DEBUG] Truncated mel_spec shape: {mel_spec.shape}, text_token length: {len(text_token)}")

        return {
            "mel_spec": mel_spec,
            "text": text_token,
            "duration": duration,
            "spk_emb": spk_emb,
        }

class OnlineDataset(Dataset):
    def __init__(
        self,
        data_path,  # 直接从 `1whtraindataset.txt` 读取数据
        target_sample_rate=16000,
        hop_length=200,
        n_mel_channels=80,
        n_fft=1024,
        win_length=800,
        mel_spec_type="vocos",
        preprocessed_mel=False,
        mel_spec_module: nn.Module | None = None,
    ):
        self.data = self.load_data(data_path)  # 读取数据文件并解析
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_spec_type = mel_spec_type
        self.preprocessed_mel = preprocessed_mel

        if not preprocessed_mel:
            # 保存参数，不直接调用函数
            self.mel_params = {
                "n_fft": n_fft,
                "num_mels": n_mel_channels,
                "sampling_rate": target_sample_rate,
                "hop_size": hop_length,
                "win_size": win_length,
                "fmin": 0,
                "fmax": 8000,
                "center": False
            }

    def load_data(self, file_path):
        """ 从 txt 文件加载数据 """
        data = []
        
        dataset_path = os.path.join("/home/node60_tmpdata/hkxie/data/token/10wh/",file_path)
        with open(os.path.join("/home/node60_tmpdata/hkxie/osum_dit/data/10wh/",file_path), 'r', encoding='utf-8') as f:
            # print(f"dataset路径为= {dataset_path}")
            # count = 0
            # import pdb;pdb.set_trace()
            # for line in f:
            for line_num, line in enumerate(tqdm(f, desc="正在加载数据"), start=1):
                parts = line.strip().split(" ")
                if len(parts) < 3:
                    print(f"[WARNING] 第 {line_num} 行字段不足3列，跳过: {line.strip()}")
                    continue

                audio_path, text_token_path, duration = parts
                # # 检查路径格式（你原来用的是 assert，但建议用 if 判断避免程序中断）
                # if not audio_path.endswith(".wav"):
                #     print(f"[WARNING] 第 {line_num} 行 audio_path 格式错误: {audio_path}")
                #     continue

                # if not text_token_path.endswith(".npy"):
                #     print(f"[WARNING] 第 {line_num} 行 text_token_path 格式错误: {text_token_path}")
                #     continue

                # try:
                #     duration = float(duration)
                # except ValueError:
                #     print(f"[WARNING] 第 {line_num} 行 duration 不是合法数字: {duration}")
                #     continue
                
                # # print(f"正在读取P{utt_id}audio文件= {audio_path}")
                # if not os.path.exists(audio_path):
                #     print(f"[WARNING] 第 {line_num} 行音频路径不存在: {audio_path}")
                #     error_log.write(f"[路径缺失] 行 {line_num}: {line.strip()}\n")
                #     continue

                # try:
                #     # 尝试读取音频文件以排查潜在卡死或崩溃问题
                #     audio, sr = torchaudio.load(audio_path)
                # except Exception as e:
                #     print(f"[ERROR] 第 {line_num} 行读取音频失败: {audio_path}, 错误: {e}")
                #     error_log.write(f"[读取失败] 行 {line_num}: {line.strip()} | 错误信息: {e}\n")
                #     continue
                # if not os.path.exists(text_token_path):
                #     print(f"[WARNING] 第 {line_num} 行token文件不存在: {text_token_path}")
                #     continue
                # 一切正常，加入数据
                data.append((audio_path, text_token_path, float(duration)))
            return data

    def _extract_spk_embedding(self,speech): #campplus_model str查看一下
        feat = kaldi.fbank(speech,
                            num_mel_bins=80,
                            dither=0,
                            sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = campplus_session.run(None,{campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
        embedding = torch.tensor([embedding]).cpu().detach()#.to(self.device) #[bs,192]
        return embedding

    def get_frame_len(self, index):
        """ 计算帧长度 """
        # print(f"self.data[index][duration]={self.data[index]}")
        duration = min(self.data[index][-1], 30.0)
        
        return duration * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # print(f"[DEBUG] __getitem__ called with index: {index}")
        while True:
            audio_path,text_path,duration = self.data[index]
            # print(f"[DEBUG] Checking duration: {duration}, audio_path: {audio_path},text_path= {text_path}")

            if 1.0 <= duration <= 55.0:
                # print("[DEBUG] Duration within valid range, proceeding.")
                break            

            index = (index + 1) % len(self.data)
            # print(f"[DEBUG] Skipping to next index: {index} due to invalid duration.")

        # 加载音频
        try:
            audio, source_sample_rate = torchaudio.load(audio_path)
            # print(f"[DEBUG] Loaded audio: shape={audio.shape}, sample_rate={source_sample_rate}")
        except Exception as e:
            print(f"[ERROR] Failed to load audio at {audio_path}: {e}")
            raise e

        # 转为单声道
        if audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
            # print(f"[DEBUG] Converted to mono: shape={audio.shape}")

        # ✅ 步骤 1：归一化（范围归一至 [-1, 1]）
        max_amplitude = audio.abs().max()
        if max_amplitude > 0:
            audio = audio / (max_amplitude + 1e-6)

        # 重采样
        if source_sample_rate != self.target_sample_rate:
            print(f"[DEBUG] Resampling from {source_sample_rate} to {self.target_sample_rate}")
            resampler = torchaudio.transforms.Resample(source_sample_rate, self.target_sample_rate)
            audio = resampler(audio)
            # print(f"[DEBUG] After resampling: shape={audio.shape}")

        # --- 新增：音频峰值归一化 ---
        # 1. 计算音频波形的最大绝对值，加一个极小值(epsilon)防止因静音片段导致除以零
        max_abs_val = torch.max(torch.abs(audio)) + 1e-8
        # 2. 将整个波形除以该最大值，使其范围缩放到 [-1, 1]
        audio = audio / max_abs_val
        # --- 归一化结束 ---

        # 转换 mel spectrogram
        try:
            mel_spec = mel_spectrogram_torch_aslp(y=audio, **self.mel_params)
            mel_spec = mel_spec.squeeze(0)
            # print(f"[DEBUG] Mel spectrogram shape: {mel_spec.shape}")
        except Exception as e:
            print(f"[ERROR] Failed to generate mel spectrogram: {e}")
            raise e

        # 音频长度标准化为 3 秒
        target_length = int(self.target_sample_rate * 3)  # 3秒对应的帧数

        if audio.shape[1] > target_length:
            # 随机裁剪3秒
            max_offset = audio.shape[1] - target_length
            start = random.randint(0, max_offset)
            audio = audio[:, start:start + target_length]
        elif audio.shape[1] < target_length:
            # 重复补足
            repeat_times = (target_length + audio.shape[1] - 1) // audio.shape[1]  # 向上取整
            audio = audio.repeat(1, repeat_times)[:, :target_length]

        # 说话人嵌入
        try:
            spk_emb = self._extract_spk_embedding(audio)
            # print(f"[DEBUG] Speaker embedding shape: {spk_emb.shape}")
        except Exception as e:
            print(f"[ERROR] Failed to extract speaker embedding: {e}")
            raise e


        # 读取文本 token
        try:
            text_token = np.load(text_path)
            # if max(text_token)>4396:
            #     print(f"当前token大于4396 路径为：{text_path}")
                
            #     continue
            text_token = torch.tensor(text_token, dtype=torch.long)
            # print(f"[DEBUG] Loaded text token: len={len(text_token)}, path={text_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load text token at {text_path}: {e}")
            raise e

        token_seq_len = len(text_token)
        mel_seq_len = mel_spec.shape[1]
        upsample_rate = 4
        target_frames = 350

        if token_seq_len > target_frames:
            # print(f"token_seq_len={token_seq_len} > target_frames{target_frames}")
            start_idx = random.randint(0, token_seq_len - target_frames)
            text_token = text_token[start_idx : start_idx + target_frames]
            mel_spec = mel_spec[:, start_idx*upsample_rate : (start_idx + target_frames)*upsample_rate]
            duration = float(target_frames / 25)
            # print(f"[DEBUG] Truncated to target_frames: {target_frames}, start_idx: {start_idx}")
            # print(f"[DEBUG] Truncated mel_spec shape: {mel_spec.shape}, text_token length: {len(text_token)}")

        return {
            "mel_spec": mel_spec,
            "text": text_token,
            "duration": duration,
            "spk_emb": spk_emb,
        }


class CustomDataset(Dataset):
    def __init__(
        self,
        data_path,  # 直接从 `1whtraindataset.txt` 读取数据
        target_sample_rate=16000,
        hop_length=200,
        n_mel_channels=80,
        n_fft=1024,
        win_length=800,
        mel_spec_type="vocos",
        preprocessed_mel=False,
        mel_spec_module: nn.Module | None = None,
    ):
        self.data = self.load_data(data_path)  # 读取数据文件并解析
        self.target_sample_rate = target_sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        self.mel_spec_type = mel_spec_type
        self.preprocessed_mel = preprocessed_mel

        if not preprocessed_mel:
            self.mel_spectrogram = mel_spec_module or MelSpec(
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_mel_channels=n_mel_channels,
                target_sample_rate=target_sample_rate,
                mel_spec_type=mel_spec_type,
            )

    def load_data(self, file_path):
        """ 从 txt 文件加载数据 """
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split("\t")  # 使用制表符分割
                if len(parts) < 4:
                    continue  # 确保有足够的列
                utt_id, mel_path, text_token_path, duration = parts
                duration = float(duration)  # 确保 duration 是 float 类型
                data.append({
                    "utt_id": utt_id,
                    "mel_path": mel_path,
                    "text_token_path": text_token_path,
                    "duration": duration
                })
        return data

    def get_frame_len(self, index):
        """ 计算帧长度 """
        return self.data[index]["duration"] * self.target_sample_rate / self.hop_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        while True:
            row = self.data[index]
            mel_path = row["mel_path"]
            text_path = row["text_token_path"]
            duration = row["duration"]
            
            # 过滤时长范围
            if 1.3 <= duration <= 30.0: 
                break  # 满足要求，继续处理

            index = (index + 1) % len(self.data)  # 防止超出索引

        # 读取 mel 特征
        mel_spec = np.load(mel_path)  # 加载 mel.npy 文件
        mel_spec = torch.tensor(mel_spec, dtype=torch.bfloat16)  # 变成 PyTorch tensor
        
        # 读取文本 token
        text_token = np.load(text_path)  # 加载 .hubert_code.npy
        text_token = torch.tensor(text_token, dtype=torch.long)  # 转为 PyTorch tensor

        return {
            "mel_spec": mel_spec,
            "text": text_token,
            "duration": duration,
        }

# Dynamic Batch Sampler
class DynamicBatchSampler(Sampler[list[int]]):
    """Extension of Sampler that will do the following:
    1.  Change the batch size (essentially number of sequences)
        in a batch to ensure that the total number of frames are less
        than a certain threshold.
    2.  Make sure the padding efficiency in the batch is high.
    3.  Shuffle batches each epoch while maintaining reproducibility.
    """

    def __init__(
        self, sampler: Sampler[int], frames_threshold: int, max_samples=0, random_seed=None, drop_residual: bool = False
    ):
        self.sampler = sampler
        self.frames_threshold = frames_threshold
        self.max_samples = max_samples
        self.random_seed = random_seed
        self.epoch = 0

        indices, batches = [], []
        data_source = self.sampler.data_source

        for idx in tqdm(
            self.sampler, desc="Sorting with sampler... if slow, check whether dataset is provided with duration"
        ):
            indices.append((idx, data_source.get_frame_len(idx)))
        indices.sort(key=lambda elem: elem[1])

        batch = []
        batch_frames = 0
        for idx, frame_len in tqdm(
            indices, desc=f"Creating dynamic batches with {frames_threshold} audio frames per gpu"
        ):
            if batch_frames + frame_len <= self.frames_threshold and (max_samples == 0 or len(batch) < max_samples):
                batch.append(idx)
                batch_frames += frame_len
            else:
                if len(batch) > 1:
                    batches.append(batch)
                if frame_len <= self.frames_threshold:
                    batch = [idx]
                    batch_frames = frame_len
                else:
                    batch = []
                    batch_frames = 0

        if not drop_residual and len(batch) > 1:
            batches.append(batch)

        del indices
        self.batches = batches

        # Ensure even batches with accelerate BatchSamplerShard cls under frame_per_batch setting
        self.drop_last = True

    def set_epoch(self, epoch: int) -> None:
        """Sets the epoch for this sampler."""
        self.epoch = epoch

    def __iter__(self):
        # Use both random_seed and epoch for deterministic but different shuffling per epoch
        if self.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(self.random_seed + self.epoch)
            # Use PyTorch's random permutation for better reproducibility across PyTorch versions
            indices = torch.randperm(len(self.batches), generator=g).tolist()
            batches = [self.batches[i] for i in indices]
        else:
            batches = self.batches
        return iter(batches)

    def __len__(self):
        return len(self.batches)


# Load dataset


def load_dataset(
    dataset_path: str,
    tokenizer: str = "pinyin",
    dataset_type: str = "CustomDataset",
    audio_type: str = "raw",
    mel_spec_module: nn.Module | None = None,
    mel_spec_kwargs: dict = dict(),
) -> CustomDataset:
    """
    dataset_type    - "CustomDataset" if you want to use tokenizer name and default data path to load for train_dataset
                    - "CustomDatasetPath" if you just want to pass the full path to a preprocessed dataset without relying on tokenizer
    """
    print("Loading dataset ...")

    if dataset_type == "CustomDataset":
        print(f"使用{dataset_type}")
        train_dataset = CustomDataset(
            data_path=dataset_path,
            preprocessed_mel=True,  # 这里假设 mel 已经预处理，若没有需要修改
            mel_spec_module=mel_spec_module,
            **mel_spec_kwargs,
        )
        return train_dataset
    elif dataset_type == "OnlineDataset":
        print(f"使用{dataset_type}")
        train_dataset = OnlineDataset(
            data_path=dataset_path,
            preprocessed_mel=False,  # 这里假设 mel 需要预处理
            mel_spec_module=mel_spec_module,
            **mel_spec_kwargs,
        )
        return train_dataset
    elif dataset_type == "OnlineLance":
        print(f"使用{dataset_type}")
        train_dataset = OnlineDatasetLance(
            *OnlineDatasetLance.init_data("/home/node60_tmpdata/hkxie/data/token/emilia/wav_token/wav_token"),
            data_path=dataset_path,
            preprocessed_mel=False,  # 这里假设 mel 需要预处理
            mel_spec_module=mel_spec_module,
            **mel_spec_kwargs,
        )
        return train_dataset
# /home/node60_tmpdata/hkxie/data/token/10wh//1whtraindataset.txt




# collation

def collate_fn(batch):
    # batch 是一个 list，包含多个样本，每个样本是一个 dict，mel_spec 是 (mel_dim, seq_len)
    mel_specs = [item["mel_spec"] for item in batch]  # List of [dim=mel_dim, seq]

    bs = len(mel_specs)  # 获取 batch 大小
    mel_dim = mel_specs[0].shape[0]  # Mel 维度
    target_frames = 300  # 3 秒的帧数 300*10ms
    
    ref_mels = []
    for mel_spec in mel_specs:
        seq_len = mel_spec.shape[1]  # 获取当前 mel 的总帧数
        if seq_len < target_frames:
            # 如果 mel_spec 长度小于 target_frames，进行 repeat 操作
            repeat_times = (target_frames + seq_len - 1) // seq_len  # 计算需要重复多少次
            repeated_mel_spec = mel_spec.repeat(1, repeat_times)  # 在时间维度上重复
            ref_mel = repeated_mel_spec[:, :target_frames]  # 截取至目标长度
            ref_mels.append(ref_mel)
        else:
            # 随机选择起始帧（避免超出边界）
            start_idx = random.randint(0, seq_len - target_frames)
            ref_mel = mel_spec[:, start_idx : start_idx + target_frames]  # 取出 375 帧
            ref_mels.append(ref_mel)

    # 拼接成 batch 维度 [bs, mel_dim, 375]
    ref_mels = torch.stack(ref_mels)
    
    #交换维度，变为 [bs, 375, mel_dim]
    ref_mels = ref_mels.permute(0, 2, 1)
    
    mel_lengths = torch.LongTensor([spec.shape[-1] for spec in mel_specs])
    max_mel_length = mel_lengths.amax()
    
    padded_mel_specs = []
    for spec in mel_specs:  # TODO. maybe records mask for attention here
        padding = (0, max_mel_length - spec.size(-1))
        padded_spec = F.pad(spec, padding, value=0)
        padded_mel_specs.append(padded_spec)

    mel_specs = torch.stack(padded_mel_specs)

    text = [item["text"] for item in batch]
    #pad_sequence 已经返回一个堆叠的张量，因此 stack 是多余的
    text = pad_sequence(text, padding_value=0, batch_first=True)
    # text = torch.stack(text)
    text_lengths = torch.LongTensor([len(item) for item in text])
    
    spk_emb = [item["spk_emb"] for item in batch]
    
    spk_emb = torch.stack(spk_emb)
    # print("spk_emb.shape=",spk_emb.shape)
    return dict(
        mel=mel_specs,
        mel_lengths=mel_lengths,
        text=text,
        text_lengths=text_lengths,
        ref_embed = ref_mels, #[bs, 375, 80]
        spk_emb = spk_emb, #[bs,128]
    )

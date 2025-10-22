"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

from random import random, randint, seed
import random

from typing import Callable
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint
import torchaudio
import numpy as np
import time
from scipy.io.wavfile import write
from f5_tts.model.ecapa_tdnn import ECAPA_TDNN
from f5_tts.model.modules import MelSpec
from f5_tts.model.utils import (
    default,
    exists,
    lens_to_mask,
    list_str_to_idx,
    list_str_to_tensor,
    mask_from_frac_lengths,
)


# 设置固定随机种子
seed = 42  # 可自定义
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def hifigan_convert_wav(wav_data, sample_rate=24000):
    wav_data = wav_data.float().cpu()
    if sample_rate != 24000:
        wav_data = torchaudio.transforms.Resample(orig_freq=24000, new_freq=sample_rate)(wav_data)
    audio = wav_data * 32768.0
    audio = audio.numpy().astype('int16')
    return audio


class CFM(nn.Module):
    def __init__(
        self,
        transformer: nn.Module,
        sigma=0.0,
        odeint_kwargs: dict = dict(
            # atol = 1e-5,
            # rtol = 1e-5,
            method="euler"  # 'midpoint'
        ),
        audio_drop_prob=0.3,
        cond_drop_prob=0.2,
        num_channels=None,
        mel_spec_module: nn.Module | None = None,
        mel_spec_kwargs: dict = dict(),
        frac_lengths_mask: tuple[float, float] = (0.7, 1.0),
        vocab_char_map: dict[str:int] | None = None,
    ):
        super().__init__()

        self.frac_lengths_mask = frac_lengths_mask
        
        # 这与论文中的 λ 相对应
        self.contrastive_weight = 0.025 
        # mel spec
        self.mel_spec = default(mel_spec_module, MelSpec(**mel_spec_kwargs))
        num_channels = default(num_channels, self.mel_spec.n_mel_channels)
        self.num_channels = num_channels

        # classifier-free guidance
        self.audio_drop_prob = audio_drop_prob
        self.cond_drop_prob = cond_drop_prob
        
        print(self.__dict__.keys())

        # transformer
        self.transformer = transformer
        dim = transformer.dim
        self.dim = dim

        # print(1)
        # print(dir(self))
        # import pdb;pdb.set_trace()
        self.transformer.forward = torch.compile(self.transformer.forward, dynamic=False, fullgraph=True)


        # conditional flow related
        self.sigma = sigma

        # sampling related
        self.odeint_kwargs = odeint_kwargs

        # vocab map for tokenization
        self.vocab_char_map = vocab_char_map

    @property
    def device(self):
        return next(self.parameters()).device


    @torch.no_grad()
    def sample(
        self,
        cond: float["b n d"] | float["b nw"],  # noqa: F722\
        spk_emb: torch.FloatTensor,
        text: int["b nt"] | list[str],  # noqa: F722
        duration: int | int["b"],  # noqa: F821
        *,
        lens: int["b"] | None = None,  # noqa: F821
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
        vocoder: Callable[[float["b d n"]], float["b nw"]] | None = None,  # noqa: F722
        no_ref_audio=False,
        duplicate_test=False,
        t_inter=0.1,
        edit_mask=None,
        output_wav_path = './test.wav'
    ):
        self.eval()
        self.text = text
        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        device = self.device
        cond = cond.to(next(self.parameters()).dtype) #["b n d"]
        bs,seq_len,mel_dim = cond.shape
        mask = None
        target_frames = 300  # 3 秒的帧数
        if seq_len < target_frames:
            ref_mel = cond.float()
        else:
            start_idx = randint(0, seq_len - target_frames)
            ref_mel = cond[:, start_idx : start_idx + target_frames].float()
            
        spk_emb = spk_emb.to(self.device)
        cond = ref_mel
        self.cond = cond
        
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            test_cond = F.pad(cond, (0, 0, cond_seq_len, max_duration - 2 * cond_seq_len), value=0.0)

        # def fn(t, x):
        #     device = self.device  # 统一到 self.device 
        #     x = x.to(device);t = t.to(device);cond = self.cond.to(device);text = self.text.to(device)
        #     pred = self.transformer(
        #         x=x, cond=cond, text=text, spk_emb = spk_emb, time=t, mask=mask, drop_audio_cond=False, drop_text=False
        #     )
        #     if cfg_strength < 1e-5:
        #         return pred

        #     null_pred = self.transformer(
        #         x=x, cond=cond, text=text, spk_emb = spk_emb, time=t, mask=mask, drop_audio_cond=True, drop_text=True
        #     )

        #     # self.contrastive_weight 0.05 in init
        #     return (1 - self.contrastive_weight) *(pred + (pred - null_pred) * cfg_strength) + self.contrastive_weight * self.T_hat.to(device)


        def fn(t, x):
            device = self.device  # 统一到 self.device 
            x = x.to(device);t = t.to(device);cond = self.cond.to(device);text = self.text.to(device)
            pos  = torch.arange(start=frame_left,end=frame_left+x.shape[1], device = x.device)
            pred = self.transformer(
                x=x, cond=cond, text=text, spk_emb = spk_emb, time=t, mask=mask, drop_audio_cond=False, drop_text=False,
                pos=pos
            )
            return pred

        # noise input
        # to make sure batch inference result is same with different batch size, and for sure single inference
        # still some difference maybe due to convolutional layers
        y0 = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed)
            y0.append(torch.randn(int(dur), self.num_channels, device=self.device, dtype=cond.dtype))
        y0 = pad_sequence(y0, padding_value=0, batch_first=True)

        
        # import pdb;pdb.set_trace()
        # self.T_hat = self.T_hat.unsqueeze(0).expand(-1, y0.shape[1], -1)
        # 使用 linear interpolation 做时长对齐
        self.T_hat = F.interpolate(
            self.T_hat.transpose(1, 2),  # [1, 80, 300]
            size=y0.shape[1],            # target length
            mode='linear',
            align_corners=True
        ).transpose(1, 2)  # [1, 272, 80]
    
        t_start = 0

        # duplicate test corner for inner time step oberservation
        if duplicate_test:
            t_start = t_inter
            y0 = (1 - t_start) * y0 + t_start * test_cond
            steps = int(steps * (1 - t_start))
        
        t = torch.linspace(t_start, 1, steps + 1, device=self.device, dtype=cond.dtype)
        if sway_sampling_coef is not None:
            t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
        trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        
        # import pdb;pdb.set_trace()
        sampled = trajectory[-1]
        out = sampled
        import time,os
        from datetime import datetime
        # 获取当前时间（时:分:秒）
        current_time = datetime.now().strftime("%H-%M-%S")
        base_name = os.path.basename(output_wav_path)[:-4]
        # out = torch.where(cond_mask, cond, out)
        wav = vocoder(sampled.permute(0, 2, 1).float())
        wav = hifigan_convert_wav(wav.squeeze())
        # 获取目录部分
        output_dir = os.path.dirname(output_wav_path)

        # 确保目录存在，如果不存在则创建
        os.makedirs(output_dir, exist_ok=True)
        # write(f"{output_dir}/{base_name}_nostreaming_time{current_time}.wav", 24000, wav)
        write(output_wav_path, 24000, wav)

        return out, trajectory


    def sample_streaming_osum2(
        self,
        cond: torch.FloatTensor,      # [b, n, d] or [b, nw]
        spk_emb: torch.FloatTensor,
        text: torch.LongTensor | list[str],
        duration: int,
        y0_full,
        *,
        lens: torch.Tensor | None = None,
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
        vocoder: Callable[[torch.Tensor], torch.Tensor] | None = None,
        output_wav_path = "./test,wav",
        output_sample_rate=24000,
        block_size=6,       # 每块帧数 #修改为token size，然后*4上采样到mel帧数
        past_blocks=3,
        current_blocks=2,
        future_blocks=1,
        current_start_block=0,
        wav_cache=None,
        end = False,
        ):
        """
        以块为单位进行流式采样的示例逻辑。
        - past_blocks=1, current_blocks=2, future_blocks=1
        - 每个block_size=24 帧
        - 总窗口大小 = (past_blocks + current_blocks + future_blocks) * block_size
        - 每次只输出 current_blocks * block_size 的部分
        - 然后窗口向后滑动 current_blocks 个 block
        """
        self.eval()

        starttime_docode = time.time()
        self.text = text
        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        device = self.device
        # 假设 cond 的 shape 是 [B, N, D]，其中 N 是帧数，D 是 mel 维度

        # target_len = 300
        # b, n, d = cond.shape

        # # 如果帧数大于 target_len，随机截取一个片段
        # if n > target_len:
        #     start = torch.randint(0, n - target_len + 1, (1,)).item()
        #     cond = cond[:, start:start + target_len, :]
        # # 如果帧数小于 target_len，使用 zero-padding 补齐到 target_len
        # elif n < target_len:
        #     pad_len = target_len - n
        #     pad_tensor = torch.zeros((b, pad_len, d), dtype=cond.dtype, device=cond.device)
        #     cond = torch.cat([cond, pad_tensor], dim=1)
        cond = cond.to(next(self.parameters()).dtype) #["b n d"]
        self.cond = cond
        mask = None
        # print(f"截取完后目前cond形状是{cond.shape}")
        # neural ode
        def fn(t, x):
            # predict flow
            device = self.device  # 统一到 self.device 
            x = x.to(device);t = t.to(device);cond = self.cond.to(device);text = text_window.to(device)

            x = x.clone(memory_format=torch.contiguous_format)
            # print(f"目前cond形状是{cond.shape}")
            # 设置对应的position id
            pos  = torch.arange(start=frame_left,end=frame_left+x.shape[1], device = x.device)
            pred = self.transformer.forward(
                x=x, cond=cond, text=text, spk_emb = spk_emb, time=t, mask=mask, drop_audio_cond=False, 
                drop_text=False, pos=pos#,chunk_offset=frame_left#
            )
            return pred
            # pred,null_pred = torch.chunk(pred,2,dim=0)
            # return pred + (pred - null_pred) * cfg_strength


        # max_duration = 2048*4
        # y0_full = torch.randn(max_duration, self.num_channels, device=self.device, dtype=self.dtype).unsqueeze(0)
        # print(y0_full)  # 每次运行结果相同
        # print("y0_full哈希值为",hash(tuple(y0_full.flatten().tolist())))  # 哈希值相同
        # import pdb;pdb.set_trace()
        # 新增音频平滑参数
        upsample_rate = int(24000/100)              # 上采样倍数 # sample_rate // mel_dim
        wav_smooth_length = 1000                   # 平滑区域长度（采样点数）
        # if current_start_block==1:
        #     past_blocks -= 2 
        local_current_left = (past_blocks*block_size)*4 #24
        wav_left = local_current_left*upsample_rate
        local_current_right = (past_blocks+current_blocks)*block_size*4 #(1+2)*6*4=72
        wav_right = local_current_right*upsample_rate #17280
        # 计算总块数
        # 假设 total_dur = y0_full.shape[1] (即实际有效时长)
        total_dur = text.shape[1]

        # 当前窗口左边界 = current_start_block - past_blocks
        # 右边界 = current_start_block + current_blocks + future_blocks - 1
        past_start = current_start_block - past_blocks
        current_end = current_start_block + current_blocks - 1
        future_end = current_start_block + current_blocks + future_blocks - 1

        # 转换成帧级别的 index（25hz cosyvoice1）
        if current_start_block==3:
            frame_left = 0
            frame_right = (future_end + 1) * block_size  # 右边界是闭区间，所以 +1
        else:
            frame_left = past_start * block_size
            frame_right = (future_end + 1) * block_size  # 右边界是闭区间，所以 +1

        # 边界裁剪
        frame_left = max(0, frame_left)
        frame_right = min(total_dur, frame_right)

        # import pdb;pdb.set_trace()
        # print(f"当前frame_left={frame_left},frame_right={frame_right}")

        # 截取噪声 y0_window: [b, window_frames, num_channels]
        y0_window = y0_full[:, frame_left*4:frame_right*4, :] #y0 mel噪声，与token是4倍上采样倍率

        # 截取对应的 text token
        text_window = self.text[:,frame_left:frame_right] #[batch,token_seq]

        # print(f"当前text_window={text_window},shape={text_window.shape}")
        # 准备时间步
        t = torch.linspace(0, 1, steps + 1, device=self.device, dtype=cond.dtype)
        # if sway_sampling_coef is not None:
        #     t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)

        # ODE 积分
        trajectory = odeint(
            fn,
            y0_window,
            t,
            **self.odeint_kwargs
        )

        sampled_window = trajectory[-1]  # [b, window_size, num_channels]

        #先整体过vocoder合成，然后再截取中间cache下来 
        window_wav = vocoder(sampled_window.permute(0, 2, 1)).to(self.device)

        # print(f"window_wav.shape={window_wav.shape}")

        # import pdb;pdb.set_trace()
        # 核心平滑逻辑
        if wav_cache is not None:
            # 生成渐变权重
            up_weight = torch.linspace(0, 1, wav_smooth_length, device=self.device).view(1,1,-1)
            down_weight = torch.linspace(1, 0, wav_smooth_length, device=self.device).view(1,1,-1)

            current_part = window_wav[..., wav_left:wav_left+wav_smooth_length] # 新块头部

            # 交叉淡化混合
            mixed = wav_cache * down_weight + current_part * up_weight

            # 替换新块头部
            window_wav[..., wav_left:wav_left+wav_smooth_length] = mixed
        else: 
            # pass
            ### 首包的时候wav_cache是none
            #只做（0，1）渐变上升
            up_weight = torch.linspace(0, 1, wav_smooth_length, device=self.device).view(1,1,-1)
            window_wav[..., wav_left:wav_left+wav_smooth_length] = window_wav[..., wav_left:wav_left+wav_smooth_length]*up_weight

        # 更新缓存（保留新块尾部） 
        wav_cache = window_wav[..., wav_right:wav_right+wav_smooth_length]

        if not end:
            # window_wav = window_wav[...,max(wav_left-wav_smooth_length,0):wav_right-wav_smooth_length]
            window_wav = window_wav[...,wav_left:wav_right]
        else:#结尾直接取完
            up_weight = torch.linspace(1, 0, wav_smooth_length, device=self.device).view(1,1,-1)
            window_wav[..., -wav_smooth_length:] = window_wav[..., -wav_smooth_length:]*up_weight
            window_wav = window_wav[...,wav_left:]
        
        window_wav = torch.clamp(window_wav, min=-0.98, max=0.98)

        # print(f"当前window_wav={window_wav.shape}")
        wav = hifigan_convert_wav(window_wav.squeeze(), output_sample_rate)
        endtime_docode = time.time()
        print(f"每个chunk解码时间为{endtime_docode-starttime_docode}")

        # 将时间戳转换为可读格式（如：2025-05-18_14-30-45）
        # formatted_time = datetime.fromtimestamp(endtime_docode).strftime("%Y-%m-%d_%H-%M-%S")
        # write(f"/mnt/sfs/tts/hkxie/tmp/time{formatted_time}_{frame_left}-{frame_right}smooth{wav_smooth_length}.wav", 24000, wav)
        return wav, wav_cache

        # # 实时返回处理后的音频块
        # if wav is not None:
        #     # yield [wav , wav_cache]  # 这里是流式返回处理好的音频
        #     # 3) 统一返回结构
        #     return {'tts_speech': wav, 'cache': wav_cache}

    def sample_streaming(
        self,
        cond: torch.FloatTensor,      # [b, n, d] or [b, nw]
        spk_emb: torch.FloatTensor,
        text: torch.LongTensor | list[str],
        duration: int,
        *,
        lens: torch.Tensor | None = None,
        steps=32,
        cfg_strength=1.0,
        sway_sampling_coef=None,
        seed: int | None = None,
        max_duration=4096,
        vocoder: Callable[[torch.Tensor], torch.Tensor] | None = None,
        output_wav_path = "./test,wav",
        block_size=6,       # 每块帧数 #修改为token size，然后*4上采样到mel帧数
        past_blocks=1,
        current_blocks=2,
        future_blocks=1,
    ):
        """
        以块为单位进行流式采样的示例逻辑。
        - past_blocks=1, current_blocks=2, future_blocks=1
        - 每个block_size=24 帧
        - 总窗口大小 = (past_blocks + current_blocks + future_blocks) * block_size
        - 每次只输出 current_blocks * block_size 的部分
        - 然后窗口向后滑动 current_blocks 个 block
        """
        self.eval()
        self.text = text
        batch, cond_seq_len, device = *cond.shape[:2], cond.device
        device = self.device
        cond = cond.to(next(self.parameters()).dtype) #["b n d"]
        bs,seq_len,mel_dim = cond.shape
        mask = None
        target_frames = 300  # 3 秒的帧数
        if seq_len < target_frames:
            ref_mel = cond.float()
        else:
            start_idx = randint(0, seq_len - target_frames)
            ref_mel = cond[:, start_idx : start_idx + target_frames].float()
            
        spk_emb = spk_emb.to(self.device)
        cond = ref_mel
        self.cond = cond

        # 如果没有 lens，就全用
        if not exists(lens):
            lens = torch.full((batch,), cond_seq_len, device=device, dtype=torch.long)

        if batch > 1:
            mask = lens_to_mask(lens)
        else:
            mask = None

        # 定义 Flow Matching 的预测函数
        def fn(t, x):#, cond_, text_, mask_
            """
            这里将 cond_、text_、mask_ 作为参数传入，以便块内调用。
            """
            
            device = self.device
            x = x.to(device)
            t = t.to(device)
            cond_ = self.cond.to(device)
        
            text_ = text_window.to(device)
            
            # import pdb;pdb.set_trace()
            
            pred = self.transformer(
                x=x, cond=cond_, text=text_,spk_emb = spk_emb, time=t, mask=mask,
                drop_audio_cond=False, drop_text=False,chunk_offset=frame_left
            )
            
            return pred
            # if cfg_strength < 1e-5:
            #     return pred

            # null_pred = self.transformer(
            #     x=x, cond=cond_, text=text_,spk_emb = spk_emb, time=t, mask=mask,
            #     drop_audio_cond=True, drop_text=True,chunk_offset=frame_left
            # )
            # return pred + (pred - null_pred) * cfg_strength

        # 2) 准备整段初始噪声 y0: 假设 total_duration 为整段长度
        #    这里仅示例 batch=1 的情况，多 batch 需再行扩展
        # print(duration)
        # if isinstance(duration, int):
            # total_duration = [duration]
        # total_duration = [duration]
        y0_full = []
        for dur in duration:
            if exists(seed):
                torch.manual_seed(seed) #24
                
            y0_full.append(torch.randn(int(dur), self.num_channels, device=self.device, dtype=cond.dtype))
        y0_full = pad_sequence(y0_full, padding_value=0, batch_first=True)  # [b, max_dur, num_channels]

        #输入是ar输入token，因此我们需要首包等待输入一个chunk+future(12token+6token)并保存作为cache然后进行推理，
        # 之后推理就有了cache作为backward 历史信息(6token)+ 12 + 6 #1s
        
        # 3) 分块相关参数
        window_blocks = past_blocks + current_blocks + future_blocks # 1 2 1
        window_size = window_blocks * block_size   # 例如 4 * 25 = 100
        shift_size = current_blocks * block_size   # 每次滑动 2 * 25 = 50
        batch,token_seq = text.shape
        # 新增音频平滑参数
        upsample_rate = int(24000/100)              # 上采样倍数 # sample_rate // mel_dim
        wav_smooth_length = 600                   # 平滑区域长度（采样点数）
        wav_cache = None                           # 初始化缓存
        
        # 计算总块数
        # 假设 total_dur = y0_full.shape[1] (即实际有效时长)
        total_dur = text.shape[1]
        # 以 chunk_size 为粒度，算出最大块数 (cosyvocietoken1 维度 25hz)
        total_num_blocks = math.ceil(total_dur / block_size)
        
        # 4) 逐块滑动采样
        generated_chunks = []
        generated_wavs = []
        current_start_block = 0

        
        import time
        
        while True:
            start_time = time.time()
            # 当前窗口左边界 = current_start_block - past_blocks
            # 右边界 = current_start_block + current_blocks + future_blocks - 1
            past_start = current_start_block - past_blocks
            current_end = current_start_block + current_blocks - 1
            future_end = current_start_block + current_blocks + future_blocks - 1

            # 转换成帧级别的 index（25hz cosyvoice1）
            frame_left = past_start * block_size
            frame_right = (future_end + 1) * block_size  # 右边界是闭区间，所以 +1

            # 边界裁剪
            frame_left = max(0, frame_left)
            frame_right = min(total_dur, frame_right)

            if frame_left >= total_dur:
                # 已经没有可生成的内容
                break
            
            # import pdb;pdb.set_trace()
            # 截取噪声 y0_window: [b, window_frames, num_channels]
            y0_window = y0_full[:, frame_left*4:frame_right*4, :] #y0 mel噪声，与token是4倍上采样倍率

            # 截取对应的 text token
            text_window = self.text[:,frame_left:frame_right] #[batch,token_seq]

            # 若窗口长度不足 window_size，则可以做 padding；这里示例直接 padding 到 window_size
            # actual_window_len = y0_window.shape[1]
            # if actual_window_len < window_size and :
            #     pad_len = window_size - actual_window_len
            #     y0_window = F.pad(y0_window, (0, 0, 0, pad_len), mode='constant', value=0)

            # 准备时间步
            t = torch.linspace(0, 1, steps + 1, device=self.device, dtype=cond.dtype)
            if sway_sampling_coef is not None:
                t = t + sway_sampling_coef * (torch.cos(torch.pi / 2 * t) - 1 + t)
            # t = t.to(self.device)
            # y0_window = y0_window.to(self.device)
            # import pdb;pdb.set_trace()
            # ODE 积分
            trajectory = odeint(
                fn,
                y0_window,
                t,
                **self.odeint_kwargs
            )
            # trajectory = odeint(
            #     lambda _t, _x: fn(_t, _x, self.cond, text_window, mask),
            #     y0_window,
            #     t,
            #     **self.odeint_kwargs
            # )
            sampled_window = trajectory[-1]  # [b, window_size, num_channels]

            # import pdb;pdb.set_trace()
            # 只取其中的 "current" 部分输出
            # current block 范围 = [current_start_block, current_start_block + current_blocks-1]
            # 换成帧下标 = [current_start_block * block_size, (current_start_block+current_blocks)*block_size)
            current_frame_left = current_start_block * block_size  #mel截取
            current_frame_right = (current_start_block + current_blocks) * block_size 

            # 但是在 sampled_window 里，0 对应 frame_left
            # 所以需要把 current_frame_left 映射到窗口内:
            local_current_left = (current_frame_left - frame_left)*4 #24
            local_current_right = (current_frame_right - frame_left)*4 #72
 
            # 裁剪mel
            current_part = sampled_window[:, local_current_left:local_current_right, :]  # [b, current_blocks*block_size, num_channels]
            generated_chunks.append(current_part)

            #先整体过vocoder合成，然后再截取中间cache下来 #17280
            window_wav = vocoder(sampled_window.permute(0, 2, 1)).to(self.device)
            
            # import pdb;pdb.set_trace()
            # 核心平滑逻辑
            if wav_cache is not None:
                # 生成渐变权重
                up_weight = torch.linspace(0, 1, wav_smooth_length, device=self.device).view(1,1,-1)
                down_weight = torch.linspace(1, 0, wav_smooth_length, device=self.device).view(1,1,-1)
                
                # 取出重叠区域 24000/100
                cache_part = wav_cache[..., -wav_smooth_length:] # 旧缓存尾部
                current_part = window_wav[..., local_current_left*upsample_rate-wav_smooth_length:local_current_left*upsample_rate] # 新块头部
                #local_current_left一直是24吗？
                # 交叉淡化混合
                mixed = wav_cache * down_weight + current_part * up_weight
                
                # 替换新块头部
                window_wav[..., local_current_left*upsample_rate-wav_smooth_length:local_current_left*upsample_rate] = mixed

            # 更新缓存（保留新块尾部）
            # import pdb;pdb.set_trace()
            wav_cache = window_wav[..., local_current_right*upsample_rate-wav_smooth_length:local_current_right*upsample_rate] # 保留双倍长度防止边界问题
            
            window_wav = window_wav[...,max(local_current_left*upsample_rate-wav_smooth_length,0):local_current_right*upsample_rate-wav_smooth_length]
            
            # window_wav = torch.clamp(window_wav, min=-0.98, max=0.98)
            
            wav = hifigan_convert_wav(window_wav.squeeze())
            
            end_time = time.time()
            print(f"当前18token时间{end_time-start_time}")
            # 将有效部分追加到结果中
            generated_wavs.append(window_wav)
            
            
            # generated_wavs.append(current_wav) #每个wav长度等于 sample rate * 2*chunk/80dim
            
            # 滑动到下一个
            current_start_block += current_blocks
            if current_frame_right >= total_dur:
                # 已经到达末尾
                break
        generated_wavs.append(wav_cache)
        # import pdb;pdb.set_trace()
        wav = torch.cat(generated_wavs, dim=-1)
        print(wav.shape)
        wav = hifigan_convert_wav(wav.squeeze())
        import time,os
        from datetime import datetime
        # 获取当前时间（时:分:秒）
        current_time = datetime.now().strftime("%H-%M-%S")
        base_name = os.path.basename(output_wav_path)[:-4]
        # write(output_wav_path, 24000, wav)
        # 获取目录部分
        output_dir = os.path.dirname(output_wav_path)

        # 确保目录存在，如果不存在则创建
        os.makedirs(output_dir, exist_ok=True)
        write(output_wav_path, 24000, wav)
        write(f"{output_dir}/{base_name}_p{past_blocks}c{current_blocks}f{future_blocks}_time{current_time}_smooth{wav_smooth_length}.wav", 24000, wav)
        # write(f"test_streaming_block{block_size}_time{int(time.time())}.wav", 48000, wav)
        # 拼接所有的 current 部分
        out = torch.cat(generated_chunks, dim=1)  # [b, total_generated_frames, num_channels]
        
        # 5) 若有 vocoder，则做后处理
        # if exists(vocoder):
        #     # vocoder 可能需要 [b, n, d]，也可能是 [b, d, n]，根据实际情况做 permute
        #     out = out.permute(0, 2, 1)  # [b, num_channels, T]
        #     out = vocoder(out)
        # yield out
        return out,trajectory

    def logit_normal_sample(self, batch, dtype, device, m=0.0, s=1.0):

        u = torch.randn((batch,),dtype=dtype,device=device) * s + m  # u ~ N(m, s^2)
        samples = torch.sigmoid(u)  # logistic(u) = 1 / (1 + exp(-u))

        return samples

    def forward_nocfg(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        *,
        lens: int["b"] | None = None,  # noqa: F821
        noise_scheduler: str | None = None,
        ref_embed,
        spk_emb,
        use_log_norm: bool = True,
    ):
        # handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device, _σ1 = *inp.shape[:2], inp.dtype, self.device, self.sigma
        # import pdb;pdb.set_trace()
        # lens and mask
        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device)

        mask = lens_to_mask(lens, length=seq_len)  # useless here, as collate_fn will pad to max length in batch

        # # get a random span to mask out for training conditionally
        # frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
        # rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)

        # if exists(mask):
        #     rand_span_mask &= mask

        # mel is x1
        x1 = inp

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        if use_log_norm:
            time = self.logit_normal_sample(batch, dtype=dtype, device=self.device)
        else:
            time = torch.rand((batch,), dtype=dtype, device=self.device)
        # TODO. noise_scheduler

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        # cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)
        cond = ref_embed #ref spk mel
        # transformer and cfg training with a drop rate
        drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
        if random() < self.cond_drop_prob:  # p_uncond in voicebox paper
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        # if want rigorously mask out padding, record in collate_fn in dataset.py, and pass in here
        # adding mask will use more memory, thus also need to adjust batchsampler with scaled down threshold for long sequences
        pred = self.transformer(
            x=φ, cond=cond, text=text, spk_emb = spk_emb,time=time, drop_audio_cond=drop_audio_cond, drop_text=drop_text
        )

        with torch.no_grad():
            pred_with_cond = self.transformer(
                x=φ, cond=cond, text=text, time=time, drop_audio_cond=False, drop_text=False
            )

            pred_without_cond = self.transformer(
                x=φ, cond=cond, text=text, time=time, drop_audio_cond=True, drop_text=True
            )
            
            flow2 = pred_with_cond - pred_without_cond

        w_scale = 0.7
        target = flow + w_scale * flow2

        # flow matching loss
        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss[mask].mean()

        return loss, cond, pred

    def forward_contrasive(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        *,
        lens: int["b"] | None = None,  # noqa: F821
        noise_scheduler: str | None = None,
        ref_embed,
        spk_emb,
        use_log_norm: bool = True,
    ):
        # handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device, _σ1 = *inp.shape[:2], inp.dtype, self.device, self.sigma
        # import pdb;pdb.set_trace()
        # lens and mask
        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device)

        mask = lens_to_mask(lens, length=seq_len)  # useless here, as collate_fn will pad to max length in batch

        # # get a random span to mask out for training conditionally
        # frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
        # rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)

        # if exists(mask):
        #     rand_span_mask &= mask

        # mel is x1
        x1 = inp

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        if use_log_norm:
            time = self.logit_normal_sample(batch, dtype=dtype, device=self.device)
        else:
            time = torch.rand((batch,), dtype=dtype, device=self.device)
        # TODO. noise_scheduler

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        # cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)
        cond = ref_embed #ref spk mel
        # transformer and cfg training with a drop rate
        drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
        if random() < self.cond_drop_prob:  # p_uncond in voicebox paper
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        # if want rigorously mask out padding, record in collate_fn in dataset.py, and pass in here
        # adding mask will use more memory, thus also need to adjust batchsampler with scaled down threshold for long sequences
        pred = self.transformer(
            x=φ, cond=cond, text=text, spk_emb = spk_emb, time=time, drop_audio_cond=drop_audio_cond, drop_text=drop_text
        )

        # ================================================================= #
        # START: 对比流匹配 (Contrastive Flow Matching) 损失计算            #
        # ================================================================= #

        # 步骤 1: 准备正样本和负样本的目标流

        # 1.1: 正样本目标流 (v_pos) 就是 target_flow
        positive_target = flow
        
        # 1.2: 批内负采样，得到负样本目标流 (v_neg)
        # 这种高效的采样方式借鉴自论文的开源代码 triplet_loss.py
        bsz = pred.shape[0]
        # 构造一个索引矩阵，用于从批内选择不同于自身的样本
        choices = torch.tile(torch.arange(bsz), (bsz, 1)).to(device)
        choices.fill_diagonal_(-1.) # 排除自身
        choices = choices.sort(dim=1)[0][:, 1:]
        # 从可行的索引中为每个样本随机选择一个负样本
        choices = choices[torch.arange(bsz), torch.randint(0, bsz-1, (bsz,))]
        negative_target = flow[choices] # 这就是论文中的 v_tilde

        # 步骤 2: 计算损失
        # 注意：这里我们不再使用您原来的 flow2 和 target 计算方式，
        # 而是完全遵循 Contrastive Flow Matching 论文的公式。
        
        # 2.1: 正样本损失 (拉近 v_theta 和 v_pos)
        # 对应论文公式中的 ||v_theta(x_t) - v||^2
        loss_positive = F.mse_loss(pred, positive_target, reduction="none")

        # 2.2: 负样本损失 (推远 v_theta 和 v_neg)
        # 对应论文公式中的 ||v_theta(x_t) - v_tilde||^2
        loss_negative = F.mse_loss(pred, negative_target, reduction="none")

        # 2.3: 组合成最终的对比损失
        # 对应论文公式: L = E[pos_loss - λ * neg_loss]
        # 优化这个损失会同时减小 pos_loss 和增大 neg_loss
        loss = loss_positive - self.contrastive_weight * loss_negative
        
        # ================================================================= #
        # END: 对比流匹配 (Contrastive Flow Matching) 损失计算              #
        # ================================================================= #

        # 应用掩码并计算最终的平均损失
        loss = loss[mask].mean()

        # 返回平均损失，以及用于调试/记录的 cond 和 pred
        return loss, cond, pred
    

    def forward(
        self,
        inp: float["b n d"] | float["b nw"],  # mel or raw wave  # noqa: F722
        text: int["b nt"] | list[str],  # noqa: F722
        *,
        lens: int["b"] | None = None,  # noqa: F821
        noise_scheduler: str | None = None,
        ref_embed,
        spk_emb,
        use_log_norm: bool = True,
    ):
        # handle raw wave
        if inp.ndim == 2:
            inp = self.mel_spec(inp)
            inp = inp.permute(0, 2, 1)
            assert inp.shape[-1] == self.num_channels

        batch, seq_len, dtype, device, _σ1 = *inp.shape[:2], inp.dtype, self.device, self.sigma
        # import pdb;pdb.set_trace()
        # lens and mask
        if not exists(lens):
            lens = torch.full((batch,), seq_len, device=device)

        mask = lens_to_mask(lens, length=seq_len)  # useless here, as collate_fn will pad to max length in batch

        # # get a random span to mask out for training conditionally
        # frac_lengths = torch.zeros((batch,), device=self.device).float().uniform_(*self.frac_lengths_mask)
        # rand_span_mask = mask_from_frac_lengths(lens, frac_lengths)

        # if exists(mask):
        #     rand_span_mask &= mask

        # mel is x1
        x1 = inp

        # x0 is gaussian noise
        x0 = torch.randn_like(x1)

        # time step
        if use_log_norm:
            time = self.logit_normal_sample(batch, dtype=dtype, device=self.device)
        else:
            time = torch.rand((batch,), dtype=dtype, device=self.device)
        # TODO. noise_scheduler

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # only predict what is within the random mask span for infilling
        # cond = torch.where(rand_span_mask[..., None], torch.zeros_like(x1), x1)
        cond = ref_embed #ref spk mel
        # transformer and cfg training with a drop rate
        drop_audio_cond = random() < self.audio_drop_prob  # p_drop in voicebox paper
        if random() < self.cond_drop_prob:  # p_uncond in voicebox paper
            drop_audio_cond = True
            drop_text = True
        else:
            drop_text = False

        # if want rigorously mask out padding, record in collate_fn in dataset.py, and pass in here
        # adding mask will use more memory, thus also need to adjust batchsampler with scaled down threshold for long sequences
        pred = self.transformer(
            x=φ, cond=cond, text=text, spk_emb = spk_emb,time=time, drop_audio_cond=drop_audio_cond, drop_text=drop_text
        )

        with torch.no_grad():
            pred_with_cond = self.transformer(
                x=φ, cond=cond, text=text, spk_emb = spk_emb, time=time, drop_audio_cond=False, drop_text=False
            )

            pred_without_cond = self.transformer(
                x=φ, cond=cond, text=text, spk_emb = spk_emb, time=time, drop_audio_cond=True, drop_text=True
            )
            
            flow2 = pred_with_cond - pred_without_cond

        w_scale = 0.7
        flow = flow + w_scale * flow2

        # ================================================================= #
        # START: 对比流匹配 (Contrastive Flow Matching) 损失计算            #
        # ================================================================= #

        # 步骤 1: 准备正样本和负样本的目标流

        # 1.1: 正样本目标流 (v_pos) 就是 target_flow
        positive_target = flow
        
        # 1.2: 批内负采样，得到负样本目标流 (v_neg)
        # 这种高效的采样方式借鉴自论文的开源代码 triplet_loss.py
        bsz = pred.shape[0]
        # 构造一个索引矩阵，用于从批内选择不同于自身的样本
        choices = torch.tile(torch.arange(bsz), (bsz, 1)).to(device)
        choices.fill_diagonal_(-1.) # 排除自身
        choices = choices.sort(dim=1)[0][:, 1:]
        # 从可行的索引中为每个样本随机选择一个负样本
        choices = choices[torch.arange(bsz), torch.randint(0, bsz-1, (bsz,))]
        negative_target = flow[choices] # 这就是论文中的 v_tilde

        # 步骤 2: 计算损失
        # 注意：这里我们不再使用您原来的 flow2 和 target 计算方式，
        # 而是完全遵循 Contrastive Flow Matching 论文的公式。
        
        # 2.1: 正样本损失 (拉近 v_theta 和 v_pos)
        # 对应论文公式中的 ||v_theta(x_t) - v||^2
        loss_positive = F.mse_loss(pred, positive_target, reduction="none")

        # 2.2: 负样本损失 (推远 v_theta 和 v_neg)
        # 对应论文公式中的 ||v_theta(x_t) - v_tilde||^2
        loss_negative = F.mse_loss(pred, negative_target, reduction="none")

        # 2.3: 组合成最终的对比损失
        # 对应论文公式: L = E[pos_loss - λ * neg_loss]
        # 优化这个损失会同时减小 pos_loss 和增大 neg_loss
        loss = loss_positive - self.contrastive_weight * loss_negative

        # ================================================================= #
        # END: 对比流匹配 (Contrastive Flow Matching) 损失计算              #
        # ================================================================= #

        # 应用掩码并计算最终的平均损失
        loss = loss[mask].mean()

        # 返回平均损失，以及用于调试/记录的 cond 和 pred
        return loss, cond, pred

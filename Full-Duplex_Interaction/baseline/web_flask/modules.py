import sys
import time
from datetime import datetime
import torch
import torchaudio
import librosa
import logging
import os
import yaml
import gradio as gr
import onnxruntime
import json
import numpy as np
import queue
import random
from omegaconf import OmegaConf
import threading
from gxl_ai_utils.config.gxl_config import GxlNode
from gxl_ai_utils.utils import utils_file
import torchaudio.compliance.kaldi as kaldi
if os.getenv("USE_NPU", "false") == "true":
    logging.info("inference on NPU...")
    USE_NPU = True
    from patches import modelling_qwen2_infer_patch # 打patch
    from patches import modelling_fm_infer_npu
else:
    logging.info("inference on GPU...")
    USE_NPU = False
    from patches import modelling_qwen2_infer_gpu # 打patch
    from patches import modelling_fm_infer_gpu
# vad
from silero_vad.model import load_silero_vad
from silero_vad.utils_vad import VADIterator
import torchaudio.compliance.kaldi as k
# interrupt
from wenet_interupt.utils.init_tokenizer import init_tokenizer as init_tokenizer_ipt
from wenet_interupt.utils.init_model import init_model as init_model_ipt
# slm
from wenet.utils.init_tokenizer import init_tokenizer
from wenet.utils.init_model import init_model as init_slm_model
# tts
from f5_tts.model.backbones.dit_mask import DiT
from f5_tts.model.ecapa_tdnn import ECAPA_TDNN
from f5_tts.model.mel_processing import mel_spectrogram_torch_aslp
from f5_tts.infer.utils_infer import load_model as load_f5_tts_model
from f5_tts.infer.utils_infer import (
    nfe_step,
    cfg_strength,
    sway_sampling_coef,
)
from third_party.hifigan.models import Generator as Vocoder_Generator
from omni.vocoder import Qwen2_5OmniToken2WavBigVGANModel, Qwen2_5OmniBigVGANConfig
from scipy.io.wavfile import write as wav_write

def save_wav(wav, sample_rate = 16000):
    wav_name = str(int(time.time())) + ".wav"
    path = os.path.join("temp_out", wav_name)
    os.makedirs("temp_out", exist_ok=True)
    wav_write(path, sample_rate, wav.astype(np.int16))
    return


class IPT_Model:
    def __init__(self,
                 model_path="/home/work_nfs11/gjli/ckpt/wenet_undersdand_and_speech/interrupt_stage1_asr_task4_0.5b_4.14/epoch_1.pt", 
                 config_path = "/home/work_nfs16/gjli/workspaces/wenet_speech_interrupt/examples/wenetspeech/whisper/conf/finetune_whisper_medium_gxl_adapter_interrupt_infer.yaml",
                 gpu_id=0):
        args = GxlNode({
            "checkpoint": model_path,
        })
        configs = utils_file.load_dict_from_yaml(config_path)
        model, configs = init_model_ipt(args, configs)

        if USE_NPU:
            device = torch.device(f'npu:{gpu_id}')
        else:
            device = torch.device(f'cuda:{gpu_id}')
        model = model.to(device).to(torch.bfloat16).eval()
        tokenizer = init_tokenizer_ipt(configs)
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.user_count = 0
        self.warm_up()
        
    def warm_up(self, input_wav_path = "./web_flask/input_data/question3.wav"):
        logging.info("IPT warm up start")
        self.s2t_infer(input_wav_path)
        logging.info("IPT warm up end")
        return
        
    def get_feat_from_wav_path(self, input_wav_path):
        """
        获取音频的特征
        Args:
            input_wav_path: str

        Returns:
            feat: tensor, shape=(1, T, 80)
            feat_lens: tensor, shape=(1,)
        """

        def _do_resample(input_wav_path, output_wav_path):
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

        timestamp_ms = int(time.time() * 1000)
        now_file_tmp_path_resample = f'./temp/{timestamp_ms}_resample.wav'
        _do_resample(input_wav_path, now_file_tmp_path_resample)
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
        feat_lens = torch.tensor([feat.shape[0]], dtype=torch.int64).to(self.device)
        feat = feat.unsqueeze(0).to(self.device).to(torch.bfloat16)
        return feat, feat_lens

    def get_feat_from_numpy(self, gradio_numpy):
        """
        获取音频的特征
        Args:
            input_wav_path: str

        Returns:
            feat: tensor, shape=(1, T, 80)
            feat_lens: tensor, shape=(1,)
        """
        sample_rate, waveform = gradio_numpy
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.to(torch.float)
        else:
            waveform = torch.from_numpy(waveform).to(torch.float)  # 转换为float32
            # @wsy:2. 修改缩进来解决乱回答的问题
            waveform = waveform / 32768.0  # 归一化到[-1.0, 1.0]
        if waveform.dim() > 1:
            waveform = waveform.squeeze()  # 添加通道维度（单声道）
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    
        window = torch.hann_window(400)
        stft = torch.stft(waveform,
                          400,
                          160,
                          window=window,
                          return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2
        # @wsy:2. 修改为16000来解决乱回答的问题
        filters = torch.from_numpy(
            librosa.filters.mel(sr=16000,
                                n_fft=400,
                                n_mels=80))
        mel_spec = filters @ magnitudes

        # NOTE(xcsong): https://github.com/openai/whisper/discussions/269
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        feat = log_spec.transpose(0, 1)
        feat_lens = torch.tensor([feat.shape[0]], dtype=torch.int64).to(self.device)
        feat = feat.unsqueeze(0).to(self.device).to(torch.bfloat16)
        return feat, feat_lens

    def s2t_infer(self, 
                  input_wav, 
                  wav_type='filepath', # 给定Easy Turn模型的prompt
                  input_prompt="​请转录音频内容，并在文末添加以下三种标签之一进行打断判定：<complete>表示语义完整，<incomplete>表示语义不完整，<backchannel>表示简短的附和。"  
                  ):
        if wav_type == 'filepath':
            feat, feat_lens = self.get_feat_from_wav_path(input_wav)
        else:
            feat, feat_lens = self.get_feat_from_numpy(input_wav)
     
        res = self.model.generate_s2t(wavs=feat, 
                                      wavs_len=feat_lens, 
                                      prompt=input_prompt)
        return res[0]
    
    def s2t_infer_fast(self, 
                  input_wav, 
                  wav_type='filepath'
                  ):
        if wav_type == 'filepath':
            feat, feat_lens = self.get_feat_from_wav_path(input_wav)
        else:
            feat, feat_lens = self.get_feat_from_numpy(input_wav)
     
        res = self.model.generate_interrupt_whisper_fc(wavs=feat, 
                                      wavs_len=feat_lens)
        return res
 
class IPTObjectPool:
    # for multiple users
    def __init__(self, size, configs):
        self.pool = self._initialize_pool(size, configs)

    def _initialize_pool(self, size, configs):
        pool = [] 
        for idx in range(size):
            pool.append(
                IPT_Model(
                 model_path=configs.ipt_model_path, 
                 config_path=configs.ipt_config_path,
                 gpu_id=idx)
            )
        return pool

    def acquire(self):
        # Find the object with the minimum user count
        min_user_obj = min(self.pool, key=lambda obj: obj.user_count)
        min_user_obj.user_count += 1
        return min_user_obj

    def release(self, obj):
        if obj.user_count > 0:
            obj.user_count -= 1

    def print_info(self):
        for i, obj in enumerate(self.pool):
            print(f"IPT Object {i} user count: {obj.user_count}")

  
class SLM_Model:
    def __init__(self,
                 model_path="/home/work_nfs16/asr_data/ckpt/understand_model_3B/full_train_llm_3B_epoch_2/step_4999.pt", 
                 config_path = "conf/config_llm_huawei_instruct_3B_cosyvoice1-token.yaml",
                 gpu_id=0):
        args = GxlNode({
            "checkpoint": model_path,
        })
        configs = utils_file.load_dict_from_yaml(config_path)
        model, configs = init_slm_model(args, configs)

        if USE_NPU:
            device = torch.device(f'npu:{gpu_id}')
        else:
            device = torch.device(f'cuda:{gpu_id}')
        model = model.to(device).to(torch.bfloat16).eval()
        tokenizer = init_tokenizer(configs)
        
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.user_count = 0
        self.warm_up()
        
    def warm_up(self, input_wav_path = "./web_flask/input_data/question3.wav"):
        logging.info("SLM warm up start")
        self.infer_s2t(input_wav_path)
        logging.info("SLM warm up end")
        return
        
    def get_feat_from_wav_path(self, input_wav_path):
        """
        获取音频的特征
        Args:
            input_wav_path: str

        Returns:
            feat: tensor, shape=(1, T, 80)
            feat_lens: tensor, shape=(1,)
        """

        def _do_resample(input_wav_path, output_wav_path):
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

        timestamp_ms = int(time.time() * 1000)
        now_file_tmp_path_resample = f'./temp/{timestamp_ms}_resample.wav'
        _do_resample(input_wav_path, now_file_tmp_path_resample)
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
        feat_lens = torch.tensor([feat.shape[0]], dtype=torch.int64).to(self.device)
        feat = feat.unsqueeze(0).to(self.device).to(torch.bfloat16)
        return feat, feat_lens

    def get_feat_from_numpy(self, gradio_numpy):
        """
        获取音频的特征
        Args:
            input_wav_path: str

        Returns:
            feat: tensor, shape=(1, T, 80)
            feat_lens: tensor, shape=(1,)
        """
        sample_rate, waveform = gradio_numpy
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.to(torch.float)
        else:
            waveform = torch.from_numpy(waveform).to(torch.float)  # 转换为float32
            # @wsy:2. 修改缩进来解决乱回答的问题
            waveform = waveform / 32768.0  # 归一化到[-1.0, 1.0]
        if waveform.dim() > 1:
            waveform = waveform.squeeze()  # 添加通道维度（单声道）
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
    
        window = torch.hann_window(400)
        stft = torch.stft(waveform,
                          400,
                          160,
                          window=window,
                          return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2
        # @wsy:2. 修改为16000来解决乱回答的问题
        filters = torch.from_numpy(
            librosa.filters.mel(sr=16000,
                                n_fft=400,
                                n_mels=80))
        mel_spec = filters @ magnitudes

        # NOTE(xcsong): https://github.com/openai/whisper/discussions/269
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        feat = log_spec.transpose(0, 1)
        feat_lens = torch.tensor([feat.shape[0]], dtype=torch.int64).to(self.device)
        feat = feat.unsqueeze(0).to(self.device).to(torch.bfloat16)
        return feat, feat_lens

    def infer_s2t(self, 
                  input_wav, 
                  wav_type='filepath',
                  input_prompt="首先将语音转录为文字，然后对语音内容进行回复，转录和文字之间使用<开始回答>分割。"
                  ):
        if wav_type == 'filepath':
            feat, feat_lens = self.get_feat_from_wav_path(input_wav)
        else:
            feat, feat_lens = self.get_feat_from_numpy(input_wav)
     
        res = self.model.generate(wavs=feat, 
                                      wavs_len=feat_lens, 
                                      prompt=input_prompt)
        return res[0]

    def infer_s2s(self, 
                  input_wav, 
                  wav_type='filepath',
                  input_prompt="先根据语音输入，直接以文字形式进行回答或对话，接着再生成语音token。"
                  ):
        if wav_type == 'filepath':
            feat, feat_lens = self.get_feat_from_wav_path(input_wav)
        else:
            feat, feat_lens = self.get_feat_from_numpy(input_wav)
     
        gen = self.model.generate_s2s(wavs=feat, 
                                      wavs_len=feat_lens, 
                                      prompt=input_prompt)
        return gen
            
    def infer_s2s_no_stream(self, 
                            input_wav, 
                            wav_type='filepath',
                            input_prompt="先根据语音输入，直接以文字形式进行回答或对话，接着再生成语音token。"
                            ):
        if wav_type == 'filepath':
            feat, feat_lens = self.get_feat_from_wav_path(input_wav)
        else:
            feat, feat_lens = self.get_feat_from_numpy(input_wav)
     
        output_text, _, speech_token = self.model.generate_s2s_no_stream(
                                            wavs=feat, 
                                            wavs_len=feat_lens, 
                                            prompt=input_prompt)
        return output_text, speech_token[:,1:]


class SLMObjectPool:
    # for multiple users
    def __init__(self, size, configs):
        self.pool = self._initialize_pool(size, configs)

    def _initialize_pool(self, size, configs):
        pool = [] 
        for idx in range(size):
            pool.append(
                SLM_Model(
                 model_path=configs.slm_model_path, 
                 config_path=configs.slm_config_path,
                 gpu_id=idx)
            )
        return pool

    def acquire(self):
        # Find the object with the minimum user count
        min_user_obj = min(self.pool, key=lambda obj: obj.user_count)
        min_user_obj.user_count += 1
        return min_user_obj

    def release(self, obj):
        if obj.user_count > 0:
            obj.user_count -= 1

    def print_info(self):
        for i, obj in enumerate(self.pool):
            print(f"SLM Object {i} user count: {obj.user_count}")


class TTS_Model:
    def __init__(self,
                 fm_model_path = "/home/node60_tmpdata/hkxie/osum_dit/ckpts/fm_10ms_ecapa_mix_emilia_hifigancosyvoice1/model_700000.pt", 
                 fm_config_path = "/home/node60_tmpdata/hkxie/osum_dit/src/f5_tts/configs/fm_10ms_nocfg_contrasive_emilia.yaml",
                 vc_model_path = "/home/node57_data/hkxie/4O/F5-TTS/src/third_party/hifigan/ckpt_hifigan/g_00400000", 
                 vc_config_path = "/home/node57_data/hkxie/4O/F5-TTS/src/third_party/hifigan/config_streamfm10ms.json",
                 fm_turns = 5,
                 gpu_id=1,
                 vocoder="hifigan"):
        ############ load flow-matching model ##########
        if USE_NPU:
            device = f'npu:{gpu_id}'
        else:
            device = f'cuda:{gpu_id}'

        with open(fm_config_path, 'r') as fin:
            fm_config = yaml.safe_load(fin)

        model_cls = DiT
        model_cfg = OmegaConf.load(fm_config.get("model_cfg", fm_config_path)).model.arch
        ckpt_file = fm_config.get("ckpt_file", fm_model_path)
        vocab_file = fm_config.get("vocab_file", "")
        print(f"当前device={device}")
        fm_model = load_f5_tts_model(
                    model_cls, 
                    model_cfg, 
                    ckpt_file, 
                    mel_spec_type="dspgan", 
                    vocab_file=vocab_file, 
                    device=device,
                    )
        fm_model = fm_model.to(torch.bfloat16).eval()
        # print(f"当前fm模型精度ing为={fm_model.dtype}")
        self.fm_model = fm_model
        self.fm_turns = fm_turns
        self.fm_device = device
        print(f"加载hifigan vocoder进行实验")
        ############ load vocoder model ##########
        class AttrDict(dict):
            def __init__(self, *args, **kwargs):
                super(AttrDict, self).__init__(*args, **kwargs)
                self.__dict__ = self
    
        with open(vc_config_path, 'r') as fin:
            vc_config = json.load(fin)

        h = AttrDict(vc_config)
        generator = Vocoder_Generator(h)
        state_dict_g = torch.load(vc_model_path)
        generator.load_state_dict(state_dict_g['generator'])
        generator = generator.to(torch.device(device)).to(torch.bfloat16).eval()
        generator.remove_weight_norm()
        
        self.vc_hifigan = generator  #只使用hifigan
        '''
        print(f"加载omni_bigvgan vocoder进行实验")
        # mel_spectrogram = torch.randn((1, 80, 75), dtype=torch.float32) #(batch,meldim,seq)
        code2wav_bigvgan_model = Qwen2_5OmniToken2WavBigVGANModel(
            Qwen2_5OmniBigVGANConfig()
        )

        # 加载权重
        state_dict = torch.load("/home/node52_tmpdata/hkxie/omni/bigvgan.ckpt", map_location="cpu")

        code2wav_bigvgan_model.load_state_dict(state_dict)

        # 将模型移到设备上
        code2wav_bigvgan_model = code2wav_bigvgan_model.to(device)
        # 将模型设置为评估模式并禁用梯度s
        code2wav_bigvgan_model.to(torch.bfloat16).eval()
        self.vc_bigvgan = code2wav_bigvgan_model
        '''    
        # self.ecapa = ECAPA_TDNN(in_channels=80, channels=512, embd_dim=128)
        self.vc_device = device
        self.cache = None
        self.token = torch.zeros((1,4096),dtype=torch.int64)
        self.count = 0
        # 说话人嵌入与条件缓存
        self.spk_emb_cache = None
        self.cond_cache = None
        self.y0_full = torch.randn(4096*4, 80, device=self.fm_device, dtype=torch.bfloat16).unsqueeze(0)
        self.wav_chunk = None
        self.in_use = False
        self.warm_up(vocoder='hifigan')
        #self.warm_up(vocoder='bigvgan')

    def warm_up(self,vocoder='hifigan'):
        print("==============warm up tts===============")
        logging.info("TTS warmup start...")
        self.infer_stream(torch.zeros((1, 18), dtype=int, device=self.vc_device), 
                            wav_type="numpy",vocoder=vocoder,
                            )
        self.infer_stream(torch.zeros((1, 12), dtype=int, device=self.vc_device), 
                            wav_type="numpy",vocoder=vocoder,
                            )
        self.infer_stream(torch.zeros((1, 12), dtype=int, device=self.vc_device), 
                            wav_type="numpy",vocoder=vocoder,
                            )
        self.infer_stream(torch.zeros((1, 12), dtype=int, device=self.vc_device), 
                            wav_type="numpy", end=True, vocoder=vocoder,
                            )
        self.count = 0
        self.token.zero_()
        self.reset()

    def extract_cond(self,mel_spec):
        target_frames = 300  # 3 秒的帧数
        seq_len = mel_spec.shape[1]
        model = self.ecapa.eval()
        if seq_len < target_frames:
            ref_mel = mel_spec
        else:
            start_idx = random.randint(0, seq_len - target_frames)
            ref_mel = mel_spec[:, start_idx : start_idx + target_frames]
        
        # ref_mels = ref_mels.squeeze(0).permute(0,2,1).float() #[bs,80,seq_t]
        cond = model(ref_mel)  # [bs, embd_dim]
        return cond.cpu()

    def _extract_spk_embedding(self, speech):
        # cosyvoice spk pretrain extracted_embedding 
        option = onnxruntime.SessionOptions()
        option.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        option.intra_op_num_threads = 1
        campplus_session = onnxruntime.InferenceSession("./f5_tts/model/campplus.onnx", 
                                                        sess_options=option, 
                                                        providers=["CPUExecutionProvider"])
        feat = kaldi.fbank(speech,
                            num_mel_bins=80,
                            dither=0,
                            sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        embedding = campplus_session.run(None,
                        {campplus_session.get_inputs()[0].name: feat.unsqueeze(dim=0).cpu().numpy()})[0].flatten().tolist()
        embedding = torch.tensor([embedding]).cpu().detach()#.to(self.device) #[bs,192]
        return embedding

    def _prepare_prompt(self, prompt_wav: str, prompt_dir: str):
        """加载或提取 prompt 中的说话人嵌入与谱图条件。"""
        base = os.path.splitext(os.path.basename(prompt_wav))[0]
        emb_np = os.path.join(prompt_dir, f"{base}_spk_emb.npy")
        mel_np = os.path.join(prompt_dir, f"{base}_mel_spec.npy")
        # import pdb;pdb.set_trace()
        if os.path.exists(emb_np) and os.path.exists(mel_np):
            spk_emb = torch.from_numpy(np.load(emb_np)).to(self.fm_device).to(torch.bfloat16)
            cond = torch.from_numpy(np.load(mel_np)).to(self.fm_device).to(torch.bfloat16)
        else:
            wav, sr = torchaudio.load(os.path.join(prompt_dir, prompt_wav))
            wav = wav.mean(dim=0, keepdim=True) if wav.size(0) > 1 else wav
            if sr != 16000:
                wav = torchaudio.transforms.Resample(sr, 16000)(wav)

            # --- 新增：音频峰值归一化 ---
            # 1. 计算音频波形的最大绝对值，加一个极小值(epsilon)防止因静音片段导致除以零
            max_abs_val = torch.max(torch.abs(wav)) + 1e-8
            # 2. 将整个波形除以该最大值，使其范围缩放到 [-1, 1]
            wav = wav / max_abs_val
            # --- 归一化结束 ---
            spk_emb = self._extract_spk_embedding(wav).to(self.fm_device)
            mel_spec = mel_spectrogram_torch_aslp(
                y=wav,
                n_fft=1024,
                num_mels=80,
                sampling_rate=16000,
                hop_size=160,
                win_size=640,
                fmin=0,
                fmax=8000,
                center=False
            )
            mel_spec = mel_spec.permute(0, 2, 1) # [80,seq_t] -> [seq_t,80]
            # cond = self.extract_cond(mel_spec).to(self.fm_device)
            cond = mel_spec.to(self.fm_device)
            
            b, t, d = mel_spec.shape
            target_len = 300  # 3 秒的帧数
            # 如果 mel_spec 的长度不等于 target_len，则进行裁剪或填充   
            if t > target_len:
                start = np.random.randint(0, t - target_len + 1)
                mel_spec = mel_spec[start:start + target_len]
            elif t < target_len:
                pad_len = target_len - t
                pad = np.zeros((pad_len, d), dtype=mel_spec.dtype)
                mel_spec = np.concatenate([mel_spec, pad], axis=0)
            # import pdb;pdb.set_trace()
            np.save(emb_np, spk_emb.cpu().numpy())
            np.save(mel_np, cond.detach().cpu().numpy())
            
        spk_emb = spk_emb.to(torch.bfloat16)
        cond = cond.to(torch.bfloat16)
        return spk_emb, cond

    def infer_no_stream(
            self,
            token,
            output_dir="./output_data",
            prompt_dir = "./f5_tts/infer/prompt/女性声优",
            prompt_wav="NEUTRAL.wav", 
            wav_type="numpy",
            output_sample_rate=24000,
            ):
        self.spk_emb_cache, self.cond_cache = self._prepare_prompt(prompt_wav, prompt_dir)

        ref_audio_len = int(len(token[0]) * 4) # 音频长度与 token 长度的比例
        output_wav_path = os.path.join(output_dir ,f"{int(time.time() * 1000)}.wav")

        # 推理过程
        # start_time_streaming = time.time()  # 记录流式推理开始时间
        with torch.inference_mode():
            speech_wav, _ = self.fm_model.fast_sample(
                cond=self.cond_cache,
                spk_emb=self.spk_emb_cache,
                text=token,
                duration=ref_audio_len,
                steps=self.fm_turns,
                cfg_strength=cfg_strength,
                vocoder=self.vc_generator,
                output_sample_rate=output_sample_rate,
                sway_sampling_coef=sway_sampling_coef,
            )

        # end_time_streaming = time.time()  # 记录流式推理结束时间
        # streaming_inference_time = end_time_streaming - start_time_streaming  # 计算流式推理耗时
        # logging.info(f"inference time: {streaming_inference_time:.4f} seconds")

        # save wav file
        if wav_type == "filepath":
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            wav_write(output_wav_path, output_sample_rate, speech_wav)
            return output_wav_path
        else:
            return (output_sample_rate, speech_wav)

    def find_min_sum_index(self, buffer, syn, N, threshold):
        """
        Find the index with the minimum sum of a sliding window in the given audio segment
        and perform operations based on this index.
        """
        arr = syn[0, 0, :]
        L = len(arr)
        mid = L // 2
        kernel = torch.ones(N).to(arr.device)
        window_sums = torch.nn.functional.conv1d(arr.abs().view(1, 1, -1), kernel.view(1, 1, -1), padding=0).squeeze()
        start_index = mid - (N // 2)
        min_sum, min_index = torch.min(window_sums[start_index:], dim=0)
        start_index = max(0, min_index.item() + start_index)
        end_index = min(L, min_index.item() + N + start_index)
        min_sum_real, min_index_real = torch.min(arr[start_index: end_index].abs(), dim=0)
        min_index = min_index_real.item() + start_index
        min_sum = min_sum / N
        syn_clone = syn.clone()
        if min_sum < threshold:
            syn = torch.cat([buffer.clone(), syn[:, :, :min_index]], dim=-1)
            buffer = syn_clone[:, :, min_index:]
        else:
            buffer = torch.cat([buffer, syn_clone], dim=-1)
            syn = None
        return buffer, syn

    def infer_stream(
                    self,
                    token, 
                    output_path="./output_data",
                    prompt_dir = "./f5_tts/infer/prompt/女性声优",
                    prompt_wav="NEUTRAL.wav", 
                    wav_type="numpy",
                    output_sample_rate=16000,
                    end=False,
                    vocoder='hifigan',
                    ):
        # 1. 准备 prompt 条件
        self.spk_emb_cache, self.cond_cache = self._prepare_prompt(prompt_wav, prompt_dir)
        if vocoder == 'hifigan':
            print(f"current_vocoder is {vocoder}")
            vocoder = self.vc_hifigan
           #print(f"current_vocoder is {vocoder}")
        elif vocoder == 'bigvgan':
            print(f"current_vocoder is {vocoder}")
            vocoder = self.vc_bigvgan
            #@print(f"current_vocoder is {vocoder}")
        # 2. 按比例估算音频总长度
        ref_audio_len = int(token.size(1) * 4)
        print(f"当前token序列为{token.shape,token},\n,self.token={self.token.shape}")
        # 3. 填充定长 token 缓存
        seq_len = self.token.shape[1]
        block_size = token.shape[1]
        if self.count == 0:
            assert block_size==18
            # 首包，前6帧为0，其后填入token
            self.token.zero_()
            start_idx = 18
            end_idx = start_idx + block_size
            self.token[:,start_idx:end_idx] = token
        elif self.count == 1:
            # 后续包滑动填充
            start_idx = 36 #因为第一次首包填充是18，之后都是12次token填充
            end_idx = start_idx + block_size
            assert block_size==12
            if end_idx > seq_len:
                # 超出时重置序列缓存
                self.token.zero_()
                self.count = 0
                start_idx = 6
                end_idx = start_idx + block_size
            self.token[:,start_idx:end_idx] = token
        else:
            # 后续包滑动填充
            start_idx = 36 + (self.count-1) * block_size
            end_idx = start_idx + block_size
            assert block_size==12
            if end_idx > seq_len:
                # 超出时重置序列缓存
                self.token.zero_()
                self.count = 0
                start_idx = 6
                end_idx = start_idx + block_size
            self.token[:,start_idx:end_idx] = token

        # 推理过程
        with torch.inference_mode():
            speech_wav,wav_cache = self.fm_model.sample_streaming_osum2(
                cond=self.cond_cache,
                spk_emb = self.spk_emb_cache,
                text=self.token,
                duration=ref_audio_len,
                y0_full=self.y0_full,
                steps=self.fm_turns,
                cfg_strength=cfg_strength,
                vocoder=vocoder,
                sway_sampling_coef=sway_sampling_coef,
                current_start_block = self.count*2+3,
                output_sample_rate = output_sample_rate,
                wav_cache = self.cache,
                end = end,
            )
        self.count += 1 #滑动窗口，下一个输出窗口
        self.cache = wav_cache

        if self.wav_chunk is None:
            self.wav_chunk = speech_wav
        else:
            # --- 音频平滑拼接 ---
            N = 320 # 滑动窗口大小，可根据采样率调整
            threshold = 0.01 # 阈值，可根据实际情况调整
            # 假设 speech_wav 和 self.wav_chunk 都是 numpy 数组，需转为 torch.Tensor
            buffer = torch.from_numpy(self.wav_chunk).unsqueeze(0).unsqueeze(0).float()
            syn = torch.from_numpy(speech_wav).unsqueeze(0).unsqueeze(0).float()
            buffer, syn_out = self.find_min_sum_index(buffer, syn, N, threshold)
            self.wav_chunk = buffer.squeeze().cpu().numpy()
            if syn_out is not None:
                self.wav_chunk = np.concatenate((self.wav_chunk, syn_out.squeeze().cpu().numpy()))
                speech_wav=syn_out.squeeze().cpu().numpy()
            # self.wav_chunk = np.concatenate((self.wav_chunk, speech_wav))
        print(f"当前返回的speech-wav={speech_wav.shape}")

        # 保存输出音频文件，每多一个chunk保存一次
        save_wav(self.wav_chunk, output_sample_rate)

        # 4. 输出结果
        if wav_type == "filepath":
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            ts = int(time.time() * 1000)
            output_wav_path = os.path.join(output_dir, f"tts_{ts}.wav")
            if end is True:
                wav_write(output_wav_path, output_sample_rate, speech_wav)
            return output_wav_path
        else:
            return (output_sample_rate, speech_wav)

    def reset(self): #
        """重置流式推理状态。"""#由总控调用
        self.token = torch.zeros((1,4096),dtype=torch.int64)
        self.cache = None
        self.count = 0
        self.wav_chunk = None
        self.y0_full = torch.randn(4096*4, 80, device=self.fm_device, dtype=torch.bfloat16).unsqueeze(0)


class TTSObjectPool:
    def __init__(self, size, configs):
        self.pool = self._initialize_pool(size, configs)
       
    def _initialize_pool(self, size, configs):
        pool = [] 
        for idx in range(size):
            pool.append(
                TTS_Model(
                 fm_model_path=configs.tts_fm_model_path,
                 fm_config_path=configs.tts_fm_config_path,
                 fm_turns=configs.tts_fm_turns,
                 vc_model_path=configs.tts_vc_model_path,
                 vc_config_path=configs.tts_vc_config_path,
                 gpu_id=idx,
                 vocoder=configs.vocoder)
            )
        return pool

    def acquire(self):
        for obj in self.pool:
            if not obj.in_use:
                obj.in_use = True
                return obj
        raise Exception("No available objects in the pool")

    def release(self, obj):
        obj.in_use = False
    
    def print_info(self):
        for i in range(len(self.pool)):
            print(f"TTS Object {i} is in use: {self.pool[i].in_use}")

       
class PCMQueue:
    def __init__(self):
        """
        Initialize the PCMQueue with an empty queue, an empty buffer, and a lock for thread safety.
        
        Parameters:
        - None
        
        Returns:
        - None
        """
        self.queue = queue.Queue()
        self.buffer = np.array([], dtype=np.float32)
        self.lock = threading.Lock()

    def put(self, items):
        """
        Add items to the buffer in a thread-safe manner.
        
        Parameters:
        - items (list or array-like): The items to be added to the buffer, a numpy array of dtype np.float32.
        
        Returns:
        - None
        """
        with self.lock:
            self.buffer = np.concatenate((self.buffer, np.array(items, dtype=np.float32)))

    def get(self, length):
        """
        Retrieve a specified number of elements from the buffer in a thread-safe manner.
        
        Parameters:
        - length (int): The number of elements to retrieve from the buffer.
        
        Returns:
        - numpy.ndarray or None: A numpy array containing the requested number of elements if available, otherwise None.
        """
        with self.lock:
            if len(self.buffer) < length:
                return None
            result = self.buffer[:length]
            self.buffer = self.buffer[length:]
            return result
    
    def clear(self):
        """
        Clear the buffer in a thread-safe manner.
        
        Parameters:
        - None
        
        Returns:
        - None
        """
        with self.lock:
            self.buffer = np.array([], dtype=np.float32)

    def has_enough_data(self, length):
        """
        Check if the buffer contains enough data to fulfill a request of a specified length.
        
        Parameters:
        - length (int): The number of elements required.
        
        Returns:
        - bool: True if the buffer contains enough data, False otherwise.
        """
        with self.lock:
            return len(self.buffer) >= length


class ThreadSafeQueue:
    def __init__(self):
        """
        Initialize the ThreadSafeQueue with an empty queue and a lock for thread safety.
        
        Parameters:
        - None
        
        Returns:
        - None
        """
        self._queue = queue.Queue()
        self._lock = threading.Lock()

    def put(self, item):
        """
        Add an item to the queue in a thread-safe manner.
        
        Parameters:
        - item (any): The item to be added to the queue.
        
        Returns:
        - None
        """
        with self._lock:
            self._queue.put(item)

    def get(self):
        """
        Retrieve an item from the queue in a thread-safe manner.
        
        Parameters:
        - None
        
        Returns:
        - any or None: The retrieved item if the queue is not empty, otherwise None.
        """
        with self._lock:
            if not self._queue.empty():
                return self._queue.get()
            else:
                return None

    def clear(self):
        """
        Clear the queue in a thread-safe manner.
        
        Parameters:
        - None
        
        Returns:
        - None
        """
        with self._lock:
            while not self._queue.empty():
                self._queue.get()

    def is_empty(self):
        """
        Check if the queue is empty in a thread-safe manner.
        
        Parameters:
        - None
        
        Returns:
        - bool: True if the queue is empty, False otherwise.
        """
        with self._lock:
            return self._queue.empty()

    def size(self):
        """
        Get the current size of the queue in a thread-safe manner.
        
        Parameters:
        - None
        
        Returns:
        - int: The number of items currently in the queue.
        """
        with self._lock:
            return self._queue.qsize()


class BatchQueue:
    def __init__(self, maxsize=0):
        """
        初始化批量操作队列
        
        Args:
            maxsize: 队列最大容量，0表示无限制
        """
        self.queue = queue.Queue(maxsize)
        self.mutex = threading.Lock()
        self.not_empty = threading.Condition(self.mutex)
        self.not_full = threading.Condition(self.mutex)
    
    def put(self, items, block=True, timeout=None):
        """
        批量放入元素
        
        Args:
            items: 可迭代对象，包含要放入的元素
            block: 是否阻塞直到有足够空间
            timeout: 阻塞超时时间
        
        Raises:
            Full: 队列已满且超时
        """
        items = list(items)
        item_count = len(items)
        
        with self.not_full:
            # 检查是否有足够空间
            if self.queue.maxsize > 0:
                if not block:
                    if self.queue.qsize() + item_count > self.queue.maxsize:
                        raise Full
                elif timeout is None:
                    while self.queue.qsize() + item_count > self.queue.maxsize:
                        self.not_full.wait()
                elif timeout < 0:
                    raise ValueError("timeout must be a non-negative number")
                else:
                    endtime = time() + timeout
                    while self.queue.qsize() + item_count > self.queue.maxsize:
                        remaining = endtime - time()
                        if remaining <= 0.0:
                            raise Full
                        self.not_full.wait(remaining)
            
            # 批量放入元素
            for item in items:
                self.queue._put(item)
            
            # 通知等待的get操作
            self.not_empty.notify_all()
    
    def get(self, n=1, block=True, timeout=None):
        """
        批量获取元素
        
        Args:
            n: 要获取的元素数量
            block: 是否阻塞直到有足够元素
            timeout: 阻塞超时时间
        
        Returns:
            list: 获取的元素列表
        
        Raises:
            Empty: 队列元素不足且超时
        """
        with self.not_empty:
            # 检查是否有足够元素
            if not block:
                if self.queue.qsize() < n:
                    raise Empty
            elif timeout is None:
                while self.queue.qsize() < n:
                    self.not_empty.wait()
            elif timeout < 0:
                raise ValueError("timeout must be a non-negative number")
            else:
                endtime = time() + timeout
                while self.queue.qsize() < n:
                    remaining = endtime - time()
                    if remaining <= 0.0:
                        raise Empty
                    self.not_empty.wait(remaining)
            
            # 批量获取元素
            result = [self.queue._get() for _ in range(n)]
            
            # 通知等待的put操作
            self.not_full.notify_all()
            return result
    
    def qsize(self):
        """获取队列当前大小"""
        with self.mutex:
            return self.queue.qsize()
    
    def empty(self):
        """判断队列是否为空"""
        with self.mutex:
            return self.queue.empty()
    
    def full(self):
        """判断队列是否已满"""
        with self.mutex:
            return self.queue.full()


class VAD:
    def __init__(self, cache_history=50):
        self.CHUNK = 1600
        self.cache_history = cache_history
        self.in_dialog = False

        with torch.no_grad():
            self.load_vad()
            self.reset_vad()
            self.load_sil()
    
    def get_chunk_size(self):
        return self.CHUNK

    def add_history(self, chunk):
        self.history[:-1] = self.history[1:].clone()
        self.history[-1:] = chunk
        if self.history_num < self.cache_history:
            self.history_num += 1

    def load_sil(self, sil_wav_path = "./web_flask/input_data/silence.wav"):
        waveform, sample_rate = torchaudio.load(sil_wav_path)
        # 检查音频的维度
        num_channels = waveform.shape[0]
        # 如果音频是多通道的，则进行通道平均
        if num_channels > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=16000)(waveform)
        self.start_sil = waveform.squeeze(0)

    def pad_sil(self, waveform):
        len_sil = random.randint(200, 300)*16
        rand_sil = self.start_sil.numpy()[0:len_sil]
        return np.concatenate((rand_sil, waveform), axis=0)
  
    def load_vad(self):
        self.vad_model = load_silero_vad()
        self.vad_model.eval()
        # if USE_NPU:
        #     self.vad_model = self.vad_model.to("npu")
        # else:
        #     self.vad_model = self.vad_model.to("cuda")
        # generate vad itertator
        self.vad_iterator = VADIterator(self.vad_model, 
                                        threshold=0.8, 
                                        sampling_rate=16000, 
                                        min_silence_duration_ms=300, 
                                        speech_pad_ms=30)
        self.vad_iterator.reset_states()

    def reset_vad(self):
        # reset all parms
        self.input_chunk = torch.zeros([1, self.CHUNK , 1])
        self.history = torch.zeros([self.cache_history, self.CHUNK, 1])
        self.history_num = 0
        self.vad_iterator.reset_states()
        self.in_dialog = False
    
    def run_vad_iterator(self, audio):
        speech_dict_out = None
        # split into chunk with 512
        for i in range(len(audio) // 512):
            speech_dict = self.vad_iterator(audio[i * 512: (i + 1) * 512], return_seconds=True)
            if speech_dict is not None:
                speech_dict_out = speech_dict
        return speech_dict_out
    
    def predict(self,
                audio: torch.Tensor):
        """
        Predict the Voice Activity Detection (VAD) status and return related features.

        Parameters:
        - audio (torch.Tensor): A 1D or 2D tensor representing the input audio signal.

        Returns:
        - return_dict (dict): A dictionary containing the VAD status and related features.
            - 'status' (str): The current VAD status, which can be 'sl' (speech start), 
                              'cl' (speech continue), or 'el' (speech end).
            - 'feature_last_chunk' (list of list of float): The feature of the last chunks.
            - 'feature' (list of list of float): The feature of the current chunk of audio.
            - 'history_feature' (list of list of list of float): The cached features of previous chunks.
        
        """

        return_dict = {}
        return_dict['status'] = None
        with torch.no_grad():
            # get fbank feature
            audio = torch.tensor(audio)
            sample_data = audio.reshape(1, -1, 1)[:, :, :1] * 32768
            self.input_chunk = sample_data.clone()

            # get vad status
            if self.in_dialog:
                speech_dict = self.run_vad_iterator(audio.reshape(-1))
                if speech_dict is not None and "end" in speech_dict:
                    return_dict['status'] = 'el'
                    self.add_history(self.input_chunk)
                    return_dict['feature'] = self.input_chunk.numpy()
                    return_dict['history_feature'] = self.history[-self.history_num:].flatten().numpy()
                    return_dict['history_feature'] = self.pad_sil(return_dict['history_feature'])
                    self.reset_vad()
                    return return_dict
                else:
                    return_dict['status'] = 'cl'
                    self.add_history(self.input_chunk)
                    return_dict['feature'] = self.input_chunk.numpy()
                    return_dict['history_feature'] = self.history[-self.history_num:].flatten().numpy()
                    return return_dict
            else:
                speech_dict = self.run_vad_iterator(audio.reshape(-1))
                if speech_dict is not None and "start" in speech_dict:
                    return_dict['status'] = 'sl'
                    self.in_dialog = True
                    # self.vad_iterator.reset_states()
                    self.add_history(self.input_chunk)
                    return_dict['feature'] = self.input_chunk.numpy()
                    return_dict['history_feature'] = self.history[-self.history_num:].flatten().numpy()
                    return return_dict
            return return_dict


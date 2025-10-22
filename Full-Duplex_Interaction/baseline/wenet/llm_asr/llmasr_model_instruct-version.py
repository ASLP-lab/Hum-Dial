import logging
import os
from typing import Dict, List, Optional, Union
import torchaudio
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

from wenet.transformer.encoder import TransformerEncoder
from wenet.llm_asr.utils4llmasr import *
from gxl_ai_utils.utils import utils_file

from wenet.llm_asr.downsampler import get_downsampler, LyzConv1dSubsampling
from wenet.utils.mask import make_pad_mask
from wenet.utils.mask import make_pad_mask
import torch.nn.functional as F
import math
import gc

from time import time
import threading
from transformers.generation.stopping_criteria import StoppingCriteriaList, StoppingCriteria
from patches.cumstom_stop_criteria import (
    S2SStopCriteria, MaxTokenStopper, InterruptStopper
)
from wenet.llm_asr.streamers import TokenIdStreamer

try:
    import torch_npu
    from torch_npu.contrib import transfer_to_npu
except:
    print("运行在英伟达平台")
from queue import Queue


# from msprobe.pytorch import seed_all,PrecisionDebugger

class LLMASR_Model(nn.Module):
    def __init__(self,
                 encoder,
                 encoder_output_dim,
                 llm_path,
                 lora=True, lora_alpha=32, lora_rank=8, lora_dropout=0.1,
                 prompt_pattern="{}：<Speech><SpeechHere></Speech>",
                 #  "USER: <Speech><SpeechHere></Speech> {}\nASSISTANT:"
                 is_inference=False,
                 downsample_rate=1,
                 llm_embed_dim=4096,
                 task_num=2,
                 adapter_type='lyz',
                 speech_token_num=0,
                 train_speech_out=False):
        """"""
        super().__init__()
        self.s2s_stop_criteria = None
        self.max_token_criteria_list = None
        self.downsample_rate = downsample_rate

        self.encoder = encoder
        self.ln_speech = nn.LayerNorm(encoder_output_dim)

        # 连接层, 51.6M
        if adapter_type == 'gxl':
            self.speech_transformer = TransformerEncoder(
                input_size=encoder_output_dim,
                output_size=encoder_output_dim,
                attention_heads=4,
                linear_units=2560,
                num_blocks=4,
                dropout_rate=0.1,
                positional_dropout_rate=0.1,
                attention_dropout_rate=0.0,
                input_layer="linear",
                pos_enc_layer_type="abs_pos",
                normalize_before=True
            )
        else:
            self.speech_transformer = None

        # LLM,
        self.low_resource = False
        if not self.low_resource:
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                llm_path,
                # torch_dtype=torch.float32 if is_inference else torch.float16,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                output_hidden_states=True,
            )
        else:
            self.llama_model = AutoModelForCausalLM.from_pretrained(
                llm_path,
                torch_dtype=torch.float16,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=True,
                output_hidden_states=True,
            )

        self.max_length = 4000
        self.min_length = 1
        self.num_beams = 1
        self.do_sample = True
        self.top_p = 0.9
        self.top_k = 5
        self.repetition_penalty = 1.05  # 1.05
        self.length_penalty = 1.0
        self.temperature = 1.0  # 1.0
        self.IGNORE_ID = -100

        # lora
        self.lora = lora
        if lora:
            utils_file.logging_limit_print("耿雪龙： 使用lora了")
            # target_modules = ['w_pack', 'o_proj', 'gate_proj', 'down_proj']
            target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'down_proj']
            if is_inference:
                self.peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=True,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=target_modules,
                )
            else:
                self.peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=target_modules,
                )
            self.llama_model = get_peft_model(self.llama_model, self.peft_config)

        # tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_path, use_fast=False, trust_remote_code=True)
        """
        设置分词器的pad_token和padding的方向。
        """
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.tokenizer.padding_side = "right"

        if hasattr(self.llama_model.config, 'hidden_size'):
            utils_file.logging_limit_print(
                f"self.llama_model.config.hidden_size: {self.llama_model.config.hidden_size}")
            if adapter_type == 'lyz':
                self.down_sample_2 = LyzConv1dSubsampling(encoder_output_dim, self.llama_model.config.hidden_size)
            elif adapter_type == 'gxl':
                self.down_sample_2 = get_downsampler(downsample_rate, encoder_output_dim)
                self.speech_llama_proj = nn.Linear(
                    encoder_output_dim, self.llama_model.config.hidden_size)
            # self.task_embeddings = torch.nn.Embedding(task_num, self.llama_model.config.hidden_size)
        else:
            raise NotImplementedError("self.llama_model.config.hidden_size not exist")

        self.embed_tokens = self.llama_model.model.model.embed_tokens if self.lora else self.llama_model.model.embed_tokens
        self.lm_head = self.llama_model.model.lm_head if self.lora else self.llama_model.lm_head

        self.speech_token_num = speech_token_num
        # init speech token module
        if speech_token_num > 0:
            utils_file.logging_info(f'耿雪龙： 进行语音token生成任务， speech_token_num: {speech_token_num}')
            self.speech_token_emded = torch.nn.Embedding(speech_token_num + 2, self.llama_model.config.hidden_size)
            self.speech_head = torch.nn.Linear(self.llama_model.config.hidden_size, speech_token_num)
        else:

            # 不做任何处理
            self.speech_head = nn.Identity()
            self.speech_token_emded = nn.Identity()
        self.train_speech_out = train_speech_out
        utils_file.logging_info(f'耿雪龙： 是否进行语音输出训练：{self.train_speech_out}')
        self.loss_fct = CrossEntropyLoss(reduction='mean')
        self.new_context = True
        self.multi_turn_cache = None
        self.multi_turn_return_dict = {
            "output_attentions": False,
            "output_hidden_states": False,
            "output_scores": False,
            "output_logits": False,
        }
        self.add_embed_head = True
        self.init_custom_stop_criteria()

    def get_label_embedding(self, labels, labels_lengths):
        """"""
        labels_pad_mask = make_pad_mask(labels_lengths)  # B, L
        labels = labels.masked_fill(labels_pad_mask, 0)
        labels_embeds = self.embed_tokens(labels)
        labels_target = labels.masked_fill(labels_pad_mask, self.IGNORE_ID)  # B, L
        labels_mask = ~labels_pad_mask
        return labels_embeds, labels_target, labels_mask

    def get_speech_token_label_embedding(self, speech_token_labels, speech_tokens_length):
        """"""
        speech_tokens_pad_mask = make_pad_mask(speech_tokens_length)  # B, L
        speech_token_labels = speech_token_labels.masked_fill(speech_tokens_pad_mask, 0)
        speech_token_labels_embeds = self.speech_token_emded(speech_token_labels)
        # utils_file.logging_limit_print(f'进行speech_token_labels修改，修改前 speech_token_labels',
        #    speech_token_labels.shape, speech_token_labels[0][-1], speech_token_labels[0][0])
        speech_token_labels = speech_token_labels + 152064
        # utils_file.logging_limit_print(f'进行speech_token_labels修改，修改后 speech_token_labels',
        #    speech_token_labels.shape, speech_token_labels[0][-1], speech_token_labels[0][0])
        speech_token_labels_target = speech_token_labels.masked_fill(speech_tokens_pad_mask, self.IGNORE_ID)  # B, L
        speech_token_labels_mask = ~speech_tokens_pad_mask
        return speech_token_labels_embeds, speech_token_labels_target, speech_token_labels_mask

    def forward(self,
                batch,
                device,
                ):
        """"""
        rank = int(os.environ.get('RANK', 0))
        output_type = batch['output_type']
        utils_file.logging_limit_print(f'xxx output_type {output_type}')
        assert output_type in ['text', 'speech2text_token', 'text2token'], f"output_type:{output_type} not support"
        # speech inputs
        if output_type == 'text' or output_type == 'speech2text_token':
            wavs = batch['feats'].to(device)
            utils_file.logging_limit_print(f'xxx wav shape {wavs.shape}')
            wavs_len = batch['feats_lengths'].to(device)
            B = wavs.shape[0]
            utils_file.logging_limit_print(f"xxx {wavs_len}")
            speech_embeds, speech_masks = self._get_embedding_from_wav(wavs, wavs_len)
            utils_file.logging_limit_print(f'xxx speech embeding shape {speech_embeds.shape}')
            utils_file.logging_limit_print(f'xxx speech mask shape {speech_masks.shape}')
            utils_file.logging_limit_print(f'xxx speech mask 0 {speech_masks[0]}')
            speech_target = torch.full(speech_masks.shape, self.IGNORE_ID).to(
                speech_embeds.device)
            utils_file.logging_limit_print(f'xxx speech target shape {speech_target.shape}')
            utils_file.logging_limit_print(f'xxx speech target 0 {speech_target[0]}')
        else:
            labels = batch['target'].to(device)
            labels_lengths = batch['target_lengths'].to(device)
            B = labels.shape[0]
            #  text 2 token ,拿到文本序列
            labels_pad_mask = make_pad_mask(labels_lengths)  # B, L
            labels = labels.masked_fill(labels_pad_mask, 0)
            speech_embeds = self.embed_tokens(labels)  # B, L, D
            speech_target = torch.full(labels_pad_mask.shape, self.IGNORE_ID).to(
                speech_embeds.device)
            speech_masks = ~labels_pad_mask

        # add bos and eos
        speech_embeds, speech_masks, speech_target = self._add_bos_eos(0 + self.speech_token_num,
                                                                       1 + self.speech_token_num,
                                                                       speech_embeds, speech_masks, speech_target)
        utils_file.logging_limit_print(f'xxx after add bos eos speech embeding shape {speech_embeds.shape}')
        utils_file.logging_limit_print(f'xxx after add bos eos speech mask shape {speech_masks.shape}')
        utils_file.logging_limit_print(f'xxx after add bos eos speech target shape {speech_target.shape}')
        utils_file.logging_limit_print(f'xxx after add bos eos speech mask 0 {speech_masks[0]}')
        utils_file.logging_limit_print(f'xxx after add bos eos speech target 0 {speech_target[0]}')

        # prompt
        if 'prompt' in batch:
            prompt = batch['prompt'].to(device)
            prompt_lengths = batch['prompt_lengths'].to(device)
            prompt_pad_mask = make_pad_mask(prompt_lengths)  # B, L
            prompt = prompt.masked_fill(prompt_pad_mask, self.tokenizer.eos_token_id)
            prompt_embeds = self.embed_tokens(prompt)  # B, L, D
            prompt_target = torch.full(prompt.shape, self.IGNORE_ID).to(
                speech_embeds.device)  # B, L
            prompt_mask = ~prompt_pad_mask
            utils_file.logging_limit_print(f'xxx prompt embeding shape {prompt_embeds.shape}')
            utils_file.logging_limit_print(f'xxx prompt mask shape {prompt_mask.shape}')
            utils_file.logging_limit_print(f'xxx prompt target shape {prompt_target.shape}')
        else:
            prompt_embeds = None
            prompt_mask = None
            prompt_target = None

        inputs_embeds_list = []
        attention_mask_list = []
        target_list = []
        qwen_instruct_prompt_pattern_1 = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        prompt_pattern1 = self.tokenizer([qwen_instruct_prompt_pattern_1] * len(batch['target']), return_tensors="pt"
                                         )['input_ids'].to(device)
        prompt_pattern1_embeds = self.embed_tokens(prompt_pattern1)
        prompt_pattern1_lens = torch.tensor([len(i) for i in prompt_pattern1]).to(device)
        prompt_pattern1_mask = ~make_pad_mask(prompt_pattern1_lens)
        prompt_pattern1_target = torch.full(prompt_pattern1.shape, self.IGNORE_ID).to(
            device)  # B, L

        qwen_instruct_prompt_pattern_2 = "<|im_end|>\n<|im_start|>assistant\n"
        prompt_pattern2 = self.tokenizer([qwen_instruct_prompt_pattern_2] * len(batch['target']), return_tensors="pt"
                                         )['input_ids'].to(device)
        prompt_pattern2_embeds = self.embed_tokens(prompt_pattern2)
        prompt_pattern2_lens = torch.tensor([len(i) for i in prompt_pattern2]).to(device)
        prompt_pattern2_mask = ~make_pad_mask(prompt_pattern2_lens)
        prompt_pattern2_target = torch.full(prompt_pattern2.shape, self.IGNORE_ID).to(
            device)  # B, L

        inputs_embeds_list.append(prompt_pattern1_embeds)
        attention_mask_list.append(prompt_pattern1_mask)
        target_list.append(prompt_pattern1_target)
        if output_type == 'speech2text_token':
            utils_file.logging_limit_print(f'xxx 开始处理speech2text_token任务')
            labels = batch['target'].to(device)
            labels_lengths = batch['target_lengths'].to(device)
            speech_token_labels = batch['speech_tokens'].to(device)
            speech_tokens_length = batch['speech_tokens_length'].to(device)

            labels_embeds, labels_target, labels_mask = self.get_label_embedding(labels, labels_lengths)

            speech_token_labels_embeds, speech_token_labels_target, speech_token_labels_mask = self.get_speech_token_label_embedding(
                speech_token_labels, speech_tokens_length)

            if prompt_embeds is not None:
                inputs_embeds_list.append(prompt_embeds)
                attention_mask_list.append(prompt_mask)
                target_list.append(prompt_target)

            inputs_embeds_list.extend(
                [speech_embeds, prompt_pattern2_embeds, labels_embeds, speech_token_labels_embeds])
            attention_mask_list.extend([speech_masks, prompt_pattern2_mask, labels_mask, speech_token_labels_mask])
            target_list.extend([speech_target, prompt_pattern2_target, labels_target, speech_token_labels_target])
        elif output_type == "text2token":
            speech_token_labels = batch['speech_tokens'].to(device)
            speech_tokens_length = batch['speech_tokens_length'].to(device)
            speech_token_labels_embeds, speech_token_labels_target, speech_token_labels_mask = self.get_speech_token_label_embedding(
                speech_token_labels, speech_tokens_length)
            if prompt_embeds is not None:
                inputs_embeds_list.append(prompt_embeds)
                attention_mask_list.append(prompt_mask)
                target_list.append(prompt_target)
            inputs_embeds_list.extend([speech_embeds, prompt_pattern2_embeds, speech_token_labels_embeds])
            attention_mask_list.extend([speech_masks, prompt_pattern2_mask, speech_token_labels_mask])
            target_list.extend([speech_target, prompt_pattern2_target, speech_token_labels_target])
        elif output_type == "text":
            labels = batch['target'].to(device)
            labels_lengths = batch['target_lengths'].to(device)
            labels_embeds, labels_target, labels_mask = self.get_label_embedding(labels, labels_lengths)
            if prompt_embeds is not None:
                inputs_embeds_list.append(prompt_embeds)
                attention_mask_list.append(prompt_mask)
                target_list.append(prompt_target)
            else:
                utils_file.logging_limit_print(
                    f'prompt is None,task: {batch["task"]}, prompt_embeds:{prompt_embeds}, prompt_mask:{prompt_mask}')
            inputs_embeds_list.extend([speech_embeds, prompt_pattern2_embeds, labels_embeds])
            attention_mask_list.extend([speech_masks, prompt_pattern2_mask, labels_mask])
            target_list.extend([speech_target, prompt_pattern2_target, labels_target])
        else:
            raise NotImplementedError(f'output_type {output_type} not support')

        inputs_embeds = torch.cat(inputs_embeds_list, dim=1)
        utils_file.logging_limit_print(f'xxx final inputs_embeds shape {inputs_embeds.shape}')
        attention_mask = torch.cat(attention_mask_list, dim=1)
        utils_file.logging_limit_print(f'xxx final attention_mask shape {attention_mask.shape}')
        utils_file.logging_limit_print(f'xxx final attention_mask 0 {attention_mask[0]}')
        target = torch.cat(target_list, dim=1)
        utils_file.logging_limit_print(f'xxx final  target shape {target.shape}')
        utils_file.logging_limit_print(f'xxx final target 0 {target[0]}')
        # utils_file.logging_limit_print(f'耿雪龙 output_type: {output_type}')
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        utils_file.logging_limit_print(f'xxx final position_ids shape {position_ids.shape}')
        utils_file.logging_limit_print(f'xxx final position_ids 0 {position_ids[0]}')
        if output_type == 'text':
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                labels=target,
                attention_mask=attention_mask,
                position_ids=position_ids.to(inputs_embeds.device)
            )
            loss = outputs['loss']
            return {"loss": loss}
        else:
            utils_file.logging_limit_print(f'进行llama_model的 diy forward')
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                # labels=target,
                attention_mask=attention_mask,
                position_ids=position_ids.to(inputs_embeds.device)
            )
            hidden_states = outputs['hidden_states'][-1]
            logits = self.lm_head(hidden_states)
            logits2 = self.speech_head(hidden_states)  # speech_head
            combined_logits = torch.cat([logits, logits2], dim=-1)
            shift_logits = combined_logits[..., :-1, :].contiguous()
            shift_target = target[..., 1:].contiguous()
            utils_file.logging_limit_print(
                f'xxx shift_logits shape: {shift_logits.shape}, shift_target shape: {shift_target.shape}')
            utils_file.logging_limit_print(f'xxx shift_target 0 {shift_target[0]}')
            shift_logits = shift_logits.view(-1, combined_logits.shape[-1])  # 注意这里维度的调整，根据logits2的维度相应改变
            shift_target = shift_target.view(-1)
            shift_target = shift_target.to(shift_logits.device)
            loss = self.loss_fct(shift_logits, shift_target)
            loss.requires_grad_(True)
            return {"loss": loss}

    def do_add_speech_embed_head(self):
        if self.add_embed_head:
            self.llama_model.speech_token_emded = self.speech_token_emded.to(torch.bfloat16)
            self.llama_model.speech_head = self.speech_head.to(torch.bfloat16)
            # self.llama_model.speech_token_emded = self.speech_token_emded.to(torch.bfloat16)
            # self.llama_model.speech_head = self.speech_head.to(torch.bfloat16) # 带lora的时候用
            self.add_embed_head = False

    def init_custom_stop_criteria(self):
        """
        创建需要的stop criteria
        1. 对于t2t任务，遇到text_eos停止
        2. 对于t2s任务，遇到speech_eos停止
        3. 对于s2s任务，遇到speech_eos停止
        同时要取消原本的停止条件
        if generation_config._eos_token_tensor is not None:
        取消 generation_config._eos_token_tensor 的停止，尝试直接给一个大于vocb_size的eos_token
        """
        self.interrupt = InterruptStopper()
        self.s2s_stop_criteria = StoppingCriteriaList()
        self.s2s_stop_criteria.append(S2SStopCriteria(text_eos_id=151645, speech_eos_id=self.speech_token_num - 1))
        self.s2s_stop_criteria.append(MaxTokenStopper(2000))
        self.s2s_stop_criteria.append(self.interrupt)
        # self.max_token_criteria_list = StoppingCriteriaList([MaxTokenStopper(300)])

    def set_task_type(self, task_type: str):
        """设置任务类型，用于设置生成的初始类型
        Args:
            task_type (str): 任务类型，从("ASR", "TTS", "S2S")选择
        """
        assert task_type in ("ASR", "TTS", "S2S")
        if task_type == "ASR":
            self.llama_model.text_phase = True
        elif task_type == "TTS":
            self.llama_model.text_phase = False
        elif task_type == "S2S":
            self.llama_model.text_phase = True

    def generate(
            self,
            wavs,
            wavs_len,
            prompt,
            **kwargs
    ):
        self.llama_model.eval()
        self.set_task_type("ASR")
        self.do_add_speech_embed_head()
        # self.do_merge_embed_head()
        speech_embeds, speech_masks = self._get_embedding_from_wav(wavs, wavs_len)
        speech_embeds, speech_masks, _ = self._add_bos_eos(0 + self.speech_token_num, 1 + self.speech_token_num,
                                                           speech_embeds, speech_masks, None)
        prompt = self.tokenizer([prompt], return_tensors="pt"
                                )['input_ids'].to(speech_embeds.device)
        prompt_embeds = self.embed_tokens(prompt)

        qwen_instruct_prompt_pattern_1 = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        prompt_pattern1 = self.tokenizer([qwen_instruct_prompt_pattern_1] * len(wavs_len), return_tensors="pt"
                                         )['input_ids'].to(speech_embeds.device)
        prompt_pattern1_embeds = self.embed_tokens(prompt_pattern1)

        qwen_instruct_prompt_pattern_2 = "<|im_end|>\n<|im_start|>assistant\n"
        prompt_pattern2 = self.tokenizer([qwen_instruct_prompt_pattern_2] * len(wavs_len), return_tensors="pt"
                                         )['input_ids'].to(speech_embeds.device)
        prompt_pattern2_embeds = self.embed_tokens(prompt_pattern2)

        embeds = torch.cat([prompt_pattern1_embeds, prompt_embeds, speech_embeds, prompt_pattern2_embeds], dim=1)
        atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(embeds.device)

        if self.embed_tokens.weight.dtype == torch.float16 or self.embed_tokens.weight.dtype == torch.bfloat16:
            # utils_file.logging_limit_print('generate(): self.embed_tokens.weight.dtype == torch.float16')
            # embeds = embeds.to(torch.float16)
            embeds = embeds.to(torch.bfloat16)
            atts = atts.to(torch.bfloat16)
        outputs = self.llama_model.generate(
            inputs_embeds=embeds,
            max_new_tokens=self.max_length,
            cache_implementation="static",
            # num_beams=self.num_beams,
            do_sample=self.do_sample,
            min_length=self.min_length,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=self.repetition_penalty,
            length_penalty=self.length_penalty,
            temperature=self.temperature,
            # attention_mask=atts,
            eos_token_id=151645,
            pad_token_id=-100,
            stopping_criteria=self.max_token_criteria_list,
            do_compile=True,
        )
        output_text = self.tokenizer.batch_decode(outputs, add_special_tokens=False, skip_special_tokens=True)

        return output_text

    def generate_s2s(
            self,
            wavs,
            wavs_len,
            prompt,
    ):
        self.llama_model.eval()
        self.set_task_type("S2S")
        self.do_add_speech_embed_head()
        # =====================准备input embedding=====================
        speech_embeds, speech_masks = self._get_embedding_from_wav(wavs, wavs_len)
        speech_embeds, speech_masks, _ = self._add_bos_eos(0 + self.speech_token_num, None,
                                                           speech_embeds, speech_masks, None)

        device = speech_embeds.device

        qwen_instruct_prompt_pattern_1 = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        prompt_pattern1 = self.tokenizer([qwen_instruct_prompt_pattern_1] * len(wavs_len), return_tensors="pt"
                                         )['input_ids'].to(speech_embeds.device)
        prompt_pattern1_embeds = self.embed_tokens(prompt_pattern1)

        qwen_instruct_prompt_pattern_2 = "<|im_end|>\n<|im_start|>assistant\n"
        prompt_pattern2 = self.tokenizer([qwen_instruct_prompt_pattern_2] * len(wavs_len), return_tensors="pt"
                                         )['input_ids'].to(speech_embeds.device)
        prompt_pattern2_embeds = self.embed_tokens(prompt_pattern2)

        prompt = self.tokenizer([prompt], return_tensors="pt"
                                )['input_ids'].to(speech_embeds.device)
        prompt_embeds = self.embed_tokens(prompt)

        hyps = [4098]
        token_emb = self.speech_token_emded(torch.tensor(hyps[-1:]).to(device)).unsqueeze(0)

        embeds = torch.cat(
            [prompt_pattern1_embeds, prompt_embeds.to(device), speech_embeds, token_emb, prompt_pattern2_embeds], dim=1)
        atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(embeds.device)
        if self.embed_tokens.weight.dtype == torch.float16:
            # utils_file.logging_limit_print('generate(): self.embed_tokens.weight.dtype == torch.float16')
            embeds = embeds.to(torch.float16)

        # max_len = 350
        top_k = 10
        top_p = 0.9
        temperature = 1.2
        invalid_eos = 10000000

        streamer = TokenIdStreamer()
        self.llama_model.generate(
            inputs_embeds=embeds,
            max_new_tokens=self.max_length,
            eos_token_id=invalid_eos,  # 一个不会出现的token，完全靠stopping_criteria
            cache_implementation="static",
            do_sample=True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            stopping_criteria=self.s2s_stop_criteria,
            do_compile=True,
            repetition_penalty=1.0,
            streamer=streamer,
        )

        buffer_tokens = []
        first_yield = True
        first_speech_eos = True
        text_eos_id = 151645
        speech_eos_id = self.speech_token_num - 1

        for token_id in streamer:
            buffer_tokens.append(token_id)
            if first_yield:
                if text_eos_id in buffer_tokens:
                    idx = buffer_tokens.index(text_eos_id)
                    output_text = self.tokenizer.batch_decode(buffer_tokens[:idx], add_special_tokens=False,
                                                              skip_special_tokens=True)
                    output_text = "".join(output_text)
                    print(output_text)
                    # yield buffer_tokens[:idx+1]
                    yield output_text
                    buffer_tokens = buffer_tokens[idx + 1:]
                    first_yield = False
            else:
                while len(buffer_tokens) >= 18 or (speech_eos_id in buffer_tokens):
                    if speech_eos_id in buffer_tokens:
                        idx = buffer_tokens.index(speech_eos_id)
                        if first_speech_eos:
                            # 第一次遇到speech_eos_id，直接跳过，不yield
                            buffer_tokens = buffer_tokens[idx + 1:]
                            first_speech_eos = False
                            break
                        else:
                            yield buffer_tokens[:idx]
                            buffer_tokens = []
                            return
                    else:
                        yield buffer_tokens[:18]
                        buffer_tokens = buffer_tokens[18:]
        # 处理剩余内容
        if buffer_tokens:
            yield buffer_tokens

    def generate_s2s_no_stream(
            self,
            wavs,
            wavs_len,
            prompt,
    ):
        self.llama_model.eval()
        self.set_task_type("S2S")
        self.do_add_speech_embed_head()
        # =====================准备input embedding=====================
        speech_embeds, speech_masks = self._get_embedding_from_wav(wavs, wavs_len)
        speech_embeds, speech_masks, _ = self._add_bos_eos(0 + self.speech_token_num, None,
                                                           speech_embeds, speech_masks, None)

        device = speech_embeds.device

        qwen_instruct_prompt_pattern_1 = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        prompt_pattern1 = self.tokenizer([qwen_instruct_prompt_pattern_1] * len(wavs_len), return_tensors="pt"
                                         )['input_ids'].to(speech_embeds.device)
        prompt_pattern1_embeds = self.embed_tokens(prompt_pattern1)

        qwen_instruct_prompt_pattern_2 = "<|im_end|>\n<|im_start|>assistant\n"
        prompt_pattern2 = self.tokenizer([qwen_instruct_prompt_pattern_2] * len(wavs_len), return_tensors="pt"
                                         )['input_ids'].to(speech_embeds.device)
        prompt_pattern2_embeds = self.embed_tokens(prompt_pattern2)

        prompt = self.tokenizer([prompt], return_tensors="pt"
                                )['input_ids'].to(speech_embeds.device)
        prompt_embeds = self.embed_tokens(prompt)

        hyps = [4098]
        token_emb = self.speech_token_emded(torch.tensor(hyps[-1:]).to(device)).unsqueeze(0)

        embeds = torch.cat(
            [prompt_pattern1_embeds, prompt_embeds.to(device), speech_embeds, token_emb, prompt_pattern2_embeds],
            dim=1)
        atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(embeds.device)
        if self.embed_tokens.weight.dtype == torch.float16:
            # utils_file.logging_limit_print('generate(): self.embed_tokens.weight.dtype == torch.float16')
            embeds = embeds.to(torch.float16)

        # max_len = 350
        top_k = 10
        top_p = 0.9
        temperature = 1.2
        invalid_eos = 10000000

        llm_out = self.llama_model.generate(inputs_embeds=embeds,
                                            max_new_tokens=self.max_length,
                                            eos_token_id=invalid_eos,
                                            cache_implementation="static",
                                            do_sample=True,
                                            temperature=temperature,
                                            top_k=top_k,
                                            top_p=top_p,
                                            stopping_criteria=self.s2s_stop_criteria,
                                            do_compile=True,
                                            repetition_penalty=1.0,
                                            )

        text_eos_idx = (llm_out[0] == 151645).nonzero(as_tuple=True)[0][0].item()
        text_res = llm_out[:, :text_eos_idx - 1]
        speech_res = llm_out[:, text_eos_idx + 1:-1]

        # print("llm_out", llm_out)
        output_text = self.tokenizer.batch_decode(text_res, add_special_tokens=False, skip_special_tokens=True)
        print(f'output_text:{output_text}')
        print(f'speech_res:{speech_res}')
        return (output_text, text_res, speech_res)

    def reset_multi_turn_cache(self, init_cache_len=0):
        if self.multi_turn_cache is not None:
            self.multi_turn_cache.reset()
        self.last_cache_len = init_cache_len

    def update_multi_turn_cache(self,
                                text_eos_idx,
                                input_speech_embed_len,
                                prompt_pattern2_embed_len,
                                past_key_values=None, ):
        """手动更新多轮对话的cache
        Args:
            input_speech_embed_len (tuple, optional): _description_. Defaults to None.
            prompt_pattern2_embed_len (tuple, optional): _description_. Defaults to None.
            past_key_values ([Tuple[torch.FloatTensor]], optional): KV Cache. num_layer, [BNSD].

        last_cache_len [history cache] | 1 [speech bos] | input_speech_embed_len [speech_embed] | 1 [speech eos] \
        | prompt_pattern2_embed_len [prompt_pattern2] | text_eos_idx [generation text] (include text eos) | any [generation audio token]

        """
        logging.info("updating multi turn kv cache")
        if past_key_values is not None:
            self.multi_turn_cache = past_key_values
        # print(input_speech_embed_len, prompt_pattern2_embed_len)

        input_speech_begin = self.last_cache_len + 1
        input_speech_end = input_speech_begin + input_speech_embed_len
        response_text_begin = input_speech_end + 1 + prompt_pattern2_embed_len
        response_text_end = response_text_begin + text_eos_idx + 1

        incre = input_speech_embed_len + text_eos_idx + 1
        new_cache_len = self.last_cache_len + incre
        for layer in self.multi_turn_cache.key_cache:
            save = torch.cat([
                layer[:, :, input_speech_begin:input_speech_end, :],
                layer[:, :, response_text_begin:response_text_end, :],
            ], dim=2)
            layer[:, :, self.last_cache_len:new_cache_len, :].copy_(save)
            layer[:, :, new_cache_len:, :].zero_()

        for layer in self.multi_turn_cache.value_cache:
            save = torch.cat([
                layer[:, :, input_speech_begin:input_speech_end, :],
                layer[:, :, response_text_begin:response_text_end, :],
            ], dim=2)
            layer[:, :, self.last_cache_len:new_cache_len, :].copy_(save)
            layer[:, :, new_cache_len:, :].zero_()

        print("===========update kv cache==============")
        print(f"last_cache_len:{self.last_cache_len} \
                input_speech_begin:{input_speech_begin}, \
                input_speech_end:{input_speech_end}, \
                response_text_begin:{response_text_begin}, \
                response_text_end:{response_text_end}, \
                new_cache_len:{new_cache_len}, \
                  ")

        self.last_cache_len = new_cache_len

        # # 采用以下操作可以debug
        # # 只保留最开始prompt，每次相当于都是重新开始context
        # # 固定种子的情况下，输入相同的数据，两次生成结果应该一摸一样
        # for layer in self.multi_turn_cache.key_cache:
        #     layer[:, :, self.last_cache_len:, :].zero_()

        # for layer in self.multi_turn_cache.value_cache:
        #     layer[:, :, self.last_cache_len:, :].zero_()

    def generate_s2s_no_stream_multi_turn(
            self,
            wavs,
            wavs_len,
            prompt,
    ):
        self.llama_model.eval()
        self.set_task_type("S2S")
        self.do_add_speech_embed_head()
        # =======================================每轮都需要的操作======================================
        # ===========准备input embedding：[speech_embeds, token_emb, prompt_pattern2_embeds]=========
        speech_embeds, speech_masks = self._get_embedding_from_wav(wavs, wavs_len)
        speech_embeds, speech_masks, _ = self._add_bos_eos(0 + self.speech_token_num, None,
                                                           speech_embeds, speech_masks, None)
        device = speech_embeds.device
        hyps = [4098]
        token_emb = self.speech_token_emded(torch.tensor(hyps[-1:]).to(device)).unsqueeze(0)
        qwen_instruct_prompt_pattern_2 = "<|im_end|>\n<|im_start|>assistant\n"
        prompt_pattern2 = self.tokenizer([qwen_instruct_prompt_pattern_2] * len(wavs_len), return_tensors="pt"
                                         )['input_ids'].to(device)
        prompt_pattern2_embeds = self.embed_tokens(prompt_pattern2)

        embed_pieces = [speech_embeds, token_emb, prompt_pattern2_embeds]

        # =====================新context时操作=====================
        if self.new_context:
            # =====================1. 准备prompt======================
            qwen_instruct_prompt_pattern_1 = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
            prompt_pattern1 = self.tokenizer([qwen_instruct_prompt_pattern_1] * len(wavs_len), return_tensors="pt"
                                             )['input_ids'].to(speech_embeds.device)
            prompt_pattern1_embeds = self.embed_tokens(prompt_pattern1)

            prompt = self.tokenizer([prompt], return_tensors="pt"
                                    )['input_ids'].to(speech_embeds.device)
            prompt_embeds = self.embed_tokens(prompt)

            embed_pieces = [prompt_pattern1_embeds, prompt_embeds.to(device), *embed_pieces]
            # =====================2. reset cache======================
            self.reset_multi_turn_cache(init_cache_len=prompt_pattern1_embeds.size(-2) + prompt_embeds.size(-2))

        embeds = torch.cat(embed_pieces, dim=1)
        atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(embeds.device)
        if self.embed_tokens.weight.dtype == torch.float16:
            # utils_file.logging_limit_print('generate(): self.embed_tokens.weight.dtype == torch.float16')
            embeds = embeds.to(torch.float16)

        # max_len = 350
        top_k = 10
        top_p = 0.9
        temperature = 1.2
        invalid_eos = 10000000

        llm_out_dict = self.llama_model.generate(inputs_embeds=embeds,
                                                 max_new_tokens=self.max_length,
                                                 eos_token_id=invalid_eos,
                                                 do_sample=True,
                                                 temperature=temperature,
                                                 top_k=top_k,
                                                 top_p=top_p,
                                                 stopping_criteria=self.s2s_stop_criteria,
                                                 do_compile=True,
                                                 repetition_penalty=1.0,
                                                 # cache操作
                                                 cache_implementation="static" if self.new_context else None,
                                                 # 只有第一轮让系统自己生成Cache
                                                 use_cache=True,
                                                 past_key_values=None if self.new_context else self.multi_turn_cache,
                                                 # 后续轮自己输入修改后的Cache
                                                 return_dict_in_generate=True,
                                                 **self.multi_turn_return_dict  # 控制模型只输出Cache
                                                 )
        llm_out, past_key_values = llm_out_dict.sequences, llm_out_dict.past_key_values

        text_eos_idx = (llm_out[0] == 151645).nonzero(as_tuple=True)[0][0].item()
        text_res = llm_out[:, :text_eos_idx - 1]
        speech_res = llm_out[:, text_eos_idx + 1:-1]

        self.update_multi_turn_cache(
            text_eos_idx=text_eos_idx,
            input_speech_embed_len=speech_embeds.size(1) - 1,
            prompt_pattern2_embed_len=prompt_pattern2_embeds.size(1),
            past_key_values=past_key_values if self.new_context else None,
        )
        self.new_context = False

        # print("llm_out", llm_out)
        output_text = self.tokenizer.batch_decode(text_res, add_special_tokens=False, skip_special_tokens=True)
        print(f'output_text:{output_text}')
        print(f'speech_res:{speech_res}')
        return (output_text, text_res, speech_res)

    def generate_s2s_multi_turn(
            self,
            wavs,
            wavs_len,
            prompt,
    ):
        self.llama_model.eval()
        self.set_task_type("S2S")
        self.do_add_speech_embed_head()
        # =======================================每轮都需要的操作======================================
        # ===========准备input embedding：[speech_embeds, token_emb, prompt_pattern2_embeds]=========
        speech_embeds, speech_masks = self._get_embedding_from_wav(wavs, wavs_len)
        speech_embeds, speech_masks, _ = self._add_bos_eos(0 + self.speech_token_num, None,
                                                           speech_embeds, speech_masks, None)
        device = speech_embeds.device
        hyps = [4098]
        token_emb = self.speech_token_emded(torch.tensor(hyps[-1:]).to(device)).unsqueeze(0)
        qwen_instruct_prompt_pattern_2 = "<|im_end|>\n<|im_start|>assistant\n"
        prompt_pattern2 = self.tokenizer([qwen_instruct_prompt_pattern_2] * len(wavs_len), return_tensors="pt"
                                         )['input_ids'].to(device)
        prompt_pattern2_embeds = self.embed_tokens(prompt_pattern2)

        embed_pieces = [speech_embeds, token_emb, prompt_pattern2_embeds]

        # =====================新context时操作=====================
        if self.new_context:
            # =====================1. 准备prompt======================
            qwen_instruct_prompt_pattern_1 = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
            prompt_pattern1 = self.tokenizer([qwen_instruct_prompt_pattern_1] * len(wavs_len), return_tensors="pt"
                                             )['input_ids'].to(speech_embeds.device)
            prompt_pattern1_embeds = self.embed_tokens(prompt_pattern1)

            prompt = self.tokenizer([prompt], return_tensors="pt"
                                    )['input_ids'].to(speech_embeds.device)
            prompt_embeds = self.embed_tokens(prompt)

            embed_pieces = [prompt_pattern1_embeds, prompt_embeds.to(device), *embed_pieces]
            # =====================2. reset cache======================
            self.reset_multi_turn_cache(init_cache_len=prompt_pattern1_embeds.size(-2) + prompt_embeds.size(-2))

        embeds = torch.cat(embed_pieces, dim=1)
        atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(embeds.device)
        if self.embed_tokens.weight.dtype == torch.float16:
            # utils_file.logging_limit_print('generate(): self.embed_tokens.weight.dtype == torch.float16')
            embeds = embeds.to(torch.float16)

        # max_len = 350
        top_k = 10
        top_p = 0.9
        temperature = 1.2
        invalid_eos = 10000000
        streamer = TokenIdStreamer()

        q = Queue()

        def inner_generate(result_queue):
            llm_out_dict = self.llama_model.generate(
                inputs_embeds=embeds,
                max_new_tokens=self.max_length,
                eos_token_id=invalid_eos,
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                stopping_criteria=self.s2s_stop_criteria,
                do_compile=True,
                repetition_penalty=1.0,
                streamer=streamer,
                # cache操作
                cache_implementation="static" if self.new_context else None,  # 只有第一轮让系统自己生成Cache
                use_cache=True,
                past_key_values=None if self.new_context else self.multi_turn_cache,  # 后续轮自己输入修改后的Cache
                return_dict_in_generate=True,
                **self.multi_turn_return_dict  # 控制模型只输出Cache
            )
            result_queue.put(llm_out_dict)

        thread = threading.Thread(
            target=inner_generate,
            args=(q,)
        )

        thread.start()
        buffer_tokens = []
        first_yield = True
        first_speech_eos = True
        text_eos_id = 151645
        speech_eos_id = self.speech_token_num - 1

        for token_id in streamer:
            buffer_tokens.append(token_id)
            if first_yield:
                if text_eos_id in buffer_tokens:
                    idx = buffer_tokens.index(text_eos_id)
                    output_text = self.tokenizer.batch_decode(buffer_tokens[:idx], add_special_tokens=False,
                                                              skip_special_tokens=True)
                    output_text = "".join(output_text)
                    print(output_text)
                    # yield buffer_tokens[:idx+1]
                    yield output_text
                    buffer_tokens = buffer_tokens[idx + 1:]
                    first_yield = False
            else:
                while len(buffer_tokens) >= 18 or (speech_eos_id in buffer_tokens):
                    if speech_eos_id in buffer_tokens:
                        idx = buffer_tokens.index(speech_eos_id)
                        if first_speech_eos:
                            # 第一次遇到speech_eos_id，直接跳过，不yield
                            buffer_tokens = buffer_tokens[idx + 1:]
                            first_speech_eos = False
                            break
                        else:
                            yield buffer_tokens[:idx]
                            # return
                            buffer_tokens = []
                            break
                    else:
                        yield buffer_tokens[:18]
                        buffer_tokens = buffer_tokens[18:]
        # 处理剩余内容
        if buffer_tokens:
            yield buffer_tokens
        thread.join()

        llm_out_dict = q.get()
        llm_out, past_key_values = llm_out_dict.sequences, llm_out_dict.past_key_values

        text_eos_idx = (llm_out[0] == 151645).nonzero(as_tuple=True)[0][0].item()
        text_res = llm_out[:, :text_eos_idx - 1]
        speech_res = llm_out[:, text_eos_idx + 1:-1]

        self.update_multi_turn_cache(
            text_eos_idx=text_eos_idx,
            input_speech_embed_len=speech_embeds.size(1) - 1,
            prompt_pattern2_embed_len=prompt_pattern2_embeds.size(1),
            past_key_values=past_key_values if self.new_context else None,
        )
        self.new_context = False

        # print("llm_out", llm_out)
        output_text = self.tokenizer.batch_decode(text_res, add_special_tokens=False, skip_special_tokens=True)
        print(f'output_text:{output_text}')
        print(f'speech_res:{speech_res}')
        return

    def generate_text2text(
            self,
            device,
            text,
    ):
        self.llama_model.eval()
        # labels_lengths = torch.tensor([len(text[0])], dtype=torch.int64, device=device)
        # labels = text[:,:]
        labels = self.tokenizer(
            [text],
            return_tensors="pt",
            add_special_tokens=False
        ).to(
            self.embed_tokens.weight.device).input_ids  # (1, L)
        labels = labels.to(device)
        labels_lengths = torch.tensor([len(labels[0])], dtype=torch.int64, device=device)
        # print(f'label_lengths:{labels_lengths}')
        # print(f'labels:{labels}')
        labels_pad_mask = make_pad_mask(labels_lengths)  # B, L
        labels = labels.masked_fill(labels_pad_mask, 0)
        speech_embeds = self.embed_tokens(labels)  # B, L, D
        speech_target = torch.full(labels_pad_mask.shape, self.IGNORE_ID).to(
            speech_embeds.device)
        speech_masks = ~labels_pad_mask
        # speech_embeds, speech_masks, speech_target = self._add_bos_eos(0 + self.speech_token_num,
        #                                                                1 + self.speech_token_num,
        #                                                                speech_embeds, speech_masks, speech_target)

        qwen_instruct_prompt_pattern_1 = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        prompt_pattern1 = self.tokenizer([qwen_instruct_prompt_pattern_1] * len(speech_embeds), return_tensors="pt"
                                         )['input_ids'].to(speech_embeds.device)
        prompt_pattern1_embeds = self.embed_tokens(prompt_pattern1)

        qwen_instruct_prompt_pattern_2 = "<|im_end|>\n<|im_start|>assistant\n"
        prompt_pattern2 = self.tokenizer([qwen_instruct_prompt_pattern_2] * len(speech_embeds), return_tensors="pt"
                                         )['input_ids'].to(speech_embeds.device)
        prompt_pattern2_embeds = self.embed_tokens(prompt_pattern2)
        embeds = torch.cat([prompt_pattern1_embeds, speech_embeds, prompt_pattern2_embeds], dim=1)
        atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(embeds.device)

        if self.embed_tokens.weight.dtype == torch.float16 or self.embed_tokens.weight.dtype == torch.bfloat16:
            # utils_file.logging_limit_print('generate(): self.embed_tokens.weight.dtype == torch.float16')
            # embeds = embeds.to(torch.float16)
            embeds = embeds.to(torch.bfloat16)
            atts = atts.to(torch.bfloat16)
        outputs = self.llama_model.generate(
            inputs_embeds=embeds,
            max_new_tokens=self.max_length,
            num_beams=self.num_beams,
            do_sample=self.do_sample,
            min_length=self.min_length,
            top_p=self.top_p,
            top_k=self.top_k,
            repetition_penalty=1.0,
            length_penalty=1.0,
            temperature=self.temperature,
            attention_mask=atts,
            eos_token_id=151645,
            pad_token_id=-100,
            do_compile=True,
            cache_implementation="static",
        )
        output_text = self.tokenizer.batch_decode(outputs, add_special_tokens=False, skip_special_tokens=True)
        # output_text = [item.replace('<|endoftext|>', '') for item in output_text]
        return output_text

    def generate_tts(
            self,
            device,
            prompt,
            text,
    ):
        # =====================准备input embedding=====================
        self.llama_model.eval()

        self.set_task_type("TTS")
        self.do_add_speech_embed_head()
        # labels_lengths = torch.tensor([len(text[0])], dtype=torch.int64, device=device)
        # labels = text[:,:]
        labels = self.tokenizer(
            [text],
            return_tensors="pt",
            add_special_tokens=False
        ).to(
            self.embed_tokens.weight.device).input_ids  # (1, L)
        labels = labels.to(device)
        labels_lengths = torch.tensor([len(labels[0])], dtype=torch.int64, device=device)
        print(f'label_lengths:{labels_lengths}')
        print(f'labels:{labels}')
        labels_pad_mask = make_pad_mask(labels_lengths)  # B, L
        labels = labels.masked_fill(labels_pad_mask, 0)
        speech_embeds = self.embed_tokens(labels)  # B, L, D
        speech_target = torch.full(labels_pad_mask.shape, self.IGNORE_ID).to(
            speech_embeds.device)
        speech_masks = ~labels_pad_mask
        speech_embeds, speech_masks, speech_target = self._add_bos_eos(0 + self.speech_token_num,
                                                                       1 + self.speech_token_num,
                                                                       speech_embeds, speech_masks, speech_target)

        qwen_instruct_prompt_pattern_1 = "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        prompt_pattern1 = self.tokenizer([qwen_instruct_prompt_pattern_1] * len(speech_embeds), return_tensors="pt"
                                         )['input_ids'].to(speech_embeds.device)
        prompt_pattern1_embeds = self.embed_tokens(prompt_pattern1)

        qwen_instruct_prompt_pattern_2 = "<|im_end|>\n<|im_start|>assistant\n"
        prompt_pattern2 = self.tokenizer([qwen_instruct_prompt_pattern_2] * len(speech_embeds), return_tensors="pt"
                                         )['input_ids'].to(speech_embeds.device)
        prompt_pattern2_embeds = self.embed_tokens(prompt_pattern2)

        prompt = self.tokenizer([prompt], return_tensors="pt"
                                )['input_ids'].to(speech_embeds.device)
        prompt_embeds = self.embed_tokens(prompt)

        hyps = [self.speech_token_num - 1]
        speech_begin_token_emb = self.speech_token_emded(torch.tensor(hyps[-1:]).to(device)).unsqueeze(0)
        embeds = torch.cat([prompt_pattern1_embeds, prompt_embeds,
                            speech_embeds,
                            prompt_pattern2_embeds,
                            speech_begin_token_emb], dim=1).to(torch.bfloat16)
        # 指定top_k top_p temperature stop
        # max_len = 250
        top_k = 15  # 5
        top_p = 0.8  # 0.9
        temperature = 1.2  # 1

        print(f"tts eos id = {self.speech_token_num - 1}")
        llm_out = self.llama_model.generate(inputs_embeds=embeds,
                                            max_new_tokens=self.max_length,
                                            eos_token_id=self.speech_token_num - 1,
                                            cache_implementation="static",
                                            do_sample=True,
                                            temperature=temperature,
                                            top_k=top_k,
                                            top_p=top_p,
                                            stopping_criteria=StoppingCriteriaList([MaxTokenStopper(2000)]),
                                            do_compile=True,
                                            repetition_penalty=1.0,
                                            )

        print(llm_out)
        return llm_out

    def _get_embedding_from_wav(self, wavs, wavs_len):
        """
        return:
        wav_embedding: (b, l, v)
        wav_mask:  (b, l), wav为有效值的位置为true
        """
        encoder_out, encoder_mask = self.encoder(wavs, wavs_len)

        speech_embeds, encoder_mask = self.down_sample_2(encoder_out, encoder_mask)
        if self.speech_transformer is not None:
            filled_wavs_len = encoder_mask.squeeze(1).sum(-1)
            speech_embeds, encoder_mask = self.speech_transformer(speech_embeds, filled_wavs_len)
            speech_embeds = self.speech_llama_proj(speech_embeds)
        return speech_embeds, encoder_mask.squeeze(1)

    def _get_embedding_from_text(self, text):
        """
        将字符串先量化，再转成词向量

        Args:
            text: str

        Returns:
            text_embeds: (1, L, D)

        """
        text_id = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False
        ).to(
            self.embed_tokens.weight.device).input_ids
        text_embeds = self.embed_tokens(text_id)
        text_embeds_len = torch.tensor([text_embeds.size(1)], dtype=torch.long)
        return text_embeds, text_embeds_len

    def _add_bos_eos(self, bos, eos, inputs_embeds, attention_mask, target=None):
        B = len(inputs_embeds)
        bos_eos_target = torch.full([B, 1], self.IGNORE_ID).to(inputs_embeds.device)  # B,1
        bos_eos_mask = torch.full([B, 1], True).to(inputs_embeds.device)  # B, 1

        if bos is not None:
            bos_embed = self.speech_token_emded(torch.full([B, 1],
                                                           bos).to(inputs_embeds.device))  # B, 1, D
            inputs_embeds = torch.cat((bos_embed, inputs_embeds), 1)  # B, (1+T), D
            attention_mask = torch.cat((bos_eos_mask, attention_mask), 1)  # B, (1+T)
            if target is not None:
                target = torch.cat((bos_eos_target, target), 1)  # B, (1+T), D

        if eos is not None:
            eos_embed = self.speech_token_emded(torch.full([B, 1],
                                                           eos).to(inputs_embeds.device))  # B, 1, D
            inputs_embeds = torch.cat((inputs_embeds, eos_embed), 1)  # B, (1+T+1), D
            attention_mask = torch.cat((attention_mask, bos_eos_mask), 1)  # B, (1+T+1)
            if target is not None:
                target = torch.cat((target, bos_eos_target), 1)  # B, (1+T+1), D

        return inputs_embeds, attention_mask, target

    def _sampler(
            self,
            logits: torch.Tensor,
            temperatures: Union[torch.Tensor, None],
            top_ps: torch.Tensor,
            top_ks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample from logits.
        Args:
            logits:
            temperatures:
            top_ps:
            top_ks:

        Returns:

        """
        assert logits.size(1) == 1
        logits = logits.squeeze(1)  # (batch_size, vocab_size)
        if temperatures is None:
            return torch.argmax(logits, dim=-1).squeeze(dim=-1)
        # Apply temperature scaling.
        logits.div_(temperatures.unsqueeze(dim=1))
        # Calculate probabilities with softmax.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        # Apply top-p, top-k.
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        top_ps_mask = (probs_sum - probs_sort) > top_ps.unsqueeze(dim=1)
        probs_sort = torch.where(top_ps_mask, 0, probs_sort)

        top_ks_mask = torch.arange(probs_idx.shape[-1], device=probs_idx.device)
        top_ks_mask = top_ks_mask.expand(probs_idx.shape[0], -1)
        top_ks_mask = top_ks_mask >= top_ks.unsqueeze(dim=1)
        probs_sort = torch.where(top_ks_mask, 0, probs_sort)
        # Re-normalization.
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        probs = torch.gather(probs_sort,
                             dim=-1,
                             index=torch.argsort(probs_idx, dim=-1))
        next_token_ids = torch.multinomial(probs, num_samples=1,
                                           replacement=True).squeeze(dim=-1)
        return next_token_ids

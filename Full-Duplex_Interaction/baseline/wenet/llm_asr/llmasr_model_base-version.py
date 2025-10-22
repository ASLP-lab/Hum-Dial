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
import torch.nn.functional as F
import math
import gc


# import torch_npu
# from torch_npu.contrib import transfer_to_npu

# from msprobe.pytorch import seed_all,PrecisionDebugger

class LLMASR_Model(nn.Module):
    def __init__(self,
                 encoder,
                 encoder_output_dim,
                 llm_path,
                 lora=True, lora_alpha=32, lora_rank=8, lora_dropout=0.1,
                 prompt_pattern="{}：<Speech><SpeechHere></Speech>",
                 # "USER: <Speech><SpeechHere></Speech> {}\nASSISTANT:"
                 is_inference=False,
                 downsample_rate=1,
                 llm_embed_dim=4096,
                 task_num=2,
                 adapter_type='lyz',
                 speech_token_num=0,
                 train_speech_out=False):
        """"""
        super().__init__()
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

        self.max_length = 400
        self.min_length = 1
        self.num_beams = 4
        self.do_sample = True
        self.top_p = 0.9
        self.top_k = 5
        self.repetition_penalty = 1.05
        self.length_penalty = 1.0
        self.temperature = 1.0
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
            self.speaker_head = torch.nn.Linear(self.llama_model.config.hidden_size, speech_token_num)
        else:

            # 不做任何处理
            self.speaker_head = nn.Identity()
            self.speech_token_emded = nn.Identity()
        self.train_speech_out = train_speech_out
        utils_file.logging_info(f'耿雪龙： 是否进行语音输出训练：{self.train_speech_out}')
        self.loss_fct = CrossEntropyLoss(reduction='mean')
        # self.debugger = PrecisionDebugger(config_path='./do_align_test/config_gpu.json', model=self.encoder)

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
        if output_type == 'speech2text_token':
            utils_file.logging_limit_print(f'xxx 开始处理speech2text_token任务')
            labels = batch['target'].to(device)
            labels_lengths = batch['target_lengths'].to(device)
            speech_token_labels = batch['speech_tokens'].to(device)
            speech_tokens_length = batch['speech_tokens_length'].to(device)

            labels_embeds, labels_target, labels_mask = self.get_label_embedding(labels, labels_lengths)
            # utils_file.logging_limit_print(f'xxx labels embeding shape {labels_embeds.shape}')
            # utils_file.logging_limit_print(f'xxx labels mask shape {labels_mask.shape}')
            # utils_file.logging_limit_print(f'xxx labels target shape {labels_target.shape}')
            # utils_file.logging_limit_print(f'xxx labels mask 0 {labels_mask[0]}')
            # utils_file.logging_limit_print(f'xxx labels target 0 {labels_target[0]}')
            speech_token_labels_embeds, speech_token_labels_target, speech_token_labels_mask = self.get_speech_token_label_embedding(
                speech_token_labels, speech_tokens_length)
            # utils_file.logging_limit_print(f'xxx speech_token_labels embeding shape {speech_token_labels_embeds.shape}')
            # utils_file.logging_limit_print(f'xxx speech_token_labels mask shape {speech_token_labels_mask.shape}')
            # utils_file.logging_limit_print(f'xxx speech_token_labels target shape {speech_token_labels_target.shape}')
            # utils_file.logging_limit_print(f'xxx speech_token_labels mask 0 {speech_token_labels_mask[0]}')
            # utils_file.logging_limit_print(f'xxx speech_token_labels target 0 {speech_token_labels_target[0]}')
            # utils_file.logging_limit_print(f'开始合并labels和speech_token_labels，合并前：labels_embeds:{labels_embeds.shape}, labels_target:{labels_target.shape}, labels_mask:{labels_mask.shape}, speech_token_labels_embeds:{speech_token_labels_embeds.shape}, speech_token_labels_target:{speech_token_labels_target.shape}, speech_token_labels_mask:{speech_token_labels_mask.shape}')
            # big_labels_embeds, big_labels_target, big_labels_mask = merge_labels_with_valid_adjacent(
            #     labels_embeds, labels_target, labels_mask, speech_token_labels_embeds, speech_token_labels_target, speech_token_labels_mask
            # )
            # utils_file.logging_limit_print(f'开始合并labels和speech_token_labels，合并后：big_labels_embeds:{big_labels_embeds.shape}, big_labels_target:{big_labels_target.shape}, big_labels_mask:{big_labels_mask.shape}')
            # utils_file.logging_limit_print(f'xxx big_labels embeding shape {big_labels_embeds.shape}')
            # utils_file.logging_limit_print(f'xxx big_labels mask shape {big_labels_mask.shape}')
            # utils_file.logging_limit_print(f'xxx big_labels target shape {big_labels_target.shape}')
            # utils_file.logging_limit_print(f'xxx big_labels mask 0 {big_labels_mask[0]}')
            # utils_file.logging_limit_print(f'xxx big_labels target 0 {big_labels_target[0]}')
            if prompt_embeds is not None:
                inputs_embeds_list.append(prompt_embeds)
                attention_mask_list.append(prompt_mask)
                target_list.append(prompt_target)
            # inputs_embeds_list.extend([speech_embeds, big_labels_embeds])
            # attention_mask_list.extend([speech_masks, big_labels_mask])
            # target_list.extend([speech_target, big_labels_target])

            # inputs_embeds_list.extend([speech_embeds, speech_token_labels_embeds])
            # attention_mask_list.extend([speech_masks, speech_token_labels_mask])
            # target_list.extend([speech_target, speech_token_labels_target])

            inputs_embeds_list.extend([speech_embeds, labels_embeds, speech_token_labels_embeds])
            attention_mask_list.extend([speech_masks, labels_mask, speech_token_labels_mask])
            target_list.extend([speech_target, labels_target, speech_token_labels_target])
        elif output_type == "text2token":
            speech_token_labels = batch['speech_tokens'].to(device)
            speech_tokens_length = batch['speech_tokens_length'].to(device)
            speech_token_labels_embeds, speech_token_labels_target, speech_token_labels_mask = self.get_speech_token_label_embedding(
                speech_token_labels, speech_tokens_length)
            if prompt_embeds is not None:
                inputs_embeds_list.append(prompt_embeds)
                attention_mask_list.append(prompt_mask)
                target_list.append(prompt_target)
            inputs_embeds_list.extend([speech_embeds, speech_token_labels_embeds])
            attention_mask_list.extend([speech_masks, speech_token_labels_mask])
            target_list.extend([speech_target, speech_token_labels_target])
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
            inputs_embeds_list.extend([speech_embeds, labels_embeds])
            attention_mask_list.extend([speech_masks, labels_mask])
            target_list.extend([speech_target, labels_target])
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
            logits2 = self.speaker_head(hidden_states)  # speech_head
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

    def generate(
            self,
            wavs,
            wavs_len,
            prompt,
    ):
        speech_embeds, speech_masks = self._get_embedding_from_wav(wavs, wavs_len)
        speech_embeds, speech_masks, _ = self._add_bos_eos(0 + self.speech_token_num, 1 + self.speech_token_num,
                                                           speech_embeds, speech_masks, None)
        prompt = self.tokenizer([prompt], return_tensors="pt"
                                )['input_ids'].to(speech_embeds.device)
        prompt_embeds = self.embed_tokens(prompt)

        embeds = torch.cat([prompt_embeds, speech_embeds], dim=1)
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
            repetition_penalty=self.repetition_penalty,
            length_penalty=self.length_penalty,
            temperature=self.temperature,
            attention_mask=atts,
            eos_token_id=151643,
            pad_token_id=-100,
        )

        # 获取生成的token IDs
        # token_ids = outputs[0].tolist()  # 假设batch_size=1，取第一个输出
        # 将token IDs转换为字符串
        # tokens = [self.tokenizer.decode([token_id], skip_special_tokens=True) for token_id in token_ids]
        # 打印token列表和字符串列表
        # print("Token IDs:", token_ids)
        # print("Tokens:", tokens)

        # 使用tokenizer将token IDs批量转换为字符串
        # output_text = self.tokenizer.batch_decode(outputs, add_special_tokens=False, skip_special_tokens=True)
        # print("Output Text:", output_text)

        output_text = self.tokenizer.batch_decode(outputs, add_special_tokens=False, skip_special_tokens=True)
        # 处理token，为英文单词前加上空格
        # processed_tokens = []
        # for token in tokens:
        #     # 检查是否为英文单词（简单判断：是否全部由字母组成）
        #     if token.isalpha() and token[0].isascii():
        #         processed_tokens.append(" " + token)  # 英文单词前加空格
        #     else:
        #         processed_tokens.append(token)  # 其他token保持不变
        # output_text = "".join(processed_tokens)
        return output_text

    def generate_s2s(
            self,
            wavs,
            wavs_len,
            prompt,
    ):
        self.llama_model.eval()
        speech_embeds, speech_masks = self._get_embedding_from_wav(wavs, wavs_len)
        speech_embeds, speech_masks, _ = self._add_bos_eos(0 + self.speech_token_num, None,
                                                           speech_embeds, speech_masks, None)

        device = speech_embeds.device

        prompt = self.tokenizer([prompt], return_tensors="pt"
                                )['input_ids'].to(speech_embeds.device)
        prompt_embeds = self.embed_tokens(prompt)
        embeds = torch.cat([prompt_embeds, speech_embeds], dim=1)
        atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(embeds.device)
        if self.embed_tokens.weight.dtype == torch.float16:
            # utils_file.logging_limit_print('generate(): self.embed_tokens.weight.dtype == torch.float16')
            embeds = embeds.to(torch.float16)
        max_len = 350
        hyps = [4098]
        llm_out = self.llama_model(
            inputs_embeds=embeds,
            past_key_values=None,
            output_hidden_states=True
        )

        batch_size = 1
        top_k = 10
        top_p = 0.9
        temperature = 1.2
        cache = llm_out.past_key_values
        token_emb = self.speech_token_emded(torch.tensor(hyps[-1:]).to(device)).unsqueeze(0)
        top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(device)
        top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(device)
        temperatures_tensor = None if not temperature else torch.FloatTensor(
            [temperature] * batch_size).to(device)
        is_speech_token = False
        hyps_text = ""
        speech_eos_num = 0
        for i in range(max_len):
            llm_out = self.llama_model(
                inputs_embeds=token_emb,
                past_key_values=cache,
                output_hidden_states=True
            )
            cache = llm_out.past_key_values
            hidden_states = llm_out.hidden_states[-1]
            if is_speech_token:
                token_logits = self.speaker_head(hidden_states)  # (B, )
            else:
                token_logits = self.lm_head(hidden_states)
            next_token_ids = self._sampler(
                token_logits,
                temperatures_tensor,
                top_ps_tensor,
                top_ks_tensor,
            )
            print(i, next_token_ids, f'is_speech_token:{is_speech_token}')
            if next_token_ids == 151643:
                print("text is over")
                print("hyps:", hyps)
                is_speech_token = True
                hyps_text = self.tokenizer.decode(hyps[1:], skip_special_tokens=True, add_special_tokens=False)
                print("hyps_text:", hyps_text)
                hyps = []
            if is_speech_token and next_token_ids == self.speech_token_num - 1:
                speech_eos_num += 1
                print(f'遇到 4096')
                if speech_eos_num >= 2:
                    print("break la!")
                    print("hyps:", hyps)
                    break
            hyps.append(next_token_ids.item())
            # if 1+i >= len(speech_token[0]):
            #     break
            # token_emb = self.speech_token_emded(torch.tensor([speech_token[0][i+1]]).to(device)).unsqueeze(0)
            if next_token_ids != 151643 and is_speech_token:
                token_emb = self.speech_token_emded(torch.tensor(hyps[-1:]).to(device)).unsqueeze(0)
            else:
                token_emb = self.embed_tokens(torch.tensor(hyps[-1:]).to(device)).unsqueeze(0)
        res = []
        for i in hyps[2:]:
            # if i != self.speech_token_num-1:
            res.append(i)
        print("res", res)
        return [hyps_text +"|" + str(res)]


    def generate_tts(
            self,
            device,
            prompt,
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
            self.embed_tokens.weight.device).input_ids # (1, L)
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

        prompt = self.tokenizer([prompt], return_tensors="pt"
                                )['input_ids'].to(speech_embeds.device)
        prompt_embeds = self.embed_tokens(prompt)
        embeds = torch.cat([prompt_embeds, speech_embeds], dim=1)
        max_len = 250
        hyps = [self.speech_token_num - 1]
        llm_out = self.llama_model(
            inputs_embeds=embeds,
            past_key_values=None,
            output_hidden_states=True
        )
        batch_size = 1
        top_k = 10
        top_p = 0.9
        temperature = 1.0
        cache = llm_out.past_key_values
        token_emb = self.speech_token_emded(torch.tensor(hyps[-1:]).to(device)).unsqueeze(0)
        top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(device)
        top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(device)
        temperatures_tensor = None if not temperature else torch.FloatTensor(
            [temperature] * batch_size).to(device)

        for i in range(max_len):
            llm_out = self.llama_model(
                inputs_embeds=token_emb,
                past_key_values=cache,
                output_hidden_states=True
            )
            del cache
            # for item in cache:
            #     item = item.cpu()
            cache = llm_out.past_key_values
            hidden_states = llm_out.hidden_states[-1]
            token_logits = self.speaker_head(hidden_states)
            del llm_out, hidden_states, token_emb
            gc.collect()
            torch.cuda.empty_cache()
            # probs =  F.log_softmax(token_logits[:,-1], dim=-1)[0]
            next_token_ids = self._sampler(
                token_logits,
                temperatures_tensor,
                top_ps_tensor,
                top_ks_tensor,
            )
            print(i, next_token_ids)
            if next_token_ids == self.speech_token_num - 1:
                print("break la!")
                print("hyps:", hyps)
                break
            hyps.append(next_token_ids.item())
            token_emb = self.speech_token_emded(torch.tensor(hyps[-1:]).to(device)).unsqueeze(0)
        res = []
        for i in hyps[1:]:
            res.append(i)
        print(res)
        return [res]


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
            # if rank == 0:
            #     utils_file.logging_limit_print(
            #         f'out of link shape: {speech_embeds.shape},encoder的第一帧的前20个数字：\n {speech_embeds[0][0][:20]}')

            # utils_file.logging_limit_print(
            #     'get_embedding_from_wav(): speech_embeds.shape,by  self.speech_transformer(speech_embeds, speech_lens):',
            #     speech_embeds.shape)
            speech_embeds = self.speech_llama_proj(speech_embeds)
            # if rank == 0:
            #     utils_file.logging_limit_print(
            #         f'out of speech_llama_proj shape: {speech_embeds.shape},encoder的第一帧的前20个数字：\n {speech_embeds[0][0][:20]}')

        # utils_file.logging_limit_print(
        #     'get_embedding_from_wav(): speech_embeds.shape,by  self.speech_llama_proj(speech_embeds):',
        #     speech_embeds.shape)

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


    def infer_sample4speech2text_token(
            self,
            wavs,
            wavs_len,
            prompt,
            speech_token=None,
    ):
        self.llama_model.eval()
        speech_embeds, speech_masks = self._get_embedding_from_wav(wavs, wavs_len)
        speech_embeds, speech_masks, _ = self._add_bos_eos(0 + self.speech_token_num, 1 + self.speech_token_num,
                                                           speech_embeds, speech_masks, None)

        device = speech_embeds.device

        prompt = self.tokenizer([prompt], return_tensors="pt"
                                )['input_ids'].to(speech_embeds.device)
        prompt_embeds = self.embed_tokens(prompt)
        # true_speech_token_part = torch.tensor([[4096,304, 1896, 1385, 1385, 1385,3,3,26]], dtype=torch.long, device=device)
        # true_speech_token_part_embeds = self.embed_tokens(true_speech_token_part)
        embeds = torch.cat([prompt_embeds, speech_embeds], dim=1)
        atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(embeds.device)
        if self.embed_tokens.weight.dtype == torch.float16:
            utils_file.logging_limit_print('generate(): self.embed_tokens.weight.dtype == torch.float16')
            embeds = embeds.to(torch.float16)
            atts = atts.half()
        inputs_embeds = embeds.to(speech_embeds.device)

        max_len = 350
        beam = 3
        beam_size = beam
        running_size = beam
        output_token = []
        hyps = [4096]
        hyps_text = ""

        scores = [1.0]
        llm_out = self.llama_model(
            inputs_embeds=embeds,
            past_key_values=None,
            output_hidden_states=True
        )
        # speech_token_list = speech_token[0]
        # speech_token_list_len = len(speech_token_list)
        if speech_token is not None:
            print(f'speech_token_list_len:{len(speech_token[0])}')
            print(f'speech_token:{speech_token[0]}')

        batch_size = 1
        top_k = 10
        top_p = 0.9
        temperature = 1.2
        cache = llm_out.past_key_values
        token_emb = self.speech_token_emded(torch.tensor(hyps[-1:]).to(device)).unsqueeze(0)
        repetition_penalty = 1.1
        top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(device)
        top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(device)
        temperatures_tensor = None if not temperature else torch.FloatTensor(
            [temperature] * batch_size).to(device)
        is_speech_token = True
        hyps_text = ""
        speech_eos_num = 0
        for i in range(max_len):
            llm_out = self.llama_model(
                inputs_embeds=token_emb,
                past_key_values=cache,
                output_hidden_states=True
            )
            cache = llm_out.past_key_values
            hidden_states = llm_out.hidden_states[-1]
            if is_speech_token:
                token_logits = self.speaker_head(hidden_states)  # (B, )
            else:
                token_logits = self.lm_head(hidden_states)
            # probs =  F.log_softmax(token_logits[:,-1], dim=-1)[0]
            # next_token_ids = torch.argmax(probs,dim=-1).item()
            # next_token_ids = torch.tensor(next_token_ids, dtype=torch.long).to(device)
            # if i ==2 or i == 80:
            #     torch.save(probs, f'probs_{i}.pt')
            next_token_ids = self._sampler(
                token_logits,
                temperatures_tensor,
                top_ps_tensor,
                top_ks_tensor,
            )
            print(i, next_token_ids, f'is_speech_token:{is_speech_token}')
            if next_token_ids == 151643:
                print("text is over")
                print("hyps:", hyps)
                is_speech_token = True
                hyps_text = self.tokenizer.decode(hyps[1:], skip_special_tokens=True, add_special_tokens=False)
                print("hyps_text:", hyps_text)
                hyps = [151643]
            if is_speech_token and next_token_ids == self.speech_token_num - 1:
                speech_eos_num += 1
                print(f'遇到 4096')
                # if speech_eos_num >= 2:
                print("break la!")
                print("hyps:", hyps)
                break
            hyps.append(next_token_ids.item())
            # if 1+i >= len(speech_token[0]):
            #     break
            # token_emb = self.speech_token_emded(torch.tensor([speech_token[0][i+1]]).to(device)).unsqueeze(0)
            token_emb = self.speech_token_emded(torch.tensor(hyps[-1:]).to(device)).unsqueeze(0)
        res = []
        for i in hyps[1:]:
            # if i != self.speech_token_num-1:
            res.append(i)
        print(res)
        return [hyps_text + str(res)]



    def infer_sample_teach_force(
            self,
            wavs,
            wavs_len,
            prompt,
            text,
            speech_token,
    ):
        labels_lengths = torch.tensor([len(text[0])], dtype=torch.int64, device=wavs.device)
        labels = text[:, :]
        labels_pad_mask = make_pad_mask(labels_lengths)  # B, L
        labels = labels.masked_fill(labels_pad_mask, 0)
        speech_embeds = self.embed_tokens(labels)  # B, L, D
        speech_target = torch.full(labels_pad_mask.shape, self.IGNORE_ID).to(
            speech_embeds.device)
        speech_masks = ~labels_pad_mask
        speech_embeds, speech_masks, speech_target = self._add_bos_eos(0 + self.speech_token_num,
                                                                       1 + self.speech_token_num,
                                                                       speech_embeds, speech_masks, speech_target)

        prompt = self.tokenizer([prompt], return_tensors="pt"
                                )['input_ids'].to(speech_embeds.device)
        prompt_embeds = self.embed_tokens(prompt)
        embeds = torch.cat([prompt_embeds, speech_embeds], dim=1)
        atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(embeds.device)
        if self.embed_tokens.weight.dtype == torch.float16:
            utils_file.logging_limit_print('generate(): self.embed_tokens.weight.dtype == torch.float16')
            embeds = embeds.to(torch.float16)
            atts = atts.half()
        device = wavs.device
        inputs_embeds = embeds.to(device)

        speech_token_list = speech_token[0].tolist()
        speech_token_list_len = len(speech_token_list)
        print(f'speech_token_list_len:{speech_token_list_len}')
        max_len = 200
        beam = 3
        beam_size = beam
        running_size = beam
        output_token = []
        hyps = [self.speech_token_num - 1]

        scores = [1.0]
        llm_out = self.llama_model(
            inputs_embeds=embeds,
            past_key_values=None,
            output_hidden_states=True
        )
        batch_size = 1
        top_k = 10
        top_p = 0.9
        temperature = 1.0
        cache = llm_out.past_key_values
        token_emb = self.speech_token_emded(torch.tensor(hyps[-1:]).to(device)).unsqueeze(0)
        repetition_penalty = 1.1
        top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(device)
        top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(device)
        temperatures_tensor = None if not temperature else torch.FloatTensor(
            [temperature] * batch_size).to(device)

        for i in range(max_len):
            llm_out = self.llama_model(
                inputs_embeds=token_emb,
                past_key_values=cache,
                output_hidden_states=True
            )
            cache = llm_out.past_key_values
            hidden_states = llm_out.hidden_states[-1]
            token_logits = self.speaker_head(hidden_states)
            # probs =  F.log_softmax(token_logits[:,-1], dim=-1)[0]
            next_token_ids = self._sampler(
                token_logits,
                temperatures_tensor,
                top_ps_tensor,
                top_ks_tensor,
            )
            print(i, next_token_ids)
            if next_token_ids == self.speech_token_num - 1:
                print("break la!")
                print("hyps:", hyps)
                break
            hyps.append(next_token_ids.item())
            token_emb = self.speech_token_emded(torch.tensor(speech_token_list[i]).to(device)).unsqueeze(0)
        res = []
        for i in hyps[1:]:
            # if i != self.speech_token_num-1:
            res.append(i)
        print(res)
        return [res]

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

    def infer_sample4speech2text_token_teacher_force(
            self,
            wavs,
            wavs_len,
            prompt,
            speech_token=None,
            answer_text=None,
    ):
        self.llama_model.eval()
        speech_embeds, speech_masks = self._get_embedding_from_wav(wavs, wavs_len)
        speech_embeds, speech_masks, _ = self._add_bos_eos(0 + self.speech_token_num, 1 + self.speech_token_num,
                                                           speech_embeds, speech_masks, None)

        device = speech_embeds.device

        prompt = self.tokenizer([prompt], return_tensors="pt"
                                )['input_ids'].to(speech_embeds.device)
        prompt_embeds = self.embed_tokens(prompt)
        text_token = self.tokenizer([answer_text], return_tensors="pt"
                                    )['input_ids'].to(speech_embeds.device)
        text_token_embeds = self.embed_tokens(text_token)
        embeds = torch.cat([prompt_embeds, speech_embeds, text_token_embeds], dim=1)
        atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(embeds.device)
        if self.embed_tokens.weight.dtype == torch.float16:
            utils_file.logging_limit_print('generate(): self.embed_tokens.weight.dtype == torch.float16')
            embeds = embeds.to(torch.float16)
            atts = atts.half()
        inputs_embeds = embeds.to(speech_embeds.device)

        max_len = 150
        beam = 3
        beam_size = beam
        running_size = beam
        output_token = []
        hyps = [self.speech_token_num]
        hyps_text = ""

        scores = [1.0]
        llm_out = self.llama_model(
            inputs_embeds=embeds,
            past_key_values=None,
            output_hidden_states=True
        )
        # speech_token_list = speech_token[0]
        # speech_token_list_len = len(speech_token_list)
        if speech_token is not None:
            print(f'speech_token_list_len:{len(speech_token[0])}')
            print(f'speech_token:{speech_token[0]}')

        batch_size = 1
        top_k = 10
        top_p = 0.9
        temperature = 1.2
        cache = llm_out.past_key_values
        token_emb = self.speech_token_emded(torch.tensor(hyps[-1:]).to(device)).unsqueeze(0)
        repetition_penalty = 1.1
        top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(device)
        top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(device)
        temperatures_tensor = None if not temperature else torch.FloatTensor(
            [temperature] * batch_size).to(device)
        is_speech_token = False
        speech_eos_num = 0
        for i in range(max_len):
            llm_out = self.llama_model(
                inputs_embeds=token_emb,
                past_key_values=cache,
                output_hidden_states=True
            )
            cache = llm_out.past_key_values
            hidden_states = llm_out.hidden_states[-1]
            token_logits = self.speaker_head(hidden_states)
            # probs =  F.log_softmax(token_logits[:,-1], dim=-1)[0]
            # if i ==2 or i == 80:
            #     torch.save(probs, f'probs_{i}.pt')
            next_token_ids = self._sampler(
                token_logits,
                temperatures_tensor,
                top_ps_tensor,
                top_ks_tensor,
            )
            print(i, next_token_ids, f'is_speech_token:{is_speech_token}')
            if next_token_ids == self.speech_token_num - 1:
                print(f'遇到 4096')
                break
            hyps.append(next_token_ids.item())
            # if 1+i >= len(speech_token[0]):
            #     break
            # token_emb = self.speech_token_emded(torch.tensor([speech_token[0][i+1]]).to(device)).unsqueeze(0)
            token_emb = self.embed_tokens(torch.tensor(hyps[-1:]).to(device)).unsqueeze(0)
        res = []
        for i in hyps[1:]:
            # if i != self.speech_token_num-1:
            res.append(i)
        print(res)
        return [answer_text + str(res[2:])]

    def infer_sample4speech2text_token_teacher_force2(
            self,
            wavs,
            wavs_len,
            prompt,
            speech_token=None,
            answer_text=None,
    ):
        self.llama_model.eval()
        speech_embeds, speech_masks = self._get_embedding_from_wav(wavs, wavs_len)
        speech_embeds, speech_masks, _ = self._add_bos_eos(0 + self.speech_token_num, 1 + self.speech_token_num,
                                                           speech_embeds, speech_masks, None)

        device = speech_embeds.device

        prompt = self.tokenizer([prompt], return_tensors="pt"
                                )['input_ids'].to(speech_embeds.device)
        prompt_embeds = self.embed_tokens(prompt)
        text_token = self.tokenizer([answer_text], return_tensors="pt"
                                    )['input_ids'].to(speech_embeds.device)
        # text_token_embeds = self.embed_tokens(text_token)
        embeds = torch.cat([prompt_embeds, speech_embeds], dim=1)
        atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(embeds.device)
        if self.embed_tokens.weight.dtype == torch.float16:
            utils_file.logging_limit_print('generate(): self.embed_tokens.weight.dtype == torch.float16')
            embeds = embeds.to(torch.float16)
            atts = atts.half()
        inputs_embeds = embeds.to(speech_embeds.device)

        max_len = 150
        beam = 3
        beam_size = beam
        running_size = beam
        output_token = []
        hyps = [self.speech_token_num - 1]
        hyps_text = ""

        scores = [1.0]
        llm_out = self.llama_model(
            inputs_embeds=embeds,
            past_key_values=None,
            output_hidden_states=True
        )
        # speech_token_list = speech_token[0]
        # speech_token_list_len = len(speech_token_list)
        if speech_token is not None:
            print(f'speech_token_list_len:{len(speech_token)}')
            print(f'speech_token:{speech_token}')

        batch_size = 1
        top_k = 10
        top_p = 0.9
        temperature = 1.2
        cache = llm_out.past_key_values
        token_emb = self.speech_token_emded(torch.tensor(hyps[-1:]).to(device)).unsqueeze(0)
        repetition_penalty = 1.1
        top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(device)
        top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(device)
        temperatures_tensor = None if not temperature else torch.FloatTensor(
            [temperature] * batch_size).to(device)
        is_speech_token = False
        speech_eos_num = 0
        token_num = len(speech_token)
        for i in range(token_num):
            llm_out = self.llama_model(
                inputs_embeds=token_emb,
                past_key_values=cache,
                output_hidden_states=True
            )
            cache = llm_out.past_key_values
            hidden_states = llm_out.hidden_states[-1]
            token_logits = self.speaker_head(hidden_states)
            # probs =  F.log_softmax(token_logits[:,-1], dim=-1)[0]
            # if i ==2 or i == 80:
            #     torch.save(probs, f'probs_{i}.pt')
            next_token_ids = self._sampler(
                token_logits,
                temperatures_tensor,
                top_ps_tensor,
                top_ks_tensor,
            )
            print(i, next_token_ids, f'is_speech_token:{is_speech_token}')
            if next_token_ids == self.speech_token_num - 1:
                print(f'遇到 4096')
                break
            hyps.append(next_token_ids.item())
            # if 1+i >= len(speech_token[0]):
            #     break
            # token_emb = self.speech_token_emded(torch.tensor([speech_token[0][i+1]]).to(device)).unsqueeze(0)
            token_emb = self.embed_tokens(torch.tensor([speech_token[i]]).to(device)).unsqueeze(0)
        res = []
        for i in hyps[1:]:
            # if i != self.speech_token_num-1:
            res.append(i)
        print(res)
        return [hyps_text + str(res)]
import logging
import os

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
        self.top_p = 0.0
        self.top_k = 0
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
        utils_file.logging_limit_print(f'进行speech_token_labels修改，修改前 speech_token_labels',
                                       speech_token_labels.shape, speech_token_labels[0][-1], speech_token_labels[0][0])
        speech_token_labels = speech_token_labels + 152064
        utils_file.logging_limit_print(f'进行speech_token_labels修改，修改后 speech_token_labels',
                                       speech_token_labels.shape, speech_token_labels[0][-1], speech_token_labels[0][0])
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
        assert output_type in ['text', 'speech2text_token', 'text2token'], f"output_type:{output_type} not support"
        # speech inputs
        if output_type == 'text' or output_type == 'speech2text_token':
            wavs = batch['feats'].to(device)
            wavs_len = batch['feats_lengths'].to(device)
            ssl_vecs = batch['ssl_vecs'].to(device)
            speech_embeds, speech_masks = self.get_embedding_from_wav(wavs, wavs_len, ssl_vecs)
            utils_file.logging_limit_print(f'speech_embeds.shape: {speech_embeds.shape}, speech_masks.shape: {speech_masks.shape}')
            

            speech_target = torch.full(speech_masks.shape, self.IGNORE_ID).to(
                speech_embeds.device)
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
        else:
            prompt_embeds = None
            prompt_mask = None
            prompt_target = None

        inputs_embeds_list = []
        attention_mask_list = []
        target_list = []
        if output_type == 'speech2text_token':
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
            target_list.extend([speech_target,  speech_token_labels_target])
        elif output_type == "text":
            labels = batch['target'].to(device)
            labels_lengths = batch['target_lengths'].to(device)
            labels_embeds, labels_target, labels_mask = self.get_label_embedding(labels, labels_lengths)
            if prompt_embeds is not None:
                inputs_embeds_list.append(prompt_embeds)
                attention_mask_list.append(prompt_mask)
                target_list.append(prompt_target)
            else:
                utils_file.logging_limit_print(f'prompt is None,task: {batch["task"]}, prompt_embeds:{prompt_embeds}, prompt_mask:{prompt_mask}')
            inputs_embeds_list.extend([speech_embeds, labels_embeds])
            attention_mask_list.extend([speech_masks, labels_mask])
            target_list.extend([speech_target, labels_target])
        else:
            raise NotImplementedError(f'output_type {output_type} not support')

        inputs_embeds = torch.cat(inputs_embeds_list, dim=1)
        attention_mask = torch.cat(attention_mask_list, dim=1)
        target = torch.cat(target_list, dim=1)
        # utils_file.logging_limit_print(f'耿雪龙 output_type: {output_type}')
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
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
        speech_embeds, speech_masks = self.get_embedding_from_wav(wavs, wavs_len)
        speech_embeds, speech_masks, _ = self._add_bos_eos(0 + self.speech_token_num, 1 + self.speech_token_num,
                                                           speech_embeds, speech_masks, None)
        if prompt != "<no_prompt>":
            prompt = self.tokenizer([prompt], return_tensors="pt"
                                    )['input_ids'].to(speech_embeds.device)
            prompt_embeds = self.embed_tokens(prompt)
        else:
            prompt_embeds = None

        if prompt_embeds is not None:
            embeds = torch.cat([prompt_embeds, speech_embeds], dim=1)
        else:
            embeds = speech_embeds
        
        atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(embeds.device)

        if self.embed_tokens.weight.dtype == torch.float16 or self.embed_tokens.weight.dtype == torch.bfloat16:
            utils_file.logging_limit_print('generate(): self.embed_tokens.weight.dtype == torch.float16')
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

        output_text = self.tokenizer.batch_decode(outputs, add_special_tokens=False, skip_special_tokens=True)

        return output_text

    def generate4speech_token(
            self,
            wavs,
            wavs_len,
            prompt,
    ):
        speech_embeds, speech_masks = self.get_embedding_from_wav(wavs, wavs_len)
        speech_embeds, speech_masks, _ = self._add_bos_eos(0 + self.speech_token_num, 1 + self.speech_token_num,
                                                           speech_embeds, speech_masks, None)
        prompt = self.tokenizer([prompt], return_tensors="pt"
                                )['input_ids'].to(speech_embeds.device)
        prompt_embeds = self.embed_tokens(prompt)

        embeds = torch.cat([prompt_embeds, speech_embeds], dim=1)
        atts = torch.ones(embeds.size()[:-1], dtype=torch.long).to(embeds.device)

        if self.embed_tokens.weight.dtype == torch.float16:
            utils_file.logging_limit_print('generate(): self.embed_tokens.weight.dtype == torch.float16')
            embeds = embeds.to(torch.float16)
            atts = atts.half()

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
        output_text = self.tokenizer.batch_decode(outputs, add_special_tokens=False, skip_special_tokens=True)

        return output_text

    def get_embedding_from_wav(self, wavs, wavs_len, ssl_vecs=None):
        """
        return:
        wav_embedding: (b, l, v)
        wav_mask:  (b, l), wav为有效值的位置为true
        """
        # utils_file.logging_limit_print('get_embedding_from_wav(): wavs.shape:', wavs.shape)
        # utils_file.logging_limit_print('get_embedding_from_wav(): wavs_len.shape:', wavs_len.shape)
        rank = int(os.environ.get('RANK', 0))
        # self.debugger.start()
        encoder_out, encoder_mask = self.encoder(wavs, wavs_len)
        # self.debugger.stop()
        # self.debugger.step()
        # if rank == 0:
        #     utils_file.logging_limit_print(
        #         f'encoder out shape: {encoder_out.shape},encoder的第一帧的前20个数字：\n{encoder_out[0][0][:20]}')

        # utils_file.logging_limit_print(
        #     'get_embedding_from_wav(): speech_embeds.shape,by  self.encoder(wavs, wavs_len):',
        #     encoder_out.shape)

        speech_embeds, encoder_mask = self.down_sample_2(encoder_out, encoder_mask)
        utils_file.logging_limit_print(f'speech_embeds.shape:{speech_embeds.shape},encoder_mask:{encoder_mask.shape}')
        speech_embeds = torch.cat((ssl_vecs, speech_embeds), dim=1)
        utils_file.logging_limit_print(f'after cat speech_embeds.shape:{speech_embeds.shape}')
        # 创建新的 speech_masks
        ssl_mask = torch.ones((ssl_vecs.shape[0], 1, 10), dtype=torch.bool, device=speech_embeds.device)
        encoder_mask = torch.cat((ssl_mask, encoder_mask), dim=2)
        utils_file.logging_limit_print(f'after cat encoder_mask.shape:{encoder_mask.shape}')
        
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

    def get_embedding_from_text(self, text):
        text_id = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False
        ).to(
            self.embed_tokens.weight.device).input_ids
        text_embeds = self.embed_tokens(text_id)
        return text_embeds

    def get_embeds_from_wav_path(self, wav_path):
        wav_i2_path = wav_path
        # utils_file.logging_limit_print('get_embeds_from_wav_path(): wav_i2_path:', wav_i2_path)
        waveform_i2, _ = torchaudio.load(wav_i2_path)
        # utils_file.logging_limit_print('get_embeds_from_wav_path(): waveform_i2.shape:', waveform_i2.shape)
        if len(waveform_i2.shape) != 1:
            waveform_i2 = waveform_i2[0]
        waveform_i2 = waveform_i2.to(self.embed_tokens.weight.device)
        wavs_len_i2 = torch.tensor([len(waveform_i2)], device=self.embed_tokens.weight.device, dtype=torch.int32)
        wavs_i2 = waveform_i2.unsqueeze(0)
        sample_i2_embeds = self.get_embedding_from_wav(wavs_i2, wavs_len_i2)
        # utils_file.logging_limit_print('get_embeds_from_wav_path(): sample_i2_embeds.shape:', sample_i2_embeds.shape)
        return sample_i2_embeds

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

    def infer_for_speech2text_token(  # speech2text-token
            self,
            wavs,
            wavs_len,
            prompt,
            text=None,
    ):
        if text is not None:
            prompt = torch.cat((prompt, text), dim=1)
        speech_embeds, speech_masks = self.get_embedding_from_wav(wavs, wavs_len)
        speech_embeds, speech_masks, _ = self._add_bos_eos(0 + self.speech_token_num, None,
                                                           speech_embeds, speech_masks, None)
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

        max_len = 300
        hyps = torch.ones([1, 1], dtype=torch.int64,
                          device=device).fill_(1 + self.speech_token_num)  # (B*N, 1)
        llm_out = self.llama_model(
            inputs_embeds=embeds,
            past_key_values=None,
            output_hidden_states=True
        )
        cache = llm_out.past_key_values
        utils_file.logging_limit_print('得到首个cache,开始进行for循环推理')
        token_emb = self.speech_token_emded(hyps[:, -1:])

        for i in range(max_len):
            llm_out = self.llama_model(
                inputs_embeds=token_emb,
                past_key_values=cache,
                output_hidden_states=True
            )
            cache = llm_out.past_key_values
            hidden_states = llm_out.hidden_states[-1]  # 最后一层的
            token_logits_1 = self.lm_head(hidden_states)
            # utils_file.logging_limit_print(f'token_logits_1.shape:{token_logits_1.shape}')
            token_logits_2 = self.speaker_head(hidden_states)
            # utils_file.logging_limit_print(f'token_logits_2.shape:{token_logits_2.shape}')
            big_logits = torch.cat([token_logits_1, token_logits_2], dim=-1)
            # utils_file.logging_limit_print(f'big_logits.shape:{big_logits.shape}')
            logp = torch.nn.functional.log_softmax(big_logits[:, -1, :], dim=-1)  # 取了最后一个
            # utils_file.logging_limit_print(f'logp.shape:{logp.shape}')
            max_index = torch.argmax(logp, dim=-1, keepdim=True)
            # utils_file.logging_limit_print(f'max_index.shape:{max_index.shape}')
            utils_file.logging_limit_print(f'max_index:{max_index}')

            hyps = torch.cat((hyps, max_index),
                             dim=1)  # (B*N, i+1)
            if max_index < 152064:
                token_emb = self.embed_tokens(hyps[:, -1:])
            else:
                if max_index == 152064 + 4096:
                    utils_file.logging_limit_print(f'耿雪龙 遇到token结束符号，结束')
                    break
                token_emb = self.speech_token_emded(hyps[:, -1:])
        best_hyps = hyps[0, :]
        text_res = []
        token_res = []
        for i in best_hyps[1:]:
            if i < 152064:
                text_res.append(i)
            else:
                token_res.append(str((i - 152064).item()))
        str_i = self.tokenizer.decode(text_res, skip_special_tokens=True, add_special_tokens=False)
        return [str_i + " | " + " ".join(token_res)]
        # output_text = self.tokenizer.batch_decode(outputs, add_special_tokens=False, skip_special_tokens=True)

    def infer_for_text2token(  # text2token
            self,
            wavs,
            wavs_len,
            prompt,
            text=None,
    ):
        if text is not None:
            prompt = torch.cat((prompt, text), dim=1)
        # speech_embeds, speech_masks = self.get_embedding_from_wav(wavs, wavs_len)
        # speech_embeds, speech_masks, _ = self._add_bos_eos(0 + self.speech_token_num, None,
        #                                                    speech_embeds, speech_masks, None)
        labels_lengths = torch.tensor([len(text)-1], dtype=torch.int64)
        labels = text[:,:-1]
        labels_pad_mask = make_pad_mask(labels_lengths)  # B, L
        labels = labels.masked_fill(labels_pad_mask, 0)
        speech_embeds = self.embed_tokens(labels)  # B, L, D
        speech_target = torch.full(labels_pad_mask.shape, self.IGNORE_ID).to(
            speech_embeds.device)
        speech_masks = ~labels_pad_mask

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

        max_len = 300
        hyps = torch.ones([1, 1], dtype=torch.int64,
                          device=device).fill_()  # (B*N, 1)
        llm_out = self.llama_model(
            inputs_embeds=embeds,
            past_key_values=None,
            output_hidden_states=True
        )
        cache = llm_out.past_key_values
        utils_file.logging_limit_print('得到首个cache,开始进行for循环推理')
        token_emb = self.embed_tokens(hyps[:, -1:])

        for i in range(max_len):
            llm_out = self.llama_model(
                inputs_embeds=token_emb,
                past_key_values=cache,
                output_hidden_states=True
            )
            cache = llm_out.past_key_values
            hidden_states = llm_out.hidden_states[-1]  # 最后一层的
            token_logits_1 = self.lm_head(hidden_states)
            # utils_file.logging_limit_print(f'token_logits_1.shape:{token_logits_1.shape}')
            token_logits_2 = self.speaker_head(hidden_states)
            # utils_file.logging_limit_print(f'token_logits_2.shape:{token_logits_2.shape}')
            big_logits = torch.cat([token_logits_1, token_logits_2], dim=-1)
            # utils_file.logging_limit_print(f'big_logits.shape:{big_logits.shape}')
            logp = torch.nn.functional.log_softmax(big_logits[:, -1, :], dim=-1)  # 取了最后一个
            # utils_file.logging_limit_print(f'logp.shape:{logp.shape}')
            max_index = torch.argmax(logp, dim=-1, keepdim=True)
            # utils_file.logging_limit_print(f'max_index.shape:{max_index.shape}')
            utils_file.logging_limit_print(f'max_index:{max_index}')

            hyps = torch.cat((hyps, max_index),
                             dim=1)  # (B*N, i+1)
            if max_index < 152064:
                token_emb = self.embed_tokens(hyps[:, -1:])
            else:
                if max_index == 152064 + 4096 :
                    utils_file.logging_limit_print(f'耿雪龙 遇到token结束符号，结束')
                    break
                token_emb = self.speech_token_emded(hyps[:, -1:])
        best_hyps = hyps[0, :]
        text_res = []
        token_res = []
        for i in best_hyps[1:]:
            if i < 152064:
                text_res.append(i)
            else:
                token_res.append(str((i - 152064).item()))
        str_i = self.tokenizer.decode(text_res, skip_special_tokens=True, add_special_tokens=False)
        return [str_i + " | " + " ".join(token_res)]
        # output_text = self.tokenizer.batch_decode(outputs, add_special_tokens=False, skip_special_tokens=True)

import math
import torch
import torch.nn as nn
from torch import Tensor
import torch_npu
import torch_npu.dynamo.torchair as tng
from typing import List, Optional, Tuple, Union
import transformers.models
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2MLP, 
    Qwen2RMSNorm,
    Qwen2RotaryEmbedding,
    Qwen2ForCausalLM,
    repeat_kv,
    _prepare_4d_causal_attention_mask_with_cache_position
)

from transformers.utils import logging
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import NEED_SETUP_CACHE_CLASSES_MAPPING
from transformers.cache_utils import Cache, StaticCache
from transformers.generation.logits_process import TopPLogitsWarper
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from .utils import InferTaskCode

logger = logging.get_logger(__name__)

_NPU_QWEN_TORCH_COMPILE = True # 是否使用NPU编译Qwen2模型
tng_config = tng.CompilerConfig()
# tng_config.fusion_config.fusion_switch_file="/mnt/sfs/zy/projs/whisper/patches/fusion_switch.cfg"
# tng_config.dump_config.enable_dump = True
# tng_config.dump_config.dump_mode = "all"
# tng_config.dump_config.dump_path = 'output/dump'

class DebugHelper:
    _ZY_DEUBG_INT = 0

class NPUQwen2RMSNorm(Qwen2RMSNorm):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__(hidden_size, eps)

    def forward(self, hidden_states):
        """
        https://www.hiascend.com/document/detail/zh/canncommercial/700/modeldevpt/ptmigr/ptaoplist_000431.html
        deepspeed will convert all weights to BF16 ......, use float32 in norm layers
        """
        return torch_npu.npu_rms_norm(hidden_states.float(), self.weight.float(), self.variance_epsilon)[0].bfloat16()

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"

# ===================================================================
# =============================Attention=============================
# ===================================================================
class NPUQwen2Attention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """
    def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.pack_qkv = False
        if self.pack_qkv:
            self.w_pack = nn.Linear(self.hidden_size,
                                    self.num_heads * self.head_dim + 2 * self.num_key_value_heads * self.head_dim,
                                    bias=True)
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
            self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
            self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self.rotary_emb = Qwen2RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )
        self.softmax_scale = 1 / math.sqrt(self.head_dim)
    
    def npu_apply_rotary_pos_emb(self, q, k, cos, sin, position_ids, unsqueeze_dim=1):
        """
        https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/apiref/apilist/ptaoplist_000152.html
        TODO: BF16情况下算子与小算子精度有差异，fp32下没有
        """
        cos = cos[position_ids].unsqueeze(unsqueeze_dim) # B,1,S,D
        sin = sin[position_ids].unsqueeze(unsqueeze_dim) # B,1,S,D
        q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
        k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
        return q_embed, k_embed

    def npu_flash_attention(self,
                            query_states: Tensor,
                            key_states: Tensor,
                            value_states: Tensor,
                            attention_mask: Tensor=None,
                            use_cache: bool = False) -> Tuple[Tensor, Optional[Tensor]]:
        """
        https://www.hiascend.com/document/detail/zh/canncommercial/700/modeldevpt/ptmigr/ptaoplist_000691.html
        Args:
            query_states (`torch.Tensor`): BN1SD,( 17,28,33,128)
            key_states (`torch.Tensor`): BN2SD, (17,4,33,128)
            value_states (`torch.Tensor`): BN2SD, (17,4,33,128)
            attention_mask (`torch.Tensor`): B1SS, (17,1,33,33)
        Return:
            attn_output (`torch.Tensor`): BN1SD, (17,28,33,128)
            attn_weights (`torch.Tensor`): BN1SS, (17,28,33,33)
        """
        attention_mask[:, :, :, : key_states.shape[-2]]
        if use_cache:   # inference mode
            if query_states.size(-2) == 1 and query_states.size(-2) != key_states.size(-2):
                # decoding
                attn_output = torch_npu.npu_incre_flash_attention(
                    query_states,
                    key_states,
                    value_states,
                    atten_mask=attention_mask.bool(),
                    num_heads=query_states.size(1),
                    input_layout="BNSD",
                    pse_shift=None,
                    padding_mask=None,
                    scale_value=self.softmax_scale
                )
            else:
                # prefill
                attn_output = torch_npu.npu_prompt_flash_attention(
                    query_states,
                    key_states,
                    value_states,
                    num_heads=query_states.size(1),
                    input_layout="BNSD",
                    pse_shift=None,
                    sparse_mode=0,
                    padding_mask=None,
                    atten_mask=attention_mask.bool(),
                    scale_value=self.softmax_scale,
                    pre_tokens=65536,
                    next_tokens=0
                )
        else:           # train mode
            attn_output = torch_npu.npu_fusion_attention(
                query_states,
                key_states,
                value_states,
                query_states.shape[1],
                "BNSD",
                padding_mask=None,
                atten_mask=attention_mask.bool(),
                scale=self.softmax_scale,
                pre_tockens=65536,
                next_tockens=0,
                keep_prob=1 - self.attention_dropout,
                inner_precise=0,
                sparse_mode=0
            )[0]
        return attn_output, None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        if self.pack_qkv:
            qkv = self.w_pack(hidden_states)
            qkv = qkv.view(bsz, q_len, self.num_heads + 2*self.num_key_value_heads, self.head_dim).transpose(1, 2)
            query_states, key_states, value_states = torch.split(qkv,
                                                                [self.num_heads,
                                                                self.num_key_value_heads,
                                                                self.num_key_value_heads],
                                                                dim=1)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # NOTE: RoPE return all embedding (to satisfy torch compile)
        cos, sin = self.rotary_emb(value_states, seq_len=past_key_value.get_max_length())
        query_states, key_states = self.npu_apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)   # replace RoPE

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # NOTE: when it is training, npu do not need repeat, fusion attention support GQA;
        # while in infer case, we need to repeat this
        if not self.training and self.num_key_value_heads < self.num_heads:
            # repeat k/v heads if n_kv_heads < n_heads
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        # q BNSD, (17,28,33,128); k BNSD, (17,4,33,128); v BNSD, (17,4,33,128); attention_mask B1SS, (17,1,33,33)
        attn_output, attn_weights = self.npu_flash_attention(query_states, key_states, value_states, attention_mask, use_cache)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        if q_len > 1:
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        else:
            # NOTE: 避免触发batch matmul以及由此带来的大量weight复制
            attn_output = attn_output.reshape(bsz, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if q_len == 1:
            attn_output = attn_output.view(bsz, 1, -1)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

# ===================================================================
# =============================Layer=================================
# ===================================================================
class NPUQwen2DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        if config.sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )
        self.self_attn = NPUQwen2Attention(config, layer_idx)                                           # replace attention

        self.mlp = Qwen2MLP(config)
        self.input_layernorm = NPUQwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)             # replace rmsmorn
        self.post_attention_layernorm = NPUQwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)    # replace rmsmorn

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

# ===================================================================
# ========================Qwen2ForCausalLM===========================
# ===================================================================
class InferQwen2ForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        # self.may_compile_forward = tng.torchair.inference.cache_compile(self._forward, ge_cache=True, dynamic=False, fullgraph=True) \
        #     if _NPU_QWEN_TORCH_COMPILE else self._forward
        self.may_compile_forward = torch.compile(self._forward, backend=tng.get_npu_backend(compiler_config=tng_config),
                                                 dynamic=False, fullgraph=True) \
            if _NPU_QWEN_TORCH_COMPILE else self._forward
        self.text_phase = True
        self.speech_repetition_penalty = None
    '''
    NOTE: 重写原Qwen2ForCausalLM forward函数，torchair直接编译原函数在返回CausalLMOutputWithPast时会出现编译错误
    '''
    def _forward(self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        return outputs
    
    def forward(self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            do_compile = True
        ) -> Union[Tuple, CausalLMOutputWithPast]:
        if past_key_values is not None:
            past_key_values.training = False
        # print(self.text_phase)
        if input_ids is not None:
            if self.text_phase:
                inputs_embeds = self.model.embed_tokens(input_ids)
            else:
                inputs_embeds = self.speech_token_emded(input_ids)
            if torch.isin(input_ids, 151645).any():
                self.text_phase = False
                if self.speech_repetition_penalty is not None:
                    self.speech_repetition_penalty.speech_phase = True
            input_ids = None
            
        if len(cache_position) > 1 or do_compile==False :
            # prefill branch
            outputs = self._forward(input_ids,
                            attention_mask,
                            position_ids,
                            past_key_values,
                            inputs_embeds,
                            labels,
                            use_cache,
                            output_attentions,
                            output_hidden_states,
                            return_dict,
                            cache_position)
        else:
            # def print_input(input, input_name):
            #     if input is None:
            #         print(f"{input_name} is None")
            #         return
            #     if isinstance(input, torch.Tensor):
            #         print(f"{input_name} has shape: {input.shape}, dtype: {input.dtype}, device: {input.device}")
            #     else:
            #         print(f"input name: {input}")
                    
            # decoding, only decoding branch 
            outputs = self.may_compile_forward(input_ids,
                            attention_mask,
                            position_ids,
                            past_key_values,
                            inputs_embeds,
                            labels,
                            use_cache,
                            output_attentions,
                            output_hidden_states,
                            return_dict,
                            cache_position)
        
        
        last_hidden_states = outputs.last_hidden_state
        
        if self.text_phase:
            logits = self.lm_head(last_hidden_states)
        else:
            logits = self.speech_head(last_hidden_states)
        
        logits = logits.float()

        return CausalLMOutputWithPast(
            loss=None,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        """
        Mainly add static cache support
        """
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        # if attention_mask is not None and position_ids is None:
        #     # create position_ids on the fly for batch generation
        #     position_ids = attention_mask.long().cumsum(-1) - 1
        #     position_ids.masked_fill_(attention_mask == 0, 1)
        #     if past_key_values:
        #         position_ids = position_ids[:, -input_ids.shape[1] :]
        #         # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`,
        #         # as otherwise the input `position_ids` would have various stride during the decoding.
        #         # Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case,
        #         # `position_ids` is already contiguous but with varying stride which retriggers a capture.
        #         position_ids = position_ids.clone(memory_format=torch.contiguous_format)
        
        position_ids = cache_position.unsqueeze(0)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and len(cache_position) > 1:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # NOTE: 与上述的position_ids相同，same as position_ids, for torch.compile and cuda graph
            input_ids = input_ids.clone(memory_format=torch.contiguous_format)
            model_inputs = {"input_ids": input_ids}
        # print("token", model_inputs)
        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if inputs_embeds is not None and len(cache_position) > 1:
                # prefill phase, inputs_embeds has shape (B,S,H)
                batch_size, sequence_length = inputs_embeds.shape[0], inputs_embeds.shape[1]
                device = inputs_embeds.device
            else:
                # decdoing phase, input_ids has shape (B,S)
                batch_size, sequence_length = input_ids.shape
                device = input_ids.device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            if inputs_embeds is not None and inputs_embeds.ndim == 2 or input_ids is not None and input_ids.size(-1) == 1:
                # we only expand attention mask in docoding mode
                attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                    attention_mask,
                    sequence_length=sequence_length,
                    target_length=past_key_values.get_max_length(),
                    dtype=dtype,
                    device=device,
                    min_dtype=min_dtype,
                    cache_position=cache_position,
                    batch_size=batch_size,
                )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
                "do_compile": kwargs['do_compile'],
            }
        )
        return model_inputs

# ===================================================================
# ==========================StaticCache==============================
# ===================================================================

class NPUStaticCache(StaticCache):
    def __init__(self, config, max_batch_size, max_cache_len, device, dtype=None):
        super().__init__(config, max_batch_size, max_cache_len, device, dtype)
    
    def update(self, key_states, value_states, layer_idx, cache_kwargs = None) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_position = cache_kwargs.get("cache_position")
        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]

        if cache_position is None:
            k_out.copy_(key_states)
            v_out.copy_(value_states)
        else:
            # NPU torchair do not support index_copy or index_put right now.
            # try to bypass this operator
            if len(cache_position) > 1:
                # prefill
                self.key_cache[layer_idx].copy_(k_out.index_add(2, cache_position, key_states))
                self.value_cache[layer_idx].copy_(v_out.index_add(2, cache_position, value_states))
            else:
                # NOTE: scatter_update_ is more efficient on NPU device
                tmp_ids = cache_position.expand(k_out.size(0))
                torch_npu.scatter_update_(k_out, tmp_ids, key_states, 2)
                torch_npu.scatter_update_(v_out, tmp_ids, value_states, 2)

        return k_out, v_out
    
    def reorder_cache(self, beam_idx):
        """Reorders the cache for beam search, given the selected beam indices.
        NOTE: torch compile适配版本，原版本没有采用inplace操作，修改未作用于图输入，会导致KV cache混乱
        `self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))`
        将原本的`=`操作替换为inplace copy
        """
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx].copy_(self.key_cache[layer_idx].index_select(0, beam_idx.to(device)))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx].copy_(self.value_cache[layer_idx].index_select(0, beam_idx.to(device)))
        
    # def reset(self):
    #     """Resets the cache values while preserving the objects.
    #     torchair暂未实现inplace版本zero_，调用zero_返回新对象，采用inplace copy保证修改到原地址
    #     """
    #     for layer_idx in range(len(self.key_cache)):
    #         # In-place ops prevent breaking the static address
    #         self.key_cache[layer_idx].copy_(self.key_cache[layer_idx].zero_())
    #         self.value_cache[layer_idx].copy_(self.value_cache[layer_idx].zero_())

# ===================================================================
# ===============================TopP================================
# ===================================================================
class SpeculationTopPLogitsWarper(TopPLogitsWarper):
    """
    SpeculationTopPLogitsWarper accomplishes the same function as TopPLogitsWarper, but uses speculative calculation
    methods to try to reduce calculation time. After sorting, it tries to get the result from the N tokens with the
    highest probability. If it fails, N is increased, until reaches the vocabulary size. So its performence highly depends
    the number of logits reserved (the less the better) and `note: sometimes the performence may not outperform TopPLogitsWarper`
    Args:
        see TopPLogitsWarper

    Examples:
        see TopPLogitsWarper
    """
    def __init__(self, top_p, filter_value = ..., min_tokens_to_keep = 1):
        super().__init__(top_p, filter_value, min_tokens_to_keep)
        self.voc_length = 0
        self.stairs = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores_shape = scores.shape
        scores = scores.view(-1, scores_shape[-1])
        bs, length = scores.shape
        begins = (torch.arange(bs, device=scores.device) * length).view(bs, 1)
        if self.voc_length != length:
            self._reset_stairs(length)
        # set `descending=False`` to make sure the bounding index is reserved.
        # Example:
        # with the prob [0.4, 0.3, 0.2, 0.1] and the p 0.5:
        # when cumsum use `descending=False`, we'll get [0.4, 0.7, 0.2, 0.1],
        # then we judge by cumsum value <(or <=) 0.5, the boundary value 0.3
        # will be excluded, which is wrong.
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        probs = sorted_logits.softmax(dim=-1)
        for stair in self.stairs:
            s_probs = probs[..., -stair:]
            s_indices = sorted_indices[..., -stair:]
            cumulative_probs = s_probs.cumsum(dim=-1)
            if (cumulative_probs[..., -1] < self.top_p).any():
                continue

            pos_to_remain = cumulative_probs > (cumulative_probs[..., [-1]] - self.top_p)
            pos_to_remain[..., -self.min_tokens_to_keep:] = 1
            pos_to_remain = pos_to_remain.view(-1).bool()
            
            indices_to_remain = (s_indices + begins).view(-1)
            indices_to_remain = indices_to_remain[pos_to_remain]

            scores = scores.view(-1)
            logits_to_remain = scores[indices_to_remain].clone()
            scores_processed = torch.full_like(scores, self.filter_value, device=scores.device)
            scores_processed[indices_to_remain] = logits_to_remain

            return scores_processed.view(scores_shape)
    
    def _reset_stairs(self, length: int):
        self.voc_length = length
        self.stairs = [self.min_tokens_to_keep]
        stair = 1
        while stair < length:
            if stair > self.min_tokens_to_keep:
                self.stairs.append(stair)
            stair *= 8
        self.stairs.append(length)

# ===================================================================
# =============================DO PATCH==============================
# ===================================================================
print("=========================替换================================")
transformers.models.qwen2.modeling_qwen2.Qwen2PreTrainedModel._supports_static_cache = True # enable static cache
transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer = NPUQwen2DecoderLayer
transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM = InferQwen2ForCausalLM
# TODO 动态HOOK，当前TopPLogitsWarper模块patch失败，找不到原因需要到transformers库中替换
# transformers.generation.logits_process.TopPLogitsWarper = SpeculationTopPLogitsWarper
# TODO 动态HOOK，当前StaticCache模块patch失败，找不到原因
NEED_SETUP_CACHE_CLASSES_MAPPING["static"] = NPUStaticCache
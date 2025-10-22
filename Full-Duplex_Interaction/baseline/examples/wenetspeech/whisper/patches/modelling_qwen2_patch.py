# patch model qwen2
# 主要是融合算子替换，训练推理：FA RoPE RMSNorm LoRA MatulAdd
# 是否能把LoRA的QKV合并成一个
# 方案 替换Qwen的 Qwen2DecoderLayer
import math
import torch
import torch.nn as nn
from torch import Tensor
import torch_npu

from typing import List, Optional, Tuple, Union

import transformers
import transformers.models
from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm, Qwen2MLP, Qwen2Attention, Qwen2DecoderLayer, Qwen2RotaryEmbedding, repeat_kv, QWEN2_ATTENTION_CLASSES

from transformers.utils import logging
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

logger = logging.get_logger(__name__)

class NPUQwen2RMSNorm(Qwen2RMSNorm):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__(hidden_size, eps)

    def forward(self, hidden_states):
        """
        https://www.hiascend.com/document/detail/zh/canncommercial/700/modeldevpt/ptmigr/ptaoplist_000431.html
        deepspeed will convert all weights to BF16 ......
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
        self.w_pack = nn.Linear(self.hidden_size,
                                self.num_heads * self.head_dim + 2 * self.num_key_value_heads * self.head_dim,
                                bias=True)
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
                            attention_mask: Tensor=None):
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
        # TODO add inference mode; add slide window
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

        qkv = self.w_pack(hidden_states)
        qkv = qkv.view(bsz, q_len, self.num_heads + 2*self.num_key_value_heads, self.head_dim).transpose(1, 2)
        query_states, key_states, value_states = torch.split(qkv,
                                                             [self.num_heads,
                                                            self.num_key_value_heads,
                                                            self.num_key_value_heads],
                                                            dim=1)
        
        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = self.npu_apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)   # replace RoPE

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # when it is training, npu do not need repeat, fusion attention support GQA;
        # while in infer case, we need to repeat this
        if not self.training and self.num_key_value_heads < self.num_heads:
            # repeat k/v heads if n_kv_heads < n_heads
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

        # q BNSD, (17,28,33,128); k BNSD, (17,4,33,128); v BNSD, (17,4,33,128); attention_mask B1SS, (17,1,33,33)
        attn_output, attn_weights = self.npu_flash_attention(query_states, key_states, value_states, attention_mask)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

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
        # self.self_attn = QWEN2_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)
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
# =============================DO PATCH==============================
# ===================================================================
transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer = NPUQwen2DecoderLayer
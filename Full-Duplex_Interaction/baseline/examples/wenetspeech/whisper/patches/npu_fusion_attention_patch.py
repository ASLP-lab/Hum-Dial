import math
from typing import Optional, Tuple

import torch
from torch import nn
import torch_npu
from wenet.transformer.attention import MultiHeadedAttention

T_CACHE = Tuple[torch.Tensor, torch.Tensor]

class NPUFusionSelfAttention(MultiHeadedAttention):
    def __init__(self,
                 n_head: int,
                 n_feat: int,
                 dropout_rate: float,
                 query_bias: bool = True,
                 key_bias: bool = True,
                 value_bias: bool = True,
                 use_sdpa: bool = False,
                 n_kv_head: Optional[int] = None,
                 head_dim: Optional[int] = None):
        super().__init__(n_head, n_feat, dropout_rate, query_bias, key_bias,
                         value_bias, use_sdpa, n_kv_head, head_dim)
        self.softmax_scale = 1 / math.sqrt(self.d_k)
    
    # def forward_qkv(
    #     self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    #     """Transform query, key and value.

    #     Args:
    #         query (torch.Tensor): Query tensor (#batch, ..., time1, size).
    #         key (torch.Tensor): Key tensor (#batch, ..., time2, size).
    #         value (torch.Tensor): Value tensor (#batch, ..., time2, size).

    #     Returns:
    #         torch.Tensor: Transformed query tensor, size
    #             (#batch, ..., n_head, time1, d_k).
    #         torch.Tensor: Transformed key tensor, size
    #             (#batch, ..., n_head_kv, time2, d_k).
    #         torch.Tensor: Transformed value tensor, size
    #             (#batch, ..., n_head_kv, time2, d_k).

    #     """
    #     q = self._forward_linearx('query', query, head_first=False)
    #     k = self._forward_linearx('key', key, head_first=False)
    #     v = self._forward_linearx('value', value, head_first=False)
    #     return q, k, v
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        pos_emb: torch.Tensor = torch.empty(0),
        cache: T_CACHE = (torch.zeros(0, 0, 0, 0), torch.zeros(0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, T_CACHE]:
        """
        Args:
            query (torch.Tensor): Query tensor (#batch, ..., time1, size).
            key (torch.Tensor): Key tensor (#batch, ..., time2, size).
            value (torch.Tensor): Value tensor (#batch, ..., time2, size).
            mask (torch.Tensor): Mask, size (#batch, 1, time2) or
                (#batch, ..., time1, time2), (0, ..., 0, 0) means fake mask.
            cache (torch.Tensor): Cache tensor (1, cache_t, head, d_k * 2),
                where `cache_t == chunk_size * num_decoding_left_chunks`
                and `head * d_k == size`
        """
        # 注意这个k没有bias
        q, k, v = self.forward_qkv(query, key, value)   # B,N,S,D
        k, v, new_cache = self._update_kv_and_cache(k, v, cache)
        atten_mask = ~mask.bool().unsqueeze(1)          # B, 1, 1, S
        atten_mask = torch.repeat_interleave(atten_mask, repeats=mask.size(-1), dim=2)
        # TODO add inference mode; add slide window; 这里没有casual吗？
        output = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            self.h,
            "BNSD",
            atten_mask=atten_mask,
            scale=self.softmax_scale,
            # pre_tockens=65536,
            # next_tockens=0,
            keep_prob=1 - self.dropout_rate,
            inner_precise=0,
            sparse_mode=0
        )[0]
        
        output = output.transpose(1,2) # B,S,N,H
        output = output.reshape(*output.shape[:2], -1)
        return self.linear_out(output), new_cache

# class NPUFusionSelfAttention(nn.Module):
#     def __init__(self,
#                  n_head: int,
#                  n_feat: int,
#                  dropout_rate: float,
#                  query_bias: bool = True,
#                  key_bias: bool = True,
#                  value_bias: bool = True,
#                  use_sdpa: bool = False,
#                  n_kv_head: Optional[int] = None,
#                  head_dim: Optional[int] = None):
#         super().__init__()

#         self.inner_dim = n_feat if head_dim is None else head_dim * n_head
#         if n_kv_head is not None:
#             assert head_dim is not None
#             self.inner_kv_dim = head_dim * n_kv_head
#             n_kv_head = n_kv_head
#         else:
#             self.inner_kv_dim = self.inner_dim
#             n_kv_head = n_head
#         # We assume d_v always equals d_k
#         self.d_k = self.inner_dim // n_head
#         assert self.d_k == self.inner_kv_dim // n_kv_head
#         self.h = n_head
#         self.h_kv = n_kv_head

#         self.linear_qkv = nn.Linear(n_feat, self.inner_dim + 2*self.inner_kv_dim, bias=query_bias)
#         self.linear_out = nn.Linear(self.inner_dim, n_feat, bias=query_bias)
#         self.dropout = nn.Dropout(p=dropout_rate)

#         self.use_sdpa = use_sdpa
#         self.dropout_rate = dropout_rate

#         self.softmax_scale = 1 / math.sqrt(self.d_k)
    
#     def _update_kv_and_cache(
#             self,
#             k: torch.Tensor,
#             v: torch.Tensor,
#             cache: T_CACHE,
#             head_first: bool = True
#     ) -> Tuple[torch.Tensor, torch.Tensor, T_CACHE]:
#         new_cache = cache
#         seq_axis = -2 if head_first else -3
#         head_axis = -3 if head_first else -2
#         if not self.training:
#             key_cache, value_cache = cache
#             if key_cache.size(0) > 0:
#                 k = torch.cat([key_cache, k], dim=seq_axis)
#             if value_cache.size(0) > 0:
#                 v = torch.cat([value_cache, v], dim=seq_axis)
#             # NOTE(xcsong): We do cache slicing in encoder.forward_chunk, since it's
#             #   non-trivial to calculate `next_cache_start` here.
#             # new_cache = torch.cat((k, v), dim=-1) if not self.training else cache
#             new_cache = (k, v)
#         # for multi query or multi group attention
#         if self.h_kv != self.h and self.h_kv != 1:
#             n_repeat = self.h // self.h_kv
#             k_shape = k.size()
#             repeat_axis = head_axis + 1
#             k = k.unsqueeze(head_axis).expand(
#                 k_shape[:repeat_axis] + torch.Size([n_repeat]) +
#                 k_shape[repeat_axis:]).reshape(
#                     k_shape[:head_axis] + torch.Size([self.h_kv * n_repeat]) +
#                     k_shape[repeat_axis:])
#             v_shape = v.size()
#             v = v.unsqueeze(head_axis).expand(
#                 v_shape[:repeat_axis] + torch.Size([n_repeat]) +
#                 v_shape[(repeat_axis):]).reshape(
#                     v_shape[:head_axis] + torch.Size([self.h_kv * n_repeat]) +
#                     v_shape[repeat_axis:])

#         return k, v, new_cache
    
#     def forward(
#         self,
#         query: torch.Tensor,
#         key: torch.Tensor,
#         value: torch.Tensor,
#         mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
#         pos_emb: torch.Tensor = torch.empty(0),
#         cache: T_CACHE = (torch.zeros(0, 0, 0, 0), torch.zeros(0, 0, 0, 0)),
#     ) -> Tuple[torch.Tensor, T_CACHE]:
#         """
#         Args:
#             query (torch.Tensor): Query tensor (#batch, ..., time1, size).
#             key (torch.Tensor): Key tensor (#batch, ..., time2, size).
#             value (torch.Tensor): Value tensor (#batch, ..., time2, size).
#             mask (torch.Tensor): Mask, size (#batch, 1, time2) or
#                 (#batch, ..., time1, time2), (0, ..., 0, 0) means fake mask.
#             cache (torch.Tensor): Cache tensor (1, cache_t, head, d_k * 2),
#                 where `cache_t == chunk_size * num_decoding_left_chunks`
#                 and `head * d_k == size`
#         """
#         # only support query == key == value
#         qkv = self.linear_qkv(query).view(*query.shape[:2], -1, self.d_k) # B,S,N,D
#         q, k, v = torch.split(qkv, [self.h, self.h_kv, self.h_kv,], dim=2)

#         k, v, new_cache = self._update_kv_and_cache(k, v, cache, head_first=False)
#         atten_mask = ~mask.bool().unsqueeze(1)          # B, 1, 1, S
#         atten_mask = torch.repeat_interleave(atten_mask, repeats=mask.size(-1), dim=2)
#         # TODO add inference mode; add slide window
#         output = torch_npu.npu_fusion_attention(
#             q,
#             k,
#             v,
#             q.shape[2],
#             "BSND",
#             atten_mask=atten_mask,
#             scale=self.softmax_scale,
#             # pre_tockens=65536,
#             # next_tockens=65536,
#             keep_prob=1 - self.dropout_rate,
#             inner_precise=0,
#             sparse_mode=0
#         )[0]
#         output = output.view(*output.shape[:-2], -1) # B,S,H
#         return self.linear_out(output), new_cache
"""
ein notation:
b - batch
n - sequence
nt - text sequence
nw - raw wave length
d - dimension
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList, ModuleDict
from einops import rearrange
from f5_tts.model.ecapa_tdnn import ECAPA_TDNN
from x_transformers.x_transformers import RotaryEmbedding
from torch.amp import autocast
from f5_tts.model.modules import (
    TimestepEmbedding,
    ConvNeXtV2Block,
    ConvPositionEmbedding,
    DiTBlock,
    AdaLayerNorm_Final,
    # AdaLayerNormZero_Final,
    precompute_freqs_cis,
    get_pos_embed_indices,
)

def exists(val):
    return val is not None

# 假设下面这些函数/模块已在 f5_tts.model.modules 中定义
# TimestepEmbedding：对时间步进行嵌入编码
# ConvNeXtV2Block：基于 ConvNeXtV2 的卷积块，用于文本特征的局部建模
# ConvPositionEmbedding：卷积位置编码模块，为输入特征添加位置信息
# DiTBlock：Transformer 的基本块，内部包含多头注意力、前馈网络等
# AdaLayerNormZero_Final：经过自适应调制的 LayerNorm，通常结合时间条件调制
# precompute_freqs_cis：预计算正弦余弦位置编码（用于 rotary embedding）
# get_pos_embed_indices：获取位置编码索引，结合序列长度生成位置索引
# RotaryEmbedding：旋转位置编码模块，为注意力计算提供位置信息

# ---------------------------
# 文本嵌入模块
# ---------------------------
class TextEmbedding(nn.Module):
    """
    将文本 token 转换为嵌入向量，可选地添加正弦位置编码与卷积块建模
    """
    def __init__(self, text_num_embeds, text_dim, conv_layers=0, align_mode: str = "repeat"):
        """
        参数：
            text_num_embeds (int): 文本词表中 token 数量（不含 filler token）。
            text_dim (int): 每个 token 的嵌入维度。
            conv_layers (int): 卷积层数量，当 >0 时，启用额外建模。
            conv_mult (int): 卷积块中隐藏层维度的扩展倍数。
        """
        super().__init__()
        # 使用 nn.Embedding 将 token 索引映射到向量空间。
        # 注意：词表大小扩展 1，预留 0 作为 filler token（填充用）。
        self.text_embed = nn.Embedding(text_num_embeds + 1, text_dim)
        print(f"当前text_num_embeds={text_num_embeds}")
        self.align_mode = align_mode

    def forward(self, text: torch.Tensor, seq_len: int, drop_text: bool = False) -> torch.Tensor:
        # import pdb;pdb.set_trace()
        # 如果启用丢弃文本条件，则将文本全部置为 0（模拟无文本条件情况）,
        if drop_text:
            text = torch.zeros_like(text)

        # text += 1
        # text[:,-1]=4096
        # text = text[:,:-2]
        # import pdb;pdb.set_trace()
        # 2. 嵌入映射 将 token 映射到向量空间 形状从 [batch, seq_len] 转为 [batch, seq_len, text_dim]
        x = self.text_embed(text)  # [batch, token_len, embed_dim]
        
        x = torch.repeat_interleave(x, repeats=4, dim=1)
        
        # 4. 检查长度并处理
        current_len = x.shape[1]
        if current_len < seq_len:
            # 不足则填充 0
            pad_len = seq_len - current_len
            padding = torch.zeros(x.size(0), pad_len, x.size(2), 
                                dtype=x.dtype, device=x.device)
            x = torch.cat([x, padding], dim=1)  # [batch, seq_len, embed_dim]
        elif current_len > seq_len:
            # 超出则截断
            x = x[:, :seq_len, :]  # [batch, seq_len, embed_dim]
        return x

# ---------------------------
# 输入嵌入模块：将音频和文本条件融合
# ---------------------------
class InputEmbedding(nn.Module):
    """
    对输入的音频（noised input）、条件音频（masked cond audio）和文本嵌入进行融合，
    输出 token mixing embedding，供后续 Transformer 使用。
    """
    def __init__(self, mel_dim: int, text_dim: int, out_dim: int):
        """
        参数：
            mel_dim (int): 音频特征（例如 mel-spectrogram）的维度。
            text_dim (int): 文本嵌入的维度。
            out_dim (int): 融合后的输出维度，也是 Transformer 的输入维度。
        """
        super().__init__()
        # 先将三个输入（noised audio, cond audio, text_embed）在最后一维拼接，
        # 总维度为 mel_dim * 2 + text_dim，然后通过线性层投影到 out_dim(768)
        self.proj = nn.Linear(mel_dim + text_dim + 128 + 192, out_dim)
        # 利用卷积位置编码模块，为融合后的特征添加局部位置信息
        self.conv_pos_embed = ConvPositionEmbedding(dim=out_dim)
        self.spk_encoder =  ECAPA_TDNN( ###替换成mega、tts2
                                input_size=80,             # 输入特征维度
                                lin_neurons=128,           # 嵌入维度
                                activation=torch.nn.ReLU,  # 激活函数
                                channels=[512, 512, 512, 512, 1536],  # 各层通道数
                                kernel_sizes=[5, 3, 3, 3, 1],         # 卷积核大小
                                dilations=[1, 2, 3, 4, 1],            # 扩张率
                                attention_channels=128,               # 注意力通道数
                                res2net_scale=8,                      # Res2Net 规模
                                se_channels=128,                      # SE 模块通道数
                                global_context=True,                  # 是否启用全局上下文
                            )
        self.spk_encoder.eval().to(torch.bfloat16)
    def forward(
        self, 
        x: torch.Tensor,       # noised input audio, 形状 [batch, seq_len, mel_dim]
        cond: torch.Tensor,    # masked cond audio, 形状 [batch, seq_len, mel_dim]
        spk_emb: torch.Tensor,
        text_embed: torch.Tensor,  # 文本嵌入, 形状 [batch, seq_len, text_dim]
        drop_audio_cond: bool = False  # 是否丢弃音频条件，用于 cfg
    ) -> torch.Tensor:
        """
        前向传播： [b,400,80] -> [b,128] so codec ecap-tdnn, [b,128]->repeat [b,seq,128] broad cast
        - 根据 drop_audio_cond 标志决定是否丢弃条件音频信息（置零）。
        - 将 x、cond 和 text_embed 拼接后线性投影，
          然后添加卷积位置编码与原始投影结果相加，得到融合后的 embedding。
        """
        # 如果丢弃音频条件，则将 cond 全部置为 0，注意！！此时spk_emb\cond这些也要置为0！
        # import pdb;pdb.set_trace()
        # cond = cond.to()
        # print(f"当前cond.shape={cond.shape}")
        if drop_audio_cond:
            cond = torch.zeros(x.size(0), 128, device=x.device, dtype=x.dtype)
            spk_emb = torch.zeros_like(spk_emb)
            # print(f"当前cond.shape={cond.shape}")
        else:
            cond = self.spk_encoder(cond)
        
        #[bs,192]
        cond = cond.unsqueeze(1).repeat(1, x.size(1), 1)
        # print(f"repeate后对齐长度后cond.shape={cond.shape}")
        spk_emb = spk_emb.repeat(1, x.size(1), 1)
        # cond = cond[:, None, :].expand(-1, x.shape[1], -1)
        # 拼接三个输入，维度在最后一个维度上拼接：结果 shape [batch, seq_len, mel_dim*2 + text_dim]
        merged = torch.cat((x, cond, text_embed, spk_emb), dim=-1)
        # 通过线性层投影到统一的 out_dim 维度
        x = self.proj(merged)
        return x

    def fast_forward(self, x, cond, spk_emb, text_embed, text_embed_uncond):
        x = torch.cat([x,x],dim=0)
        spk = spk_emb.repeat(1, x.size(1), 1)
        spk = torch.cat([spk,torch.zeros_like(spk)],dim=0)
        cond = torch.cat([cond,torch.rand_like(cond)],dim=0)
        cond = cond.unsqueeze(1).repeat(1, x.size(1), 1)
        # cond = self.spk_encoder(cond).unsqueeze(1).repeat(1, x.size(1), 1)
        text_emb = torch.cat([text_embed,text_embed_uncond],dim=0)
        # print(x.shape,cond.shape,text_embed.shape)
        x = self.proj(torch.cat((x, cond, text_emb, spk), dim=-1))

        return x


# ---------------------------
# DiT 模块：Transformer Backbone（基于 DiTBlock 构建）
# ---------------------------
class DiT(nn.Module):
    """
    DiT 模块构建了一个基于 DiTBlock 的 Transformer 模型，
    同时融合了时间、文本和音频条件信息，并利用多种注意力 mask 策略控制局部与全局上下文。
    """
    def __init__(
        self,
        *,
        dim,               # Transformer 的输入/内部维度
        depth = 8,         # Transformer block 的层数
        heads = 8,         # 多头注意力的头数
        dim_head = 64,     # 每个注意力头的维度
        dropout = 0.1,   # dropout 概率
        ff_mult = 4,       # 前馈网络的扩展倍数（隐藏层维度 = dim * ff_mult）
        mel_dim = 80,      # 音频特征维度（例如 mel-spectrogram 的通道数）
        text_num_embeds = 6562,  # 文本嵌入的词表大小（不含 filler token）
        text_dim = None,   # 文本嵌入的维度（若为 None 则默认等于 mel_dim）
        text_mask_padding=True,
        qk_norm=None,
        conv_layers = 0,   # 是否对文本嵌入进行额外的卷积建模
        pe_attn_head=None,
        long_skip_connection = False,  # 是否使用长跳跃连接（残差拼接）
        checkpoint_activations = False,  # 是否使用 activation checkpoint 节省显存
        forward_layers=[0],           # 只有第 0 层是 forward
        backward_layers=[5, 10, 15], # 第 3, 5, 9, 15 层是 backward
    ):
        super().__init__()
        
        # 时间嵌入模块：对时间步进行嵌入，生成条件向量 t
        self.time_embed = TimestepEmbedding(dim)

        # 如果未指定文本嵌入维度，则默认与 mel_dim 相同
        if text_dim is None:
            text_dim = mel_dim
        # 文本嵌入模块：将文本 token 映射到向量空间，并可选地加入位置编码与卷积建模
        self.text_embed = TextEmbedding(text_num_embeds, text_dim, conv_layers=conv_layers, align_mode="interpolate")
        # 输入嵌入模块：融合 noised audio、条件音频与文本嵌入，生成 Transformer 的输入 embedding
        self.input_embed = InputEmbedding(mel_dim, text_dim, dim)

        # Rotary Embedding：为注意力模块提供旋转位置编码
        self.rotary_embed = RotaryEmbedding(dim_head)

        self.dim = dim
        self.depth = depth

        # **指定 forward/backward 层的索引**
        forward_layers = set(forward_layers) if forward_layers else set()
        backward_layers = set(backward_layers) if backward_layers else set()

        # **创建 Transformer Blocks**
        self.transformer_blocks = nn.ModuleList([
            DiTBlock(
                dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout,
                block_size=24,
                t_p=1 if i in backward_layers else 0,  # backward
                t_f=1 if i in forward_layers else 0    # forward
            ) for i in range(depth)
        ])
        
        # 如果启用长跳跃连接，则在 Transformer 输出后，将输入（残差）与输出拼接再融合
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None

        # 最终归一化层（结合 AdaLayerNormZero_Final，实现自适应调制）
        self.norm_out = AdaLayerNorm_Final(dim)
        # 输出投影层：将 Transformer 输出映射到目标 mel 频谱维度
        self.proj_out = nn.Linear(dim, mel_dim)

        # 是否使用 activation checkpoint 来节省显存
        self.checkpoint_activations = checkpoint_activations #False

    def ckpt_wrapper(self, module):
        """
        封装模块，使其支持 activation checkpoint 以节省显存。
        使用方式参考 https://github.com/chuanyangjin/fast-DiT/blob/main/models.py
        """
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs
        return ckpt_forward
    
    def clear_cache(self):
        self.text_cond, self.text_uncond = None, None
        
    def forward(
        self,
        x: torch.Tensor,     # noised input audio mel，形状 [batch, seq_len, mel_dim]
        cond: torch.Tensor,  # masked cond audio，形状 [batch, seq_len, mel_dim]
        spk_emb: torch.Tensor,
        text: torch.Tensor,  # 文本 token，形状 [batch, n_text]
        time: torch.Tensor,  # 时间步，形状 [batch] 或标量，会扩展为 [batch]
        drop_audio_cond,     # 是否丢弃音频条件
        drop_text,           # 是否丢弃文本条件
        mask: torch.BoolTensor | None = None , # 可选的 padding mask，形状 [batch, seq_len]
        pos: torch.Tensor = None ,
        chunk_offset: int = 0      # <--- 新增：用于指定当前 chunk 在全局序列中的起始位置
    ) -> torch.Tensor:
        """
        前向传播流程：
        1. 根据时间步生成时间嵌入 t；
        2. 利用 TextEmbedding 将文本 token 转换为文本嵌入；
        3. 利用 InputEmbedding 融合 noised audio、cond audio 与文本嵌入，生成输入 embedding；
        4. 通过 RotaryEmbedding 获取针对当前序列长度的旋转位置编码（rope）；
        5. 将输入 embedding 依次送入多组 DiTBlock（本例中应结合 block_mask、backward_mask、forward_mask），
           同时支持 activation checkpoint 节省显存；
        6. 如果启用长跳跃连接，则将原始输入与 transformer 输出拼接后融合；
        7. 最后经过归一化和投影，输出目标 mel 频谱。
        在流式 / 分块推理场景下，可以通过 chunk_offset 指定该 chunk 在全局序列中的起始位置。
        这样 self.rotary_embed 就能生成对应的旋转位置编码。
        
        """
        
        batch, seq_len = x.shape[0], x.shape[1] #seq_len为当前的长度
        
        # import pdb;pdb.set_trace()
        # 如果 time 为标量，则重复扩展为 batch 大小
        if time.ndim == 0:
            time = time.repeat(batch)
        
        # 生成时间嵌入 t
        t = self.time_embed(time)
        
        # import pdb;pdb.set_trace()
        
        # 将文本 token 经过 TextEmbedding 处理，得到文本嵌入
        text_embed = self.text_embed(text, seq_len, drop_text=drop_text)

        x = self.input_embed(x, cond, spk_emb, text_embed ,drop_audio_cond=drop_audio_cond)

        # 生成 rotary embedding 参数，依据当前序列长度生成位置编码（rope）
        rope = self.rotary_embed.forward_from_seq_len(seq_len)
        # rope =  self.rotary_embed.forward(pos)
        # 若启用长跳跃连接，则保存原始输入作为 residual（残差）
        if self.long_skip_connection is not None: #none
            residual = x

        # ---------------------------
        # Transformer Block 部分：
        # 这里的代码注释提示需要将三组 DiTBlock（transformer_block_mask、transformer_backward_mask、transformer_forward_blocks）
        # 结合使用，目前仅示例遍历某个 transformer_blocks 列表
        # 实际应用时可设计交替或者级联的使用策略，以充分利用不同注意力 mask 的优势。
        # ---------------------------
        # 假设 self.transformer_blocks 为组合后的 block 列表（例如将三组 block 按一定顺序拼接）
        # 这里为了示例，假设我们只遍历其中一组 block
        # 请根据具体需求修改为：for block in (self.transformer_block_mask + self.transformer_backward_mask + self.transformer_forward_blocks):
        # import pdb;pdb.set_trace()
        for block in self.transformer_blocks:
            if self.checkpoint_activations:
                # print("启用chekcpoiunt activateion")
                x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, t, mask, rope)
            else:
                x = block(x, t, mask=mask, rope=rope)
            # x = block(x, t, mask=mask, rope=rope)

        # 如果启用了长跳跃连接，则将 transformer 输出与原始 residual 拼接后通过线性层融合
        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        # 经过归一化层（通常结合时间条件 t 进行调制）
        x = self.norm_out(x, t)
        # 最终通过输出投影层将特征映射到目标 mel 维度
        output = self.proj_out(x) #Linear(in_features=1024, out_features=80, bias=True)

        return output

    def fast_forward(
        self,
        x: torch.Tensor,     # noised input audio mel，形状 [batch, seq_len, mel_dim]
        cond: torch.Tensor,  # masked cond audio，形状 [batch, seq_len, mel_dim]
        spk_emb: torch.Tensor,
        text: torch.Tensor,  # 文本 token，形状 [batch, n_text]
        time: torch.Tensor,  # 时间步，形状 [batch] 或标量，会扩展为 [batch]
        drop_audio_cond,     # 是否丢弃音频条件
        drop_text,           # 是否丢弃文本条件
        mask: torch.BoolTensor | None = None , # 可选的 padding mask，形状 [batch, seq_len]
        chunk_offset: int = 0      # <--- 新增：用于指定当前 chunk 在全局序列中的起始位置
    ) -> torch.Tensor:
        """
        前向传播流程：
        1. 根据时间步生成时间嵌入 t；
        2. 利用 TextEmbedding 将文本 token 转换为文本嵌入；
        3. 利用 InputEmbedding 融合 noised audio、cond audio 与文本嵌入，生成输入 embedding；
        4. 通过 RotaryEmbedding 获取针对当前序列长度的旋转位置编码（rope）；
        5. 将输入 embedding 依次送入多组 DiTBlock（本例中应结合 block_mask、backward_mask、forward_mask），
           同时支持 activation checkpoint 节省显存；
        6. 如果启用长跳跃连接，则将原始输入与 transformer 输出拼接后融合；
        7. 最后经过归一化和投影，输出目标 mel 频谱。
        在流式 / 分块推理场景下，可以通过 chunk_offset 指定该 chunk 在全局序列中的起始位置。
        这样 self.rotary_embed 就能生成对应的旋转位置编码。
        
        """
        batch, seq_len = x.shape[0]*2, x.shape[1] #seq_len为当前的长度
        
        # import pdb;pdb.set_trace()
        
        if time.ndim == 0:
            time = time.repeat(batch)
        
        # 生成时间嵌入 t
        t = self.time_embed(time) #[bs,time]
        
        # 将文本 token 经过 TextEmbedding 处理，得到文本嵌入 ，我可以在这里forward进行asr btfsemb的提取，修改
        text_embed = self.text_embed(text, seq_len, drop_text=False)
        text_embed_uncond = self.text_embed(text, seq_len, drop_text=True)
        
        # 融合 noised audio、cond audio 与文本嵌入，得到输入 embedding
        
        # x = self.input_embed(x, cond, spk_emb, text_embed ,drop_audio_cond=drop_audio_cond)
        # import pdb;pdb.set_trace()
        pos  = torch.arange(start=chunk_offset,end=chunk_offset+seq_len, device = x.device)
        x = self.input_embed.fast_forward(x, cond, spk_emb, text_embed, text_embed_uncond)
        # 生成 rotary embedding 参数，依据当前序列长度生成位置编码（rope）
        rope =  self.rotary_embed.forward(pos) #infer 时，修改，需要不是从0开始，需要跟chunk在整体序列index保持一致

        # 若启用长跳跃连接，则保存原始输入作为 residual（残差）
        if self.long_skip_connection is not None: #none
            residual = x

        # ---------------------------
        # Transformer Block 部分：
        # 这里的代码注释提示需要将三组 DiTBlock（transformer_block_mask、transformer_backward_mask、transformer_forward_blocks）
        # 结合使用，目前仅示例遍历某个 transformer_blocks 列表
        # 实际应用时可设计交替或者级联的使用策略，以充分利用不同注意力 mask 的优势。
        # ---------------------------
        # 假设 self.transformer_blocks 为组合后的 block 列表（例如将三组 block 按一定顺序拼接）
        # 这里为了示例，假设我们只遍历其中一组 block
        # 请根据具体需求修改为：for block in (self.transformer_block_mask + self.transformer_backward_mask + self.transformer_forward_blocks):
        import pdb;pdb.set_trace()
        for block in self.transformer_blocks:
            # if self.checkpoint_activations:
            #     x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), x, t, mask, rope)
            # else:
            #     x = block(x, t, mask=mask, rope=rope)
            x = block(x, t, mask=mask, rope=rope)

        # 如果启用了长跳跃连接，则将 transformer 输出与原始 residual 拼接后通过线性层融合
        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        # 经过归一化层（通常结合时间条件 t 进行调制）
        x = self.norm_out(x, t)
        # 最终通过输出投影层将特征映射到目标 mel 维度
        output = self.proj_out(x) #Linear(in_features=1024, out_features=80, bias=True)

        return output
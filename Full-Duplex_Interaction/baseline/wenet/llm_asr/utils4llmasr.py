import random
from typing import Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from wenet.utils.common import pad_list
from gxl_ai_utils.utils import utils_file


def add_sos_eos4speech_llm(ys_pad: torch.Tensor, sos: int, eos: int,
                           ignore_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Add <sos> and <eos> labels.
    为out后接一个eos. in基本保持不变

    Args:
        ys_pad (torch.Tensor): batch of padded target sequences (B, Lmax)
        sos (int): index of <sos>
        eos (int): index of <eeos>
        ignore_id (int): index of padding

    Returns:
        ys_in (torch.Tensor) : (B, Lmax)
        ys_out (torch.Tensor) : (B, Lmax + 1)

    Examples:
        >>> sos_id = 10
        >>> eos_id = 11
        >>> ignore_id = -1
        >>> ys_pad
        tensor([[ 1,  2,  3,  4,  5],
                [ 4,  5,  6, -1, -1],
                [ 7,  8,  9, -1, -1]], dtype=torch.int32)
        >>> ys_in,ys_out=add_sos_eos(ys_pad, sos_id , eos_id, ignore_id)
        >>> ys_in
        tensor([[ 1,  2,  3,  4,  5],
                [ 4,  5,  6, 11, 11],
                [ 7,  8,  9, 11, 11]])
        >>> ys_out
        tensor([[ 1,  2,  3,  4,  5, 11],
                [ 4,  5,  6, 11, -1, -1],
                [ 7,  8,  9, 11, -1, -1]])
    """
    _sos = torch.tensor([sos],
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)
    _eos = torch.tensor([eos],
                        dtype=torch.long,
                        requires_grad=False,
                        device=ys_pad.device)
    ys = [y[y != ignore_id] for y in ys_pad]  # parse padded ys
    # ys_in = [torch.cat([_sos, y], dim=0) for y in ys]
    ys_in = [y for y in ys]
    ys_out = [torch.cat([y, _eos], dim=0) for y in ys]
    return pad_list(ys_in, eos), pad_list(ys_out, ignore_id)

global_prompt_dict = None
def get_prompt_by_task(task_name):
    """
    根据task给定指定的prompt, 并实现prompt的多样随意性
    Args:
        task_name:

    Returns:

    """
    global global_prompt_dict
    if global_prompt_dict is None:
        global_prompt_dict = utils_file.load_dict_from_yaml('conf/prompt.yaml')
    random_index = random.randint(0, len(global_prompt_dict[task_name])-1)
    return global_prompt_dict[task_name][random_index]


import torch

def merge_labels_with_valid_adjacent(
    labels_embeds1, labels_target1, labels_mask1,
    labels_embeds2, labels_target2, labels_mask2,
    pad_value=0, ignore_id=-100
):
    """
    合并两组标签，有效特征紧邻拼接，无效特征后移
    Args:
        labels_embeds1 (Tensor): 标签1嵌入，形状 (B, L1, D)
        labels_target1 (Tensor): 标签1目标，形状 (B, L1)
        labels_mask1 (Tensor):  标签1掩码，形状 (B, L1)
        labels_embeds2 (Tensor): 标签2嵌入，形状 (B, L2, D)
        labels_target2 (Tensor): 标签2目标，形状 (B, L2)
        labels_mask2 (Tensor):  标签2掩码，形状 (B, L2)
        pad_value (int): 嵌入填充值
        ignore_id (int): 目标填充值（如IGNORE_ID）
    Returns:
        merged_embeds (Tensor): 合并嵌入，形状 (B, L1+L2, D)
        merged_target (Tensor): 合并目标，形状 (B, L1+L2)
        merged_mask (Tensor):  合并掩码，形状 (B, L1+L2)
    """
    batch_size = labels_embeds1.size(0)
    max_len = labels_embeds1.size(1) + labels_embeds2.size(1)
    
    merged_embeds = []
    merged_target = []
    merged_mask = []
    
    for i in range(batch_size):
        # 提取有效特征索引
        valid_indices1 = torch.where(labels_mask1[i])[0]
        valid_indices2 = torch.where(labels_mask2[i])[0]
        
        # 合并有效特征段
        valid_embeds = torch.cat([
            labels_embeds1[i, valid_indices1],
            labels_embeds2[i, valid_indices2]
        ], dim=0)
        
        valid_target = torch.cat([
            labels_target1[i, valid_indices1],
            labels_target2[i, valid_indices2]
        ], dim=0)
        
        valid_mask = torch.cat([
            labels_mask1[i, valid_indices1],
            labels_mask2[i, valid_indices2]
        ], dim=0)
        
        # 填充无效部分
        pad_length = max_len - len(valid_embeds)
        padded_embeds = torch.cat([
            valid_embeds,
            torch.full((pad_length, labels_embeds1.size(2)), pad_value, device=labels_embeds1.device)
        ], dim=0)
        
        padded_target = torch.cat([
            valid_target,
            torch.full((pad_length,), ignore_id, device=labels_target1.device)
        ], dim=0)
        
        padded_mask = torch.cat([
            valid_mask,
            torch.zeros(pad_length, dtype=torch.bool, device=labels_mask1.device)
        ], dim=0)
        
        merged_embeds.append(padded_embeds)
        merged_target.append(padded_target)
        merged_mask.append(padded_mask)
    
    # 堆叠批次结果
    merged_embeds = torch.stack(merged_embeds, dim=0).to(labels_embeds1.device)
    merged_target = torch.stack(merged_target, dim=0).to(labels_target1.device)
    merged_mask = torch.stack(merged_mask, dim=0).to(labels_mask1.device)
    
    return merged_embeds, merged_target, merged_mask


def make_streaming_mode_from_s2s(text_tokens_padded, text_tokens_lens, speech_tokens_padded, speech_tokens_lens,):
    """

    Args:
        text_tokens_padded: (B, Lmax)
        text_tokens_lens: (B,)
        speech_tokens_padded: (B, Lmax2)
        speech_tokens_lens: (B,)

    Returns:
        streaming_mode_tokens_padded: (B, Lmax+Lmax2+1)
        streaming_mode_tokens_lens: (B,)

    首先assert每个单元的文字有效token的数量的3倍是少于该单元的speech token的数量。
    然后做如下排列：对于batch内的每个item, 先排6个文字有效token,然后再排18个speech 有效token,然后再排6个文字token,然后排18个speech token,以此类推，直到有效文本token用尽。
    对于最后一个文字块，如果有效文字token的总数是6的整数倍，则往结果后面拼上一个特殊符号999作为标记，接着把所有语音有效token拼在后面；
    对于非整数部分，则在最后一个不满6个文字有效token的块中拼上999这个特殊符好。然后把所有语音有效token拼在后面。
    由于999特殊符号一定要加，所有有效token的总长度需要加1
    """
    batch_size = text_tokens_padded.size(0)
    device = text_tokens_padded.device

    # 验证每个单元的文字有效token的数量的3倍少于该单元的speech token的数量
    for i in range(batch_size):
        assert text_tokens_lens[i] * 3 <= speech_tokens_lens[
            i], f"Batch {i}: Text tokens * 3 should be less than speech tokens"

    # 初始化结果
    streaming_mode_tokens_list = []
    streaming_mode_lens = []

    for i in range(batch_size):
        text_tokens = text_tokens_padded[i, :text_tokens_lens[i]]
        speech_tokens = speech_tokens_padded[i, :speech_tokens_lens[i]]

        streaming_tokens = []
        text_idx = 0
        speech_idx = 0

        while text_idx < text_tokens_lens[i]:
            streaming_tokens.extend(text_tokens[text_idx:text_idx + 6].tolist())
            text_idx += 6

            # 如果文字token用尽，但不是正好用尽，则添加999作为标记
            if text_idx > text_tokens_lens[i]:
                streaming_tokens.append(999)
            # 再排18个speech有效token
            streaming_tokens.extend(speech_tokens[speech_idx:speech_idx + 18].tolist())
            speech_idx += 18
        if text_idx == text_tokens_lens[i]:
            # 如果文字token正好用尽，添加999
            streaming_tokens.append(999)
        # 把剩下的所有语音有效token拼在后面
        streaming_tokens.extend(speech_tokens[speech_idx:].tolist())

        # 将结果添加到列表中
        streaming_mode_tokens_list.append(torch.tensor(streaming_tokens, device=device))
        streaming_mode_lens.append(len(streaming_tokens))

    streaming_mode_tokens_padded = pad_sequence(streaming_mode_tokens_list, batch_first=True, padding_value=0)
    streaming_mode_tokens_lens = torch.tensor(streaming_mode_lens, device=device)
    return streaming_mode_tokens_padded, streaming_mode_tokens_lens


def do_embedding_for_two_embeds(input_token_ids, dividing_id, embedding1, embedding2):
    """

    Args:
        input_token_ids: (B, Lmax) ,其词表范围是[0, vocab_size1+vocab_size2)
        dividing_id: int
        embedding1: nn.Embedding(vocab_size1, embedding_dim)
        embedding2: nn.Embedding(vocab_size2, embedding_dim)

    Returns:
        embedding1_output: (B, Lmax, D)

    把两个embeddings 虚拟成一个大的词向量
    """
    mask4embedding1 = input_token_ids < dividing_id
    mask4embedding2 = input_token_ids >= dividing_id
    embedding1_output = embedding1(input_token_ids[mask4embedding1])
    embedding2_output = embedding2(input_token_ids[mask4embedding2]-dividing_id)
    res_output = torch.zeros(input_token_ids.size(0), input_token_ids.size(1), embedding1.embedding_dim, device=embedding1.weight.device)
    res_output[mask4embedding1] = embedding1_output
    res_output[mask4embedding2] = embedding2_output
    return res_output
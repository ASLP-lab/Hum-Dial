import torch
import torch_npu

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.mixtral.modeling_mixtral.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def npu_apply_rotary_pos_emb(q, k, cos, sin):
    """
    https://www.hiascend.com/document/detail/zh/Pytorch/60RC1/apiref/apilist/ptaoplist_000152.html
    TODO: 当前RoPE反向算子只支持cos和sin为batch和num_head为1的情况，因此需要将B1SD转化为1, 1, S, D再转回来
    TODO: BF16情况下算子与小算子精度有差异，fp32下没有
    """
    B,N,S,D = q.shape
    # cos = cos.view(1, 1, B*S, D)
    # sin = sin.view(-1, 1, B*S, D)
    # q_embed = q.transpose(0,1).reshape(-1, 1, B*S, D)
    # k_embed = k.transpose(0,1).reshape(-1, 1, B*S, D)
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    # q_embed = q_embed.view(-1, B, S, D).transpose(0, 1)
    # k_embed = k_embed.view(-1, B, S, D).transpose(0, 1)
    return q_embed, k_embed

def compute_err(act, golden):
    act, golden = act.flatten(), golden.flatten()
    err = act - golden
    abs_error = err.abs()
    rel_err = abs_error / golden.abs()
    print("千分之一", torch.sum(rel_err < 1e-3) / rel_err.numel())

def test_rotary_pos_emb():
    torch_npu.npu.set_compile_mode(jit_compile=False)
    # q = torch.randn((17, 28, 32, 128), dtype=torch.bfloat16, requires_grad=True).npu()
    # k = torch.randn((17, 4, 32, 128), dtype=torch.bfloat16, requires_grad=True).npu()
    # cos = torch.randn((17, 1, 32, 128), dtype=torch.bfloat16, requires_grad=True).npu()
    # sin = torch.randn((17, 1, 32, 128), dtype=torch.bfloat16, requires_grad=True).npu()
    q = torch.randn((17, 28, 32, 128), dtype=torch.float, requires_grad=True).npu()
    k = torch.randn((17, 4, 32, 128), dtype=torch.float, requires_grad=True).npu()
    cos = torch.randn((17, 1, 32, 128), dtype=torch.float, requires_grad=True).npu()
    sin = torch.randn((17, 1, 32, 128), dtype=torch.float, requires_grad=True).npu()
    # ori
    ori_q, ori_k = apply_rotary_pos_emb(q, k, cos, sin, None)
    # fusion
    
    q, k = npu_apply_rotary_pos_emb(q, k, cos, sin)
    print("compute over")
    compute_err(q, ori_q)
    compute_err(k, ori_k)

    out = torch.sum(q[:, :4, ...] + k)
    out.backward()

def test_layer_norm():
    q = torch.randn((17, 4, 1024), dtype=torch.bfloat16, requires_grad=True).npu()
    model = torch.nn.LayerNorm(1024).to(q.device)
    x = model(q)
    
if __name__ == "__main__":
    test_layer_norm()

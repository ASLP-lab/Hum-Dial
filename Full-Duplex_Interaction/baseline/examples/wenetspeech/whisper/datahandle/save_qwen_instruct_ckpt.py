import torch
from transformers import AutoModelForCausalLM



llama_model = AutoModelForCausalLM.from_pretrained(
    "/mnt/sfs/asr/env/.cache/transformers/models--Qwen--Qwen2.5-7B-Instruct-1M/models--Qwen--Qwen2.5-7B-Instruct-1M/snapshots/e28526f7bb80e2a9c8af03b831a9af3812f18fba",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    output_hidden_states=True,
)
# 将llama_model的参数保存到文件中
state_dict = llama_model.state_dict()
keys = list(state_dict.keys())
for key in keys:
    print(key)
# 保存模型参数到文件中
torch.save(state_dict, "/mnt/sfs/asr/env/.cache/transformers/models--Qwen--Qwen2.5-7B-Instruct-1M/llama_model.pt")
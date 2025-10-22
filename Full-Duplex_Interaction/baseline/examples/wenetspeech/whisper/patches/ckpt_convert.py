import torch

def merge_v2(source):
    llama_weight = 'llama_model.base_model.model.model.layers.{layer_idx}.self_attn.{layer_name}.weight'
    llama_bias = 'llama_model.base_model.model.model.layers.{layer_idx}.self_attn.{layer_name}.bias'
    for i in range(28):
        layer_idx = str(i)
        source[llama_weight.format(layer_idx=layer_idx, layer_name='w_pack')]= torch.cat(
            [source[llama_weight.format(layer_idx=layer_idx, layer_name='q_proj')],
            source[llama_weight.format(layer_idx=layer_idx, layer_name='k_proj')],
            source[llama_weight.format(layer_idx=layer_idx, layer_name='v_proj')]],
            dim=0
        ).contiguous()
        print(source[llama_weight.format(layer_idx=layer_idx, layer_name='w_pack')].shape)
        source[llama_bias.format(layer_idx=layer_idx, layer_name='w_pack')]= torch.cat(
            [source[llama_bias.format(layer_idx=layer_idx, layer_name='q_proj')],
            source[llama_bias.format(layer_idx=layer_idx, layer_name='k_proj')],
            source[llama_bias.format(layer_idx=layer_idx, layer_name='v_proj')]],
            dim=0
        ).contiguous()
        print(source[llama_bias.format(layer_idx=layer_idx, layer_name='w_pack')].shape)

if __name__ == "__main__":
    ori_ckpt_path = "/mnt/sfs/asr/ckpt/epoch_3.pt"
    ckpt_output_path = "/mnt/obs/ckpt/um/qwen2_multi_task_4_fa/epoch_0_with_speechp/step_0/mp_rank_00_model_states_merge_llama_qkv.pt"
    ori_ckpt = torch.load(ori_ckpt_path, map_location="cpu")
    merge_v2(ori_ckpt)
    torch.save(ori_ckpt, ckpt_output_path)
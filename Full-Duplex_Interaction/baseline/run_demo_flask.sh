#!/bin/bash

# set cuda path
export CUDA_HOME=/usr/local/cuda-12.1 
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:./anaconda3/envs/easy_turn_osum_echat/lib/python3.12/site-packages/nvidia/cudnn/lib  # 指定 Anaconda 虚拟环境的 cuDNN 动态库路径（确保路径存在），这里的虚拟环境名为easy_turn_osum_echat
# set python path
export PYTHONPATH=./baseline:${PYTHONPATH}  #baseline的文件夹路径
# set gpu
export CUDA_VISIBLE_DEVICES="2,3,4,5" # max_users数量由GPU数量决定, 尽量不少于两块GPU,否则可能会报错
export USE_NPU="false"
export TORCH_LOGS=recompiles
python web_flask/chat_stream.py \
    --ipt_model_path "./baseline/ckpt/easy_turn.pt" \
    --ipt_config_path "./baseline/conf/config_easy_turn_infer.yaml" \
    --slm_model_path "./baseline/ckpt/osum_echat.pt" \
    --slm_config_path "./baseline/conf/config_llm_gpu_instruct_3B_cosyvoice1-token.yaml" \
    --tts_fm_model_path "./baseline/ckpt/fm_model_900000.pt" \
    --tts_fm_config_path "./baseline/conf/fm_10ms_nocfg_contrasive_emilia.yaml" \
    --tts_vc_model_path "./baseline/ckpt/hifigan_g_00400000" \
    --max_users 4 \
    --port 5000 \
    || exit 1;



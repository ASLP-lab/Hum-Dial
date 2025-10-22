#!/bin/bash

# ----------------------------
# Step 1: 激活conda环境（按需修改路径）
# ----------------------------
# source /home/environment2/hkxie/anaconda3/bin/activate /home/environment2/hkxie/anaconda3/envs/F5-TTS
# source /home/work_nfs19/hkxie/environment/anaconda3/bin/activate /home/work_nfs19/hkxie/environment/anaconda3/envs/f5-tts
# source /home/work_nfs19/hkxie/environment/anaconda3/bin/activate /home/work_nfs19/hkxie/environment/anaconda3/envs/F5-TTS
source /home/work_nfs19/hkxie/environment/anaconda3/bin/activate /home/work_nfs19/hkxie/environment/anaconda3/envs/covomix
# /home/node57_data/hkxie/4O/streaming_fm/data

# ----------------------------
# Step 2: 定位到F5-TTS根目录
# ----------------------------
# 获取脚本绝对路径（兼容软链接）
SCRIPT_PATH=$(readlink -f "$0")
# 定位到项目根目录：从脚本路径向上回退5级（src/f5_tts/train -> 根目录）
PROJECT_ROOT=$(dirname "$(dirname "$(dirname "$(dirname "$SCRIPT_PATH")")")")
cd "$PROJECT_ROOT" || { echo "Failed to enter project root"; exit 1; }

# -----s-----------------------
# Step 3: 验证当前路径
# ----------------------------
echo "当前工作目录：$(pwd)"
echo "预期根目录：/home/work_nfs14/code/hkxie/TTS/F5-TTS"  # 请核对路径是否一致

# ----------------------------
# Step 4: 执行训练命令
# ----------------------------
# export NCCL_P2P_DISABLE=1
# export NCCL_IB_DISABLE=1
#517 8卡
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
accelerate launch --num_processes=8 --main_process_port 64439 src/f5_tts/train/train.py --config-name fm_10ms_nocfg_contrasive_ecapa.yaml

python3 /home/node57_data/hkxie/xxtool/send.py "node60 covomix_streamingfm_s3token2_ecaptdnn_spk 训练中断？显存？"
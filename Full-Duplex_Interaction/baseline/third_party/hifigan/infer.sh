source /home/environment2/hkxie/anaconda3/bin/activate /home/environment/ypjiang/anaconda3/envs/torch2
source /home/environment2/hkxie/anaconda3/bin/activate /home/environment2/hkxie/anaconda3/envs/covomix


# python3 /home/work_nfs14/code/hkxie/hifigan_decoder/inference_e2e.py \
#     --input_mels_dir=$1 \
#     --output_dir=$2 \
#     --checkpoint_file=/home/work_nfs14/code/hkxie/hifigan_decoder/dac_hifigan_hqft/g_00800000

python3 inference.py \
    --input_wavs_dir=$1 \
    --output_dir=$2 \
    --checkpoint_file=/home/node40_data/hkxie/hifi-gan/ckpt_hifigan/g_00210000
    #/home/work_nfs14/code/hkxie/hifigan_decoder/dac_hifigan/g_00300000 ##4层codebook 32k
    #--checkpoint_file=/home/work_nfs14/code/hkxie/hifigan_decoder/fulltoken_hifigan/g_00400000 ##遗失
    # bash infer.sh /home/work_nfs14/code/hkxie/TTS/hifi-gan/orig_wav /home/work_nfs14/code/hkxie/TTS/hifi-gan/output
#source /home/node57_data/hkxie/environment/anaconda3/bin/activate /home/node57_data/hkxie/environment/anaconda3/envs/maskgct
# source /home/environment2/hkxie/anaconda3/bin/activate /home/environment2/hkxie/anaconda3/envs/covomix

#提取mel，混合mel和单spk的mel

python mel_extract.py \
    --processed_path /home/work_nfs14/code/hkxie/ASR/understanding_LLM_task/datalist/wav.scp \
    --target_path /home/node57_data/hkxie/4O/streaming_fm/data/mel_bigvgan_24k_100b_256x/


python3 /home/node57_data/hkxie/xxtool/send.py "node44 bigvgan mel 提取完毕 or 异常报错 ？"

#处理完后，搬运数据
# /home/node57_data/hkxie/dataset/acoustic
# rsync -avz --progress --exclude='t2s/train/wav/' /home/node57_data/hkxie/dataset/acoustic aslp@10.68.109.101:/mnt/nfs14/code/hkxie/tmpdata/

# # #处理trim norm流程脚本 #只要norm就行了，trim切静音会把对话一方的静音切除
# bash /home/node57_data/hkxie/ds.sh \
#     $input_wav_dir /home/node57_data/hkxie/dialogue/\
#     $normed_wav_dir /home/node57_data/hkxie/dialogue_trim_normed/\
#     48000
# python3 /home/node57_data/hkxie/norm_trim.py \
#     /home/node57_data/hkxie/dialogue/ \
#     /home/node57_data/hkxie/dialogue_trim_normed/ \
#     16000
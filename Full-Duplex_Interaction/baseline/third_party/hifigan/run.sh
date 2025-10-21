source /home/environment2/hkxie/anaconda3/bin/activate /home/environment2/hkxie/anaconda3/envs/covomix

# cd /home/node40_data/hkxie/hifi-gan
CUDA_VISIBLE_DEVICES=0,1,2,3
python train.py --config config_streamfm10ms.json --input_training_file 24k_wavs.scp --input_validation_file valwavs.txt --checkpoint_path ./ckpt_hifigan --validation_interval 5000 --checkpoint_interval 5000
# 24k_train_wavs.txt


source /home/environment2/hkxie/anaconda3/bin/activate /home/environment2/hkxie/anaconda3/envs/F5-TTS

cd /home/node60_tmpdata/hkxie/osum_dit/src/f5_tts/infer
# python infer_streaming_official.py \
#     --wav_path /home/node57_data/hkxie/4O/F5-TTS/src/f5_tts/infer/cosyvoice2_token_test \
#     --token_path /home/node57_data/hkxie/4O/F5-TTS/src/f5_tts/infer/cosyvoice2_token_test \
#     --output_path /home/node60_tmpdata/hkxie/covomix/F5-TTS/testout/20250528_infer

# python infer_streaming_official.py \
#     --wav_path /home/node57_data/hkxie/4O/streaming_fm/testset/s3token1 \
#     --token_path /home/node57_data/hkxie/4O/streaming_fm/testset/s3token1 \
#     --output_path /home/node60_tmpdata/hkxie/osum_dit/testout/20250624_infer/cfg



# python infer_streaming_official.py \
#     --wav_path /home/node57_data/hkxie/4O/streaming_fm/testset/origin_wav \
#     --token_path /home/node57_data/hkxie/4O/streaming_fm/testset/s3token1 \
#     --output_path /home/node60_tmpdata/hkxie/osum_dit/testout/20250629_infer_nocfg_copysyn_emilia_30wsteps_10


# cd /home/node57_data/hkxie/4O/streaming_fm/testset/streaming_xcodec2_evaluate
# source /home/environment2/hkxie/anaconda3/bin/activate /home/environment2/hkxie/anaconda3/envs/covomix
# bash eval_mertics.sh /home/node60_tmpdata/hkxie/osum_dit/testout/20250629_infer_nocfg_copysyn_emilia_30wsteps_10/no_streaming/ ./ 0


# echo "0705_70wsteps"

# python infer_streaming_official.py \
#     --wav_path /home/node57_data/hkxie/4O/streaming_fm/testset/origin_wav \
#     --token_path /home/node57_data/hkxie/4O/streaming_fm/testset/s3token1 \
#     --output_path /home/node60_tmpdata/hkxie/osum_dit/testout/20250629_infer_nocfg_copysyn_emilia_60wsteps_10


# cd /home/node57_data/hkxie/4O/streaming_fm/testset/streaming_xcodec2_evaluate
# source /home/environment2/hkxie/anaconda3/bin/activate /home/environment2/hkxie/anaconda3/envs/covomix
# bash eval_mertics.sh /home/node60_tmpdata/hkxie/osum_dit/testout/20250629_infer_nocfg_copysyn_emilia_60wsteps_10/no_streaming/ ./ 0



echo "0705_70wsteps"

python infer_streaming_official.py \
    --wav_path /home/node57_data/hkxie/4O/streaming_fm/testset/origin_wav \
    --token_path /home/node57_data/hkxie/4O/streaming_fm/testset/s3token1 \
    --output_path /home/node60_tmpdata/hkxie/osum_dit/testout/20250705_infer_nocfgstreaming_copysyn_emilia_70wsteps_10


# cd /home/node57_data/hkxie/4O/streaming_fm/testset/streaming_xcodec2_evaluate
# source /home/environment2/hkxie/anaconda3/bin/activate /home/environment2/hkxie/anaconda3/envs/covomix
# bash eval_mertics.sh /home/node60_tmpdata/hkxie/osum_dit/testout/20250629_infer_nocfg_copysyn_emilia_60wsteps_10/no_streaming/ ./ 0
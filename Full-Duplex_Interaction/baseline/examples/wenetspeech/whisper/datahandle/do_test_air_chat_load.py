import torchaudio
torchaudio.set_audio_backend("sox")
torchaudio.utils.sox_utils.set_buffer_size(16500)
wav_file ="/mnt/sfs/asr/test_data/test_sets_format_3000/public_test/AirBench_speech/wav/568.38_596.46.wav"
wav_file = "/mnt/sfs/asr/test_data/test_sets_format_3000/public_test/AirBench_speech/wav/common_voice_en_17782101.mp3"
waveform, sample_rate = torchaudio.load(wav_file)
print(waveform)
print(waveform.shape)
waveform = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=16000)(waveform)
print(waveform.shape)
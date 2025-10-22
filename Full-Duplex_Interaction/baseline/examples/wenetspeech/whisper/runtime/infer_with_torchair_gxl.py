import random
import sys
import time
import torch
import torch_npu
import torchaudio
import os

import tqdm

sys.path.insert(0,'.')
sys.path.insert(0,'../../../../')
from patches import modelling_qwen2_infer_patch # 打patch

from gxl_ai_utils.utils import utils_file
from wenet.utils.init_tokenizer import init_tokenizer
from gxl_ai_utils.config.gxl_config import GxlNode
from wenet.utils.init_model import init_model
import logging
import librosa
# from cosyvoice_util import token_list2wav2
# gpu_id = 7
# os.environ['ASCEND_RT_VISIBLE_DEVICES'] = str(gpu_id)
# ======================= debug & profiling =====================
from patches.modelling_qwen2_infer_patch import DebugHelper
# from msit_llm import seed_all, DumpConfig, register_hook

experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization, profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False
    )
# prof = torch_npu.profiler.profile(
#     activities=[
#         torch_npu.profiler.ProfilerActivity.CPU,
#         torch_npu.profiler.ProfilerActivity.NPU
#         ],
#     schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=0),
#     on_trace_ready=torch_npu.profiler.tensorboard_trace_handler("./output/profiles/infer_compile_phase3"),
#     record_shapes=True,
#     profile_memory=True,
#     with_stack=False,
#     with_flops=False,
#     with_modules=True,
#     experimental_config=experimental_config)
prof = None

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')
# seed_all(42)
# # ======================= load mmodel =====================
# config_path = "/mnt/sfs/asr/code/zy/osum_xlgeng_zy/examples/wenetspeech/whisper/conf/config_llm_huawei_instruct-version_cosyvoice1-token.yaml" # 不用管，代码写死用instruct格式进行推理
checkpoint_path = "/mnt/sfs/asr/ckpt/qwen2_multi_task_4_6gpus_gxl_adapter/full_train_llm_3B_epoch_2_new/step_19999.pt"
checkpoint_path = "/mnt/sfs/asr/ckpt/qwen2_multi_task_4_6gpus_gxl_adapter/full_train_new_pattern_from_epoch9_now_epoch0/step_44999.pt"
checkpoint_path = ""
config_path = "../conf/config_llm_huawei_instruct_3B_cosyvoice1-token.yaml"
# checkpoint_path = "/mnt/sfs/asr/ckpt/qwen2_multi_task_4_6gpus_gxl_adapter/epoch26_cosyvoice1_new-set_token_10w_s2s/step_84999.pt"
# checkpoint_path = "/mnt/sfs/asr/ckpt/qwen2_multi_task_4_6gpus_gxl_adapter/epoch26_cosyvoice1_new-set_token_10w_s2s/step_84999.pt"
args = GxlNode({
    "checkpoint": checkpoint_path,
})
configs = utils_file.load_dict_from_yaml(config_path)
model, configs = init_model(args, configs)
device = torch.device(f'npu')
model = model.to(device)
tokenizer = init_tokenizer(configs)
print(model)

def do_resample(input_wav_path, output_wav_path):
    """"""
    waveform, sample_rate = torchaudio.load(input_wav_path)
    # 检查音频的维度
    num_channels = waveform.shape[0]
    # 如果音频是多通道的，则进行通道平均
    if num_channels > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    waveform = torchaudio.transforms.Resample(
        orig_freq=sample_rate, new_freq=16000)(waveform)
    utils_file.makedir_for_file(output_wav_path)
    torchaudio.save(output_wav_path, waveform, 16000)

def get_feat_from_wav_path(input_wav_path):
    """
    获取音频的特征
    Args:
        input_wav_path: str

    Returns:
        feat: tensor, shape=(1, T, 80)
        feat_lens: tensor, shape=(1,)
    """
    timestamp_ms = int(time.time() * 1000)
    now_file_tmp_path_resample = f'~/.cache/.temp/{timestamp_ms}_resample.wav'
    do_resample(input_wav_path, now_file_tmp_path_resample)
    input_wav_path = now_file_tmp_path_resample
    waveform, sample_rate = torchaudio.load(input_wav_path)
    waveform = waveform.squeeze(0)  # (channel=1, sample) -> (sample,)
    window = torch.hann_window(400)
    stft = torch.stft(waveform,
                      400,
                      160,
                      window=window,
                      return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = torch.from_numpy(
        librosa.filters.mel(sr=sample_rate,
                            n_fft=400,
                            n_mels=80))
    mel_spec = filters @ magnitudes

    # NOTE(xcsong): https://github.com/openai/whisper/discussions/269
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    feat = log_spec.transpose(0, 1)
    feat_lens = torch.tensor([feat.shape[0]], dtype=torch.int64).to(device)
    feat = feat.unsqueeze(0).to(device)
    return feat, feat_lens

def do_s2t(input_wav_path, input_prompt, profile=False):
    model.eval()    # 设置为推理模式，关闭lora的dropout
    feat, feat_lens = get_feat_from_wav_path(input_wav_path)
    print(f'feat shape: {feat.shape}')
    print(f'feat_lens: {feat_lens}')
    if profile: prof.start()
    torch_npu.npu.synchronize()
    start_time = time.time()  # 开始计时
    # 采用静态Cache
    res_text = model.generate(wavs=feat, wavs_len=feat_lens, prompt=input_prompt, cache_implementation="static")[0]
    end_time = time.time()
    print(f"推理消耗时间: {end_time - start_time:.2f} 秒")
    if profile: prof.step()
    if profile: prof.stop()
    return res_text

def do_t2s(input_prompt, text_for_tts, profile=False):
    model.eval()
    if profile: prof.start()
    torch_npu.npu.synchronize()
    start_time = time.time()
    res_text = model.generate_tts(device=device, text=text_for_tts, prompt=input_prompt)[0]
    end_time = time.time()
    print(f"推理消耗时间: {end_time - start_time:.2f} 秒")
    if profile: prof.step()
    if profile: prof.stop()
    return res_text

def do_s2s(input_wav_path, input_prompt, profile=False):
    model.eval()
    feat, feat_lens = get_feat_from_wav_path(input_wav_path)
    print(f'feat shape: {feat.shape}')
    print(f'feat_lens: {feat_lens}')
    if profile: prof.start()
    torch_npu.npu.synchronize()
    start_time = time.time()
    output_text, text_res, speech_res = model.generate_s2s_no_stream(wavs=feat, wavs_len=feat_lens, prompt=input_prompt)
    end_time = time.time()
    print(f"推理消耗时间: {end_time - start_time:.2f} 秒")
    if profile: prof.step()
    if profile: prof.stop()
    return (text_res, speech_res)

def do_t2t(question_txt, profile=False):
    model.eval()
    if profile: prof.start()
    torch_npu.npu.synchronize()
    start_time = time.time()
    res_text = model.generate_text2text(device=device, text=question_txt)[0]
    end_time = time.time()
    print(f"推理消耗时间: {end_time - start_time:.2f} 秒")
    if profile: prof.step()
    if profile: prof.stop()
    return res_text

def _main_tts():
    torch_npu.npu.set_compile_mode(jit_compile=False)
    # input_wav_path = "/mnt/sfs/asr/update_data/align_cn_noize/roobo_100097437.wav"
    # input_prompt = "将这段音频的语音内容详细记录为文字稿。"
    # res_text = do_decode(input_wav_path, input_prompt, profile=False)
    # print(f"推理结果: {res_text}")
    
    # ===================warm up
    print("start warmup...")
    input_wav_path = "/mnt/sfs/asr/update_data/align_cn_noize/roobo_100097437.wav"
    input_prompt = "将这段音频的语音内容详细记录为文字稿。"
    res_text = do_s2t(input_wav_path, input_prompt, profile=False)
    print(f'推理结果: {res_text}')
    print("finish warmup")
    # ===================warm up finish
    

    # ===================TTS
    input_prompt = "恳请将如下文本转换为其对应的语音token，力求生成最为流畅、自然的语音。"
    text_for_tts = "你好，欢迎使用语音合成服务，中国人都是非常好的人。"
    res_text = do_t2s(input_prompt, text_for_tts, profile=False)
    # token_list2wav2(res_text, output_file_path='./test_1.wav')
    # ===================TTS
    input_prompt = "恳请将如下文本转换为其对应的语音token，力求生成最为流畅、自然的语音。"
    text_for_tts = "中国没有接招，中国商务部回应称，“美方威胁升级对华关税，是错上加错，再次暴露了美方的讹诈本质，中方对此绝不接受。如果美方一意孤行，中方必将奉陪到底。"
    res_text = do_t2s(input_prompt, text_for_tts, profile=False)
    # token_list2wav2(res_text, output_file_path='./test_2.wav')
    # ===================TTS
    input_prompt = "恳请将如下文本转换为其对应的语音token，力求生成最为流畅、自然的语音。"
    text_for_tts = "太平洋的另一边，中国也没把话说死，而是在声明中留有余地称，美国应“停止对华经贸打压，与中方在相互尊重的基础上，通过平等对话妥善解决分歧。”这反映出中国希望美国降低一些恶意措施，释放善意之后，才能为谈判铺平道路。"
    res_text = do_t2s(input_prompt, text_for_tts, profile=False)
    # token_list2wav2(res_text, output_file_path='./test_3.wav')
    # ===================TTS
    input_prompt = "恳请将如下文本转换为其对应的语音token，力求生成最为流畅、自然的语音。"
    text_for_tts = "泽连斯基贴出的影片显示，其中一名被指中国俘虏的男子手戴手铐，用普通话描述最近一场战斗的情况。"
    res_text = do_t2s(input_prompt, text_for_tts, profile=False)
    # token_list2wav2(res_text, output_file_path='./test_4.wav')
    # ===================TTS
    input_prompt = "恳请将如下文本转换为其对应的语音token，力求生成最为流畅、自然的语音。"
    text_for_tts = "他还说：“俄罗斯拉拢中国及其他国家直接或间接参与这场欧洲战争，这明确显示普京丝毫无意结束战争。"
    res_text = do_t2s(input_prompt, text_for_tts, profile=False)
    # token_list2wav2(res_text, output_file_path='./test_5.wav')
    # ===================TTS
    input_prompt = "恳请将如下文本转换为其对应的语音token，力求生成最为流畅、自然的语音。"
    text_for_tts = "中国外交部发言人周三在例行记者会上表示，基辅声称“许多”中国公民协助俄罗斯参战的说法“毫无根据”。泽连斯基贴出的影片显示，其中一名被指中国俘虏的男子手戴手铐，用普通话描述最近一场战斗的情况。美国应“停止对华经贸打压，与中方在相互尊重的基础上，通过平等对话妥善解决分歧。”这反映出中国希望美国降低一些恶意措施，释放善意之后，才能为谈判铺平道路。"
    res_text = do_t2s(input_prompt, text_for_tts, profile=False)
    # token_list2wav2(res_text, output_file_path='./test_6.wav')
    # ===================TTS
    input_prompt = "恳请将如下文本转换为其对应的语音token，力求生成最为流畅、自然的语音。"
    text_for_tts = "其中一名被指中国俘虏的男子手戴手铐，用普通话描述最近一场战斗的情况。"
    res_text = do_t2s(input_prompt, text_for_tts, profile=False)
    # token_list2wav2(res_text, output_file_path='./test_7.wav')
    # ===================TTS
    input_prompt = "恳请将如下文本转换为其对应的语音token，力求生成最为流畅、自然的语音。"
    text_for_tts = "基辅声称“许多”中国公民协助俄罗斯参战的说法“毫无根据”。"
    res_text = do_t2s(input_prompt, text_for_tts, profile=False)
    # token_list2wav2(res_text, output_file_path='./test_8.wav')
    # ===================TTS
    input_prompt = "恳请将如下文本转换为其对应的语音token，力求生成最为流畅、自然的语音。"
    text_for_tts = "中国外交部发言人周三在例行记者会上表示"
    res_text = do_t2s(input_prompt, text_for_tts, profile=False)
    # token_list2wav2(res_text, output_file_path='./test_9.wav')

def _main_s2s():
    torch_npu.npu.set_compile_mode(jit_compile=False)
    # input_wav_path = "/mnt/sfs/asr/update_data/align_cn_noize/roobo_100097437.wav"
    # input_prompt = "将这段音频的语音内容详细记录为文字稿。"
    # res_text = do_decode(input_wav_path, input_prompt, profile=False)
    # print(f"推理结果: {res_text}")
    # ===================warm up
    # print("start warmup...")
    # input_wav_path = "/mnt/sfs/asr/update_data/align_cn_noize/roobo_100097437.wav"
    # input_prompt = "将这段音频的语音内容详细记录为文字稿。"
    # res_text = do_s2t(input_wav_path, input_prompt, profile=False)
    # print(f'ASR推理结果: {res_text}')

    # ===================warm up
    print("start warmup...")
    input_wav_path = "./input_data/question1.wav"
    input_prompt = "以聊天的方式对以下音频获取相应的回答文本，之后把该回答文本转换为对应的语音token，旨在生成流畅自然的语音。"
    res_text = do_s2s(input_wav_path, input_prompt, profile=False)
    print(f'推理结果: {res_text}')
    print("finish warmup")
    # ===================warm up finish

    # ===================S2S
    print("start warmup...")
    input_wav_path = "./input_data/question1.wav"
    input_prompt = "以聊天的方式对以下音频获取相应的回答文本，之后把该回答文本转换为对应的语音token，旨在生成流畅自然的语音。"
    res_text = do_s2s(input_wav_path, input_prompt, profile=False)
    print(f'推理结果: {res_text}')
    print("finish warmup")

    # ===================S2S
    print("start warmup...")
    input_wav_path = "./input_data/question1.wav"
    input_prompt = "以聊天的方式对以下音频获取相应的回答文本，之后把该回答文本转换为对应的语音token，旨在生成流畅自然的语音。"
    res_text = do_s2s(input_wav_path, input_prompt, profile=False)
    print(f'推理结果: {res_text}')
    print("finish warmup")

    # ===================S2S
    print("start warmup...")
    input_wav_path = "./input_data/question1.wav"
    input_prompt = "以聊天的方式对以下音频获取相应的回答文本，之后把该回答文本转换为对应的语音token，旨在生成流畅自然的语音。"
    res_text = do_s2s(input_wav_path, input_prompt, profile=False)
    print(f'推理结果: {res_text}')
    print("finish warmup")

    # ===================S2S
    print("start warmup...")
    input_wav_path = "./input_data/question1.wav"
    input_prompt = "以聊天的方式对以下音频获取相应的回答文本，之后把该回答文本转换为对应的语音token，旨在生成流畅自然的语音。"
    res_text = do_s2s(input_wav_path, input_prompt, profile=False)
    print(f'推理结果: {res_text}')
    print("finish warmup")

def _main_s2t():
    """"""
    # data.list ,
    # emotion-> <EMOTION>
    dict_list = utils_file.load_dict_list_from_jsonl("")
    prompt_dict = utils_file.load_dict_from_yaml("../conf/prompt_config_new.yaml")
    prompt_list = prompt_dict['prompt_list']
    # 随机选择一个prompt
    res_dict = {}
    output_path = "./**/**.scp"
    # 随机选择一个emotion
    for dict_item in tqdm.tqdm(dict_list, desc="inference", total=len(dict_list)):
        input_prompt = random.choice(prompt_list)
        key = dict_item['key']
        wav_path = dict_item['wav']
        res_text = do_s2t(wav_path, input_prompt, profile=False)
        print(f'推理结果: {res_text}')
        res_dict[key] = res_text
    utils_file.write_dict_to_scp(res_dict, output_path)

def _main_t2t():
    """"""
    while True:
        input_text = input("请输入文本: ")
        if input_text == "exit":
            break
        res_text = do_t2t(input_text, profile=False)
        print(f'推理结果: {res_text}')



if __name__ == '__main__':
    # _main_tts()
    # _main_s2s()
    # _main_t2t()
    input_wav_path = "./input_data/1_0.wav"
    text_res, speech_res = do_s2s(input_wav_path,
                                  "以聊天的方式对以下音频获取相应的回答文本，之后把该回答文本转换为对应的语音token，旨在生成流畅自然的语音。",
                                  profile=False)
    print(f'推理结果: {text_res}')
    print(f'音频token: {speech_res}')
    input_wav_path = "./input_data/2_0.wav"
    text_res, speech_res = do_s2s(input_wav_path,
                                  "以聊天的方式对以下音频获取相应的回答文本，之后把该回答文本转换为对应的语音token，旨在生成流畅自然的语音。",
                                  profile=False)
    print(f'推理结果: {text_res}')
    print(f'音频token: {speech_res}')
    input_wav_path = "./input_data/3_0.wav"
    text_res, speech_res = do_s2s(input_wav_path,
                                  "以聊天的方式对以下音频获取相应的回答文本，之后把该回答文本转换为对应的语音token，旨在生成流畅自然的语音。",
                                  profile=False)
    print(f'推理结果: {text_res}')
    print(f'音频token: {speech_res}')
    input_wav_path = "./input_data/4_0.wav"
    text_res, speech_res = do_s2s(input_wav_path,
                                  "以聊天的方式对以下音频获取相应的回答文本，之后把该回答文本转换为对应的语音token，旨在生成流畅自然的语音。",
                                  profile=False)
    print(f'推理结果: {text_res}')
    print(f'音频token: {speech_res}')
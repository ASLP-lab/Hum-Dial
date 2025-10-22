import base64
import json
import time

import gradio as gr


import sys


sys.path.insert(0, '../../../../')
sys.path.insert(0, '.')
# sys.path.insert(0,'.')
# sys.path.insert(0,'../../../../')
from patches import modelling_qwen2_infer_patch # 打patch
from cosyvoice_util import token_list2wav
from gxl_ai_utils.utils import utils_file
try:
    from wenet.utils.init_tokenizer import init_tokenizer
    from gxl_ai_utils.config.gxl_config import GxlNode
    from wenet.utils.init_model import init_model
    import logging
    import librosa
    import torch
    import torchaudio
except ImportError:
    pass
is_npu = True

try:
    import torch_npu
except ImportError:
    is_npu = False
    print("torch_npu is not available. if you want to use npu, please install it.")

# 将图片转换为 Base64
with open("./实验室.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")


TASK_PROMPT_MAPPING = {
    "ASR (Automatic Speech Recognition)": "执行语音识别任务，将音频转换为文字。",
    "SRWT (Speech Recognition with Timestamps)": "请转录音频内容，并为每个英文词汇及其对应的中文翻译标注出精确到0.1秒的起止时间，时间范围用<>括起来。",
    "VED (Vocal Event Detection)(类别:laugh，cough，cry，screaming，sigh，throat clearing，sneeze，other)": "请将音频转录为文字记录，并在记录末尾标注<音频事件>标签，音频事件共8种：laugh，cough，cry，screaming，sigh，throat clearing，sneeze，other。",
    "SER (Speech Emotion Recognition)(类别:sad，anger，neutral，happy，surprise，fear，disgust，和other)": "请将音频内容转录成文字记录，并在记录末尾标注<情感>标签，情感共8种：sad，anger，neutral，happy，surprise，fear，disgust，和other。",
    "SSR (Speaking Style Recognition)(类别:新闻科普，恐怖故事，童话故事，客服，诗歌散文，有声书，日常口语，其他)": "请将音频内容进行文字转录，并在最后添加<风格>标签，标签共8种：新闻科普、恐怖故事、童话故事、客服、诗歌散文、有声书、日常口语、其他。",
    "SGC (Speaker Gender Classification)(类别:female,male)": "请将音频转录为文本，并在文本结尾处标注<性别>标签，性别为female或male。",
    "SAP (Speaker Age Prediction)(类别:child、adult和old)": "请将音频转录为文本，并在文本结尾处标注<年龄>标签，年龄划分为child、adult和old三种。",
    "STTC (Speech to Text Chat)": "首先将语音转录为文字，然后对语音内容进行回复，转录和文字之间使用<开始回答>分割。",
    "Only Age Prediction(类别:child、adult和old)": "请更具如下音频中说话者的声音判断其性别，并直接给出<年龄>标签，标明是child、adult还是old。",
    "Only Gender Classification(类别:female,male)": "请判断语音中发言者的性别，并在结果中标注<性别>标签，性别分为female和male。",
    "Only Style Recognition(类别:新闻科普，恐怖故事，童话故事，客服，诗歌散文，有声书，日常口语，其他)": "请判断如下音频的风格，直接给出<风格>标签，风格分为8类：新闻科普、恐怖故事、童话故事、客服、诗歌散文、有声书、日常口语、其他。",
    "Only Emotion Recognition(类别:sad，anger，neutral，happy，surprise，fear，disgust，和other)": "请判别如下音频里发言者的情感，径直给出<情感>标签，情感共计8类：sad，anger，neutral，happy，surprise，fear，disgust，和other。",
    "Only  Event Detection(类别:laugh，cough，cry，screaming，sigh，throat clearing，sneeze，other)": "分析以下音频中包含的音频事件，并直接给出<音频事件>标签，音频事件有8种分类：laugh，cough，cry，screaming，sigh，throat clearing，sneeze，other。",
    "Chat": "根据语音输入，直接以文字形式进行回答或对话。",
    "ASR+AGE+GENDER": '请将这段音频进行转录，并在转录完成的文本末尾附加<年龄> <性别>标签。年龄分为child、adult、old三种，性别为female或male。',
    "AGE+GENDER": "请识别以d下音频发言者的年龄和性别。年龄分为child、adult、old三种，性别为male和female。",
    "ASR+STYLE+AGE+GENDER": "请对以下音频内容进行转录，并在文本结尾分别添加<风格>、<年龄>、<性别>标签。风格有8种类型：新闻科普、恐怖故事、童话故事、客服、诗歌散文、有声书、日常口语、其他。年龄划分为child、adult、old三个阶段，性别则为male和female。",
    "STYLE+AGE+GENDER": "请对以下音频进行分析，识别说话风格、说话者年龄和性别。风格类型有8种，包括新闻科普、恐怖故事、童话故事、客服、诗歌散文、有声书、日常口语、其他。年龄阶段为child、adult、old三种，性别为male或female。",
    "<TRANSCRIBE> <PUNCTUATION>": "需对提供的语音文件执行文本转换，同时为转换结果补充必要的标点。",
    "<TRANSCRIBE> <DIALECT>": "请将下列音频内容转录为文字，并在转录文本后标注中文方言种类标签。其中分为 粤语:<CN_YUEYU>,上海话：<CN_SHANGHAI>，四川话：<CN_SICHUAN>，普通话：<CN>。",
    "<TRANSCRIBE> <CAPTION> <AGE> <GENDER>": "请将以下音频内容进行转录，并在转录完成的文本末尾分别附加<音频事件>、<年龄>、<性别>标签。音频事件分为8类：laugh，cough，cry，screaming，sigh，throat clearing，sneeze，other。年龄分为child、adult、old三种，性别为male和female。",
    "<TRANSCRIBE> <EMOTION> <AGE> <GENDER>": "请将下列音频内容进行转录，并在转录文本的末尾分别添加<情感>、<年龄>、<性别>标签。情感包括8类：sad，anger，neutral，happy，surprise，fear，disgust，以及other。年龄划分为child、adult、old三种，性别分为male和female。",
    "<S2TCHAT> <TEXT2TOKEN>":"先根据语音输入，直接以文字形式进行回答或对话，接着再生成语音token。",
    "<S2TCHAT> <TEXT2TOKEN> <EMOTION>":"请先判断回复的情绪，并输出情感标签，然后以聊天的方式对以下音频获取相应的回答文本，之后把该回答文本转换为对应的带情感的语音token，旨在生成流畅自然的语音,情感共分为8类：sad，anger，neutral，happy，surprise，fear，disgust，sorry。"
}

experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization, profiler_level=torch_npu.profiler.ProfilerLevel.Level1, l2_cache=False
    )

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')
checkpoint_path = "/mnt/sfs/asr/ckpt/qwen2_multi_task_4_6gpus_gxl_adapter/full_train_llm_3B_epoch_0/step_32499.pt"
config_path = "../conf/config_llm_huawei_instruct_3B_cosyvoice1-token.yaml"
args = GxlNode({
    "checkpoint": checkpoint_path,
})
configs = utils_file.load_dict_from_yaml(config_path)
model, configs = init_model(args, configs)
device = torch.device(f'npu')
model = model.to(device)
tokenizer = init_tokenizer(configs)
print(model)

#
# def init_model_my():
#     logging.basicConfig(level=logging.DEBUG,
#                         format='%(asctime)s %(levelname)s %(message)s')
#     checkpoint_path = "/mnt/sfs/asr/ckpt/qwen2_multi_task_4_6gpus_gxl_adapter/epoch_33_LLMinstruct_cosyvoice1_10Wtts_1WenTTS_2Khqtts_1KenS2S_3Ks2s_5Ws2t/step_47499.pt"
#     checkpoint_path="/mnt/sfs/asr/ckpt/qwen2_multi_task_4_6gpus_gxl_adapter/epoch_34_LLMinstruct_cosyvoice1_10Wtts_1WenTTS_2Khqtts_1KenS2S_3Ks2s_5Ws2t/step_17499.pt"
#     config_path = "../conf/config_llm_huawei_instruct-version_cosyvoice1-token.yaml"
#     args = GxlNode({
#         "checkpoint": checkpoint_path,
#     })
#     configs = utils_file.load_dict_from_yaml(config_path)
#     model, configs = init_model(args, configs)
#     if is_npu:
#         device = torch.device(f'npu:{gpu_id}')
#     else:
#         device =torch.device(f'cuda:{gpu_id}')
#     model = model.to(device)
#     tokenizer = init_tokenizer(configs)
#     print(model)
#     return model, tokenizer, device
#

# model, tokenizer, device = init_model_my()
model.eval()
prof =None
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
    if profile: prof.stop() # 嘻嘻嘻
    return f'{output_text[0]}|{str(speech_res[0].tolist()[1:])}'


def true_decode_fuc(input_wav_path, input_prompt):
    # input_prompt = TASK_PROMPT_MAPPING.get(input_prompt, "未知任务类型")
    print(f"wav_path: {input_wav_path}, prompt:{input_prompt}")
    if input_wav_path is None:
        print("音频信息未输入，可能是tts任务")
    if input_prompt.endswith("_TTS"):
        text_for_tts = input_prompt.replace("_TTS", "")
        input_prompt = "恳请将如下文本转换为其对应的语音token，力求生成最为流畅、自然的语音。"
        # res_text = model.generate_tts(device=device, text=text_for_tts, prompt=input_prompt)[0]
        res_text = do_t2s(input_prompt, text_for_tts)
    elif input_prompt.endswith("_self_prompt"):
        input_prompt = input_prompt.replace("_self_prompt", "")
        # feat, feat_lens = get_feat_from_wav_path(input_wav_path)
        # res_text = model.generate(wavs=feat, wavs_len=feat_lens, prompt=input_prompt)[0]
        res_text = do_s2t(input_wav_path, input_prompt)
    elif input_prompt == "先根据语音输入，直接以文字形式进行回答或对话，接着再生成语音token。" or input_prompt=="请先判断回复的情绪，并输出情感标签，然后以聊天的方式对以下音频获取相应的回答文本，之后把该回答文本转换为对应的带情感的语音token，旨在生成流畅自然的语音,情感共分为8类：sad，anger，neutral，happy，surprise，fear，disgust，sorry。":
        """"""
        # feat, feat_lens = get_feat_from_wav_path(input_wav_path)
        # res_text = model.generate_s2s(wavs=feat, wavs_len=feat_lens, prompt=input_prompt)[0]
        res_text = do_s2s(input_wav_path, input_prompt)
    else:
        # feat, feat_lens = get_feat_from_wav_path(input_wav_path)
        # res_text = model.generate(wavs=feat, wavs_len=feat_lens, prompt=input_prompt)[0]
        res_text = do_s2t(input_wav_path, input_prompt)
        res_text = res_text.replace("<youth>", "<adult>").replace("<middle_age>", "<adult>").replace("<middle>", "<adult>")
    # res_text = "耿雪龙 哈哈哈"
    print("识别结果为：", res_text)
    return res_text


def do_decode(input_wav_path, input_prompt):
    print(f'input_wav_path= {input_wav_path}, input_prompt= {input_prompt}')
    # 省略处理逻辑
    output_res = true_decode_fuc(input_wav_path, input_prompt)
    # output_res = f"耿雪龙 哈哈哈, {input_prompt}, {input_wav_path}"
    return output_res


def save_to_jsonl(if_correct, wav, prompt, res):
    data = {
        "if_correct": if_correct,
        "wav": wav,
        "task": prompt,
        "res": res
    }
    with open("results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")



def download_audio(input_wav_path):
    if input_wav_path:
        # 返回文件路径供下载
        return input_wav_path
    else:
        return None


def get_wav_from_token_list(input_list):
    return token_list2wav(input_list)


# 创建Gradio界面
with gr.Blocks() as demo:
    # 添加标题
    gr.Markdown(
        f"""
        <div style="display: flex; align-items: center; justify-content: center; text-align: center;">
            <h1 style="font-family: 'Arial', sans-serif; color: #014377; font-size: 32px; margin-bottom: 0; display: inline-block; vertical-align: middle;">
                OSUM Speech Understanding Model Test
            </h1>
        </div>
        """
    )

    # 添加音频输入和任务选择
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            # audio_input = gr.Audio(label="录音", type="filepath")
            audio_input = gr.Audio(label="录音", sources=["microphone", "upload"], type="filepath", visible=True)
        with gr.Column(scale=1, min_width=300):  # 给输出框设置最小宽度，确保等高对齐
            output_text = gr.Textbox(label="输出结果", lines=6, placeholder="生成的结果将显示在这里...",
                                     interactive=False)

    # 添加任务选择和自定义输入框
    with gr.Row():
        task_dropdown = gr.Dropdown(
            label="任务",
            choices=list(TASK_PROMPT_MAPPING.keys()) + ["自主输入文本", "TTS任务"],  # 新增"TTS任务"选项
            value="ASR (Automatic Speech Recognition)"
        )
        custom_prompt_input = gr.Textbox(label="自定义任务提示", placeholder="请输入自定义任务提示...",
                                         visible=False)  # 新增文本输入框
        tts_input = gr.Textbox(label="TTS输入文本", placeholder="请输入TTS任务的文本...", visible=False)  # 新增TTS输入框

    # 添加音频播放组件
    audio_player = gr.Audio(label="播放音频", type="filepath", interactive=False)

    # 添加按钮（下载按钮在左边，开始处理按钮在右边）
    with gr.Row():
        download_button = gr.DownloadButton("下载音频", variant="secondary",
                                            elem_classes=["button-height", "download-button"])
        submit_button = gr.Button("开始处理", variant="primary", elem_classes=["button-height", "submit-button"])

    # 添加确认组件
    with gr.Row(visible=False) as confirmation_row:
        gr.Markdown("请判断结果是否正确：")
        confirmation_buttons = gr.Radio(
            choices=["正确", "错误"],
            label="",
            interactive=True,
            container=False,
            elem_classes="confirmation-buttons"
        )
        save_button = gr.Button("提交反馈", variant="secondary")

    # 添加底部内容
    with gr.Row():
        # 底部内容容器
        with gr.Column(scale=1, min_width=800):  # 设置最小宽度以确保内容居中
            gr.Markdown(
                f"""
                <div style="position: fixed; bottom: 20px; left: 50%; transform: translateX(-50%); display: flex; align-items: center; justify-content: center; gap: 20px;">
                    <div style="text-align: center;">
                        <p style="margin: 0;"><strong>Audio, Speech and Language Processing Group (ASLP@NPU),</strong></p>
                        <p style="margin: 0;"><strong>Northwestern Polytechnical University</strong></p>
                    </div>
                    <img src="data:image/png;base64,{encoded_string}" alt="OSUM Logo" style="height: 80px; width: auto;">
                </div>
                """
            )

    # 绑定事件
    def show_confirmation(output_res, input_wav_path, input_prompt):
        return gr.update(visible=True), output_res, input_wav_path, input_prompt

    def save_result(if_correct, wav, prompt, res):
        save_to_jsonl(if_correct, wav, prompt, res)
        return gr.update(visible=False)

    def handle_submit(input_wav_path, task_choice, custom_prompt, tts_text):
        if task_choice == "自主输入文本":
            input_prompt = custom_prompt + "_self_prompt"  # 自定义输入加上"_self_prompt"
        elif task_choice == "TTS任务":
            input_prompt = tts_text + "_TTS"  # TTS输入加上"_TTS"
        else:
            input_prompt = TASK_PROMPT_MAPPING.get(task_choice, "未知任务类型")  # 使用预定义的提示
        output_res = do_decode(input_wav_path, input_prompt)
        wav_path_output = input_wav_path
        if task_choice == "TTS任务":
            assert isinstance(output_res, list), "TTS任务的输出必须为列表"
            wav_path_output = get_wav_from_token_list(output_res)
            output_res = "生成的token: "+str(output_res)
        if task_choice == "<S2TCHAT> <TEXT2TOKEN>":
            text_res, token_list_str = output_res.split("|")
            # 将字符串的list转成list
            token_list = json.loads(token_list_str)
            # 下载音频
            wav_path_output = get_wav_from_token_list(token_list)
            output_res = text_res
        return output_res, wav_path_output

    task_dropdown.change(
        fn=lambda choice: gr.update(visible=choice == "自主输入文本"),
        inputs=task_dropdown,
        outputs=custom_prompt_input
    )

    task_dropdown.change(
        fn=lambda choice: gr.update(visible=choice == "TTS任务"),
        inputs=task_dropdown,
        outputs=tts_input
    )

    submit_button.click(
        fn=handle_submit,
        inputs=[audio_input, task_dropdown, custom_prompt_input, tts_input],
        outputs=[output_text, audio_player]
    ).then(
        fn=show_confirmation,
        inputs=[output_text, audio_input, task_dropdown],
        outputs=[confirmation_row, output_text, audio_input, task_dropdown]
    )

    download_button.click(
        fn=download_audio,
        inputs=[audio_input],
        outputs=[download_button]  # 输出到 download_button
    )

    save_button.click(
        fn=save_result,
        inputs=[confirmation_buttons, audio_input, task_dropdown, output_text],
        outputs=confirmation_row
    )

demo.launch(server_name="0.0.0.0", server_port=7860)

print("start warmup...")
input_wav_path = "/mnt/sfs/asr/update_data/align_cn_noize/roobo_100097437.wav"
input_prompt = "将这段音频的语音内容详细记录为文字稿。"
res_text = do_s2t(input_wav_path, input_prompt, profile=False)
print(f'ASR推理结果: {res_text}')
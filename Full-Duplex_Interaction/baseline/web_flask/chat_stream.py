import argparse
import os
import json
import re
import queue
import torch
import yaml
import threading
import struct
import time
import torchaudio
import datetime
import builtins
import numpy as np
import logging

from copy import deepcopy
from threading import Timer
from scipy.io.wavfile import write as wav_write
from flask import Flask, render_template, request
from flask_socketio import SocketIO, disconnect, emit
from wenet.llm_asr.streamers import TokenIdStreamer

from web_flask.parms import GlobalParams
from web_flask.modules import IPTObjectPool, SLMObjectPool, TTSObjectPool
from web_flask.pem import generate_self_signed_cert

MAX_USERS=None
TIMEOUT=None
IPT_POOL=None
SLM_POOL=None
TTS_POOL=None
SPK_DIR="./f5_tts/infer/prompt"
EMO_LST=["NEUTRAL", "ANGER", "DISGUST", "FEAR", "HAPPY", "SAD", "SURPRISED"]
SR_IN=16000
SR_OUT=24000
APP = Flask(__name__, template_folder='../web_flask/resources')
SOCKET_IO = SocketIO(APP)
CONNECTED_USERS = {}

def get_args():
    parser = argparse.ArgumentParser(description='OSUM')
    parser.add_argument('--ipt_model_path', 
                        default="/home/work_nfs11/gjli/ckpt/wenet_undersdand_and_speech/interrupt_stage1_asr_task4_0.5b_4.14/epoch_1.pt", 
                        help='interrupt path to load')
    parser.add_argument('--ipt_config_path',
                        default="/home/work_nfs16/gjli/workspaces/wenet_speech_interrupt/examples/wenetspeech/whisper/conf/finetune_whisper_medium_gxl_adapter_interrupt_infer.yaml", 
                        help='interrupt config path to load')
    parser.add_argument('--slm_model_path', 
                        default="/home/work_nfs16/asr_data/ckpt/understand_model_3B/full_train_llm_3B_epoch_2/step_4999.pt", 
                        help='speech llm path to load')
    parser.add_argument('--slm_config_path',
                        default="conf/config_llm_huawei_instruct_3B_cosyvoice1-token.yaml", 
                        help='speech llm config path to load')
    parser.add_argument('--tts_fm_model_path', 
                        default="/home/node57_data/hkxie/4O/F5-TTS/ckpts/F5TTS_fm_10ms_ecapa_tdnn_spk_hifigancosyvoice1/model_1000000.pt", 
                        help='tts flowmatching model path to load')
    parser.add_argument('--tts_fm_config_path', 
                        default="conf/fm_10ms_cosyvoice1_ecaptdnn_spk.yaml", 
                        help='tts flowmatching config to load')
    parser.add_argument('--tts_fm_turns',
                        default=10,
                        type=int,
                        help='tts flowmatching model turns')
    parser.add_argument('--tts_vc_model_path', 
                        default="/home/node57_data/hkxie/4O/F5-TTS/src/third_party/hifigan/ckpt_hifigan/g_00400000", 
                        help='tts vocder model to load')
    parser.add_argument('--tts_vc_config_path', 
                        default="./conf/config_streamfm10ms.json", 
                        help='tts vocder config to load')
    parser.add_argument('--vocoder',
                    choices=['bigvgan', 'hifigan'],  # 限定两种选项
                    default='hifigan',               # 默认使用HiFiGAN
                    help='TTS vocoder selection (choose between "bigvgan" or "hifigan")')
    parser.add_argument('--ip', 
                        default="127.0.0.1", 
                        help='ip of server')
    parser.add_argument('--port', 
                        default='5000', 
                        help='port of server')
    parser.add_argument('--max_users', 
                        type=int, 
                        default=1)
    parser.add_argument('--timeout', 
                        type=int, 
                        default=300)
    args = parser.parse_args()
    print(args)
    return args

def save_wav(wav, sample_rate=SR_IN):
    wav_name = str(int(time.time())) + ".wav"
    path = os.path.join("temp", wav_name)
    os.makedirs("temp", exist_ok=True)
    wav_write(path, sample_rate, wav.astype(np.int16))
    return

def tts_generate(sid, token, emotion, end, vocoder):
    if CONNECTED_USERS[sid][1].stop_generate:
        SOCKET_IO.emit('stop_tts', to=sid)
        CONNECTED_USERS[sid][1].output_audio_queue.clear()
        CONNECTED_USERS[sid][1].tts_obj.reset()
        return
    
    print("################ emotion ", emotion)
    if emotion not in EMO_LST:
        emotion = "NEUTRAL"
    speaker = CONNECTED_USERS[sid][1].speaker
    prompt_dir = os.path.join(SPK_DIR, speaker)
    prompt_wav = emotion + ".wav"
    if not os.path.exists(os.path.join(prompt_dir, prompt_wav)):
        prompt_wav = "NEUTRAL.wav"
    print("################ prompt_wav", os.path.join(prompt_dir, prompt_wav))

    if not end:
        speech_token = torch.tensor(token, 
                                dtype=torch.int64, 
                                device=CONNECTED_USERS[sid][1].tts_obj.fm_model.device,
                                ).unsqueeze(0)
        sample_rate, speech_wav = CONNECTED_USERS[sid][1].tts_obj.infer_stream(
                                            speech_token, 
                                            wav_type="numpy",
                                            prompt_dir = prompt_dir,
                                            prompt_wav = prompt_wav, 
                                            output_sample_rate=SR_OUT,
                                            end = False,
                                            vocoder=vocoder,
                                            )
    else:
        token = token + [288] * (12 - len(token))
        speech_token = torch.tensor(token, 
                                dtype=torch.int64, 
                                device=CONNECTED_USERS[sid][1].tts_obj.fm_model.device,
                                ).unsqueeze(0)
        sample_rate, speech_wav = CONNECTED_USERS[sid][1].tts_obj.infer_stream(
                                            speech_token, 
                                            wav_type="numpy",
                                            prompt_dir = prompt_dir,
                                            prompt_wav = prompt_wav, 
                                            output_sample_rate=SR_OUT,
                                            end = True,
                                            vocoder=vocoder,
                                            )
        CONNECTED_USERS[sid][1].tts_obj.reset()

    if not CONNECTED_USERS[sid][1].stop_generate:
        CONNECTED_USERS[sid][1].output_audio_queue.put(speech_wav)
    else:
        SOCKET_IO.emit('stop_tts', to=sid)
        CONNECTED_USERS[sid][1].output_audio_queue.clear()
        CONNECTED_USERS[sid][1].tts_obj.reset()
        return 


# def slm_generate(data, sid):
#     # 直接使用固定 token_seq 替代 streamer
#     # token_seq = [1879, 21, 3250, 20, 1289, 2164, 492, 2610, 106, 421, 2893, 3147, 2597, 1117, 1117, 1117, 1446, 1446, 1446, 1446, 26, 2166, 773, 1204, 2000, 2858, 1730, 2, 432, 2764, 2058, 409, 1755, 
#     #             2554, 2554, 33, 1516, 2105, 40, 1584, 700, 2110, 1073, 701, 4082, 1655, 2062, 2247, 532, 2548, 3293, 1403, 22, 2399, 1037, 962, 3740, 4064, 432, 1784, 2664, 41, 1615, 73, 2243, 2714, 202, 
#     #             1193, 1193, 3573, 1027, 1446, 1446, 3106, 3898, 53, 53, 1425, 2841, 966, 4016, 1961, 2293, 511, 802, 802, 3062, 569, 1342, 1600, 3612, 1608, 2250, 2000, 203, 455, 463, 1915, 1821, 891, 773, 773, 
#     #             629, 1660, 3106, 2323, 2250, 1516, 3554, 754, 1346, 303, 1018, 165, 2287, 661, 2306, 4064, 962, 41, 3740, 569, 3947, 1098, 1346, 2554, 2554, 2554, 2554, 368, 1660, 984, 1117, 1446, 1504, 1504, 3347, 1446, 4096]
#         # 选择vocoder
#     vocoder = CONNECTED_USERS[sid][1].vocoder
#     token_seq = [1155, 3614, 1122, 2392, 2735, 1949, 3710, 3684, 3347, 2429, 1655, 1655 ,  3929, 36, 1385, 2230, 3898, 3898, 3898, 53, 53, 3649, 1412, 124, 3649, 1615, 28, 1193, 343, 325,325, 1600, 3250, 20, 1507, 109, 4056, 2247, 980, 1941, 1896, 2504,
#                  2515, 1030, 3121, 66, 3573, 1241, 62, 1796, 1660, 463, 1567, 1098,1346, 1346, 1615, 3649, 20, 1729, 1555, 58, 1403, 1240, 3251, 24,62, 1223, 1223, 229, 229, 3551, 26, 1446, 3898, 3898, 3898, 53,
#                  53, 53, 2684, 484, 472, 1650, 69, 3818, 3818, 1346, 44, 193,2815, 69, 3887, 3238, 1567, 84, 568, 3068, 1516, 2409, 518, 2569,2935, 2148, 1513, 700, 870, 43, 3147, 20, 1507, 2209, 1333, 2844,
#                  2172, 3489, 386, 3121, 518, 3238, 690, 73, 1785, 1883, 1579, 1602,1256, 1735, 1256, 1256, 1218, 1879, 1446, 36, 1446, 1879, 1027, 3898,53, 53, 53, 2287, 870, 1122, 1470, 1796, 2935, 2255, 3947, 2399,
#                  1712, 532, 1602, 1602, 1602, 1602, 532, 532, 49, 3929, 29, 3211]

#     buffer_tokens = []
#     first_yield = True
#     first_speech = True
#     first_speech_eos = True
#     text_eos_id = 151645
#     speech_eos_id = CONNECTED_USERS[sid][1].slm_obj.model.speech_token_num - 1 #4096
#     tokenizer = CONNECTED_USERS[sid][1].slm_obj.model.tokenizer
#     CONNECTED_USERS[sid][1].tts_obj.reset()
#     CONNECTED_USERS[sid][1].stop_generate = False
#     emotion = 'NEUTRAL'
#     emotion = 'HAPPY'
    
#     # 用固定 token_seq 替代 streamer 的 yield
#     for token_id in token_seq:
#         if CONNECTED_USERS[sid][1].stop_generate:
#             SOCKET_IO.emit('stop_tts', to=sid)
#             CONNECTED_USERS[sid][1].output_audio_queue.clear()
#             return

#         buffer_tokens.append(token_id)

#         if first_speech:
#             while len(buffer_tokens) >= 18 or (speech_eos_id in buffer_tokens):
#                 if speech_eos_id not in buffer_tokens:
#                     print("first 18:", buffer_tokens[:18])
#                     tts_generate(sid, buffer_tokens[:18], emotion, False, vocoder)
#                     buffer_tokens = buffer_tokens[18:]
#                     first_speech = False
#                 else:
#                     idx = buffer_tokens.index(speech_eos_id)
#                     if first_speech_eos:
#                         buffer_tokens = buffer_tokens[idx+1:]
#                         first_speech_eos = False
#                         break
#                     else:
#                         print("first less 18", buffer_tokens[:idx])
#                         token = buffer_tokens[:idx]
#                         token = token + [288] * (18 - len(token))
#                         tts_generate(sid, token, emotion, False, vocoder)
#                         buffer_tokens = []
#                         first_speech = False
#                         return
#         else:
#             while len(buffer_tokens) >= 12 or (speech_eos_id in buffer_tokens):
#                 if speech_eos_id not in buffer_tokens:
#                     print("mid 12:", buffer_tokens[:12])
#                     tts_generate(sid, buffer_tokens[:12], emotion, False, vocoder)
#                     buffer_tokens = buffer_tokens[12:]
#                 else:
#                     idx = buffer_tokens.index(speech_eos_id)
#                     print("end less 12", buffer_tokens[:idx])
#                     tts_generate(sid, buffer_tokens[:idx], emotion, True, vocoder)
#                     buffer_tokens = []
#                     return

def slm_generate(data, sid):
    print("-------调试是否进入slm模型--------")
    # 选择vocoder
    vocoder = CONNECTED_USERS[sid][1].vocoder
    # try:
    kwargs = CONNECTED_USERS[sid][1].slm_obj.infer_s2s(
                                    (SR_IN, data['history_feature']),
                                    wav_type="numpy",
                                    )
    streamer = TokenIdStreamer()
    kwargs["streamer"] = streamer
    decode_thread = threading.Thread(
        target=CONNECTED_USERS[sid][1].slm_obj.model.llama_model.generate,
        kwargs=kwargs,
        )
    
    # token_seq = [1155, 3614, 1122, 2392, 2735, 1949, 3710, 3684, 3347, 2429, 1655, 1655 ,  3929, 36, 1385, 2230, 3898, 3898, 3898, 53, 53, 3649, 1412, 124, 3649, 1615, 28, 1193, 343, 325,325, 1600, 3250, 20, 1507, 109, 4056, 2247, 980, 1941, 1896, 2504,
    #              2515, 1030, 3121, 66, 3573, 1241, 62, 1796, 1660, 463, 1567, 1098,1346, 1346, 1615, 3649, 20, 1729, 1555, 58, 1403, 1240, 3251, 24,62, 1223, 1223, 229, 229, 3551, 26, 1446, 3898, 3898, 3898, 53,
    #              53, 53, 2684, 484, 472, 1650, 69, 3818, 3818, 1346, 44, 193,2815, 69, 3887, 3238, 1567, 84, 568, 3068, 1516, 2409, 518, 2569,2935, 2148, 1513, 700, 870, 43, 3147, 20, 1507, 2209, 1333, 2844,
    #              2172, 3489, 386, 3121, 518, 3238, 690, 73, 1785, 1883, 1579, 1602,1256, 1735, 1256, 1256, 1218, 1879, 1446, 36, 1446, 1879, 1027, 3898,53, 53, 53, 2287, 870, 1122, 1470, 1796, 2935, 2255, 3947, 2399,
    #              1712, 532, 1602, 1602, 1602, 1602, 532, 532, 49, 3929, 29, 3211]

    buffer_tokens = []
    first_yield = True
    first_speech = True
    first_speech_eos = True
    text_eos_id = 151645
    speech_eos_id = CONNECTED_USERS[sid][1].slm_obj.model.speech_token_num - 1
    tokenizer = CONNECTED_USERS[sid][1].slm_obj.model.tokenizer
    CONNECTED_USERS[sid][1].stop_generate = False
    CONNECTED_USERS[sid][1].slm_obj.model.interrupt.stop = False
    CONNECTED_USERS[sid][1].tts_obj.reset()

    decode_thread.start()
    for token_id in streamer:
        if CONNECTED_USERS[sid][1].stop_generate:
            SOCKET_IO.emit('stop_tts', to=sid)
            CONNECTED_USERS[sid][1].output_audio_queue.clear()
            return
        buffer_tokens.append(token_id)
        if first_yield:
            if text_eos_id in buffer_tokens:
                idx = buffer_tokens.index(text_eos_id)
                output_text = tokenizer.batch_decode(buffer_tokens[:idx], add_special_tokens=False, skip_special_tokens=True)
                output_text = "".join(output_text)
                # 文本输出
                print(">>>>>>>>>> 回答 >>>>>>>>>>: ", output_text)
                match = re.findall(r'<([^>]+)>', output_text)
                if match != []:
                    emotion = match[288]
                else:
                    emotion = "NEUTRAL"

                buffer_tokens = buffer_tokens[idx+1:]
                first_yield = False
        elif first_speech:
            while len(buffer_tokens) >= 18 or (speech_eos_id in buffer_tokens):
                if not speech_eos_id in buffer_tokens:
                    print("first 18:", buffer_tokens[:18])
                    tts_generate(sid, buffer_tokens[:18], emotion, False, vocoder)
                    buffer_tokens = buffer_tokens[18:]
                    first_speech = False
                else:
                    idx = buffer_tokens.index(speech_eos_id)
                    if first_speech_eos:
                        # 第一次遇到speech_eos_id，直接跳过，不yield
                        buffer_tokens = buffer_tokens[idx+1:]
                        first_speech_eos = False
                        break
                    else: 
                        print("first less 18", buffer_tokens[:idx])
                        token = buffer_tokens[:idx]
                        token = token + [288] * (18 - len(token))
                        tts_generate(sid, token, emotion, False, vocoder)
                        buffer_tokens = []
                        first_speech = False
                        return
        else:
            while len(buffer_tokens) >= 12 or (speech_eos_id in buffer_tokens):
                if not speech_eos_id in buffer_tokens:
                    print("mid 12:", buffer_tokens[:12])
                    tts_generate(sid, buffer_tokens[:12], emotion, False, vocoder)
                    buffer_tokens = buffer_tokens[12:]
                else:
                    idx = buffer_tokens.index(speech_eos_id)
                    print("end less 12", buffer_tokens[:idx])
                    tts_generate(sid, buffer_tokens[:idx], emotion, True, vocoder)
                    buffer_tokens = []
                    return
    # except:
    #     print("音色切换bug")
    #     return

def send_pcm(sid):
    """
    Sends PCM audio data to the dialogue system for processing.

    Parameters:
    - sid (str): The session ID of the user.
    """

    chunk_szie = CONNECTED_USERS[sid][1].wakeup_and_vad.get_chunk_size()
    min_vad_num = 5

    print("Sid: ", sid, " Start listening")
    while True:
        if CONNECTED_USERS[sid][1].stop_pcm:
            print("Sid: ", sid, " Stop pcm")
            CONNECTED_USERS[sid][1].stop_generate = True
            break
        time.sleep(0.01)
        e = CONNECTED_USERS[sid][1].input_audio_queue.get(chunk_szie)
        if e is None:
            continue
        # print("Sid: ", sid, " Received PCM data: ", len(e))

        res = CONNECTED_USERS[sid][1].wakeup_and_vad.predict(np.float32(e))

        if res['status'] == 'sl':
            print("Sid: ", sid, " Vad start")
            interrupt_used = False
        elif res['status'] == 'cl':
            print("Sid: ", sid, " Vad continue")
            if CONNECTED_USERS[sid][1].wakeup_and_vad.history_num > min_vad_num and not interrupt_used and not CONNECTED_USERS[sid][1].stop_pcm: 
                
                ipt_res = CONNECTED_USERS[sid][1].ipt_obj.s2t_infer((SR_IN, res['history_feature']), wav_type="numpy")
                ipt_res = ipt_res.replace("<|endoftext|>", "")
                print(ipt_res)
                
                if "<backchannel>" in ipt_res and CONNECTED_USERS[sid][1].pending_audio is None:
                    SOCKET_IO.emit('interrupt_result', {'text': ipt_res + "|检测到用户backchannel，过滤掉"}, to=sid)
                    CONNECTED_USERS[sid][1].wakeup_and_vad.reset_vad()
                else:
                    interrupt_used = True
                    CONNECTED_USERS[sid][1].interrupt()
  
            elif CONNECTED_USERS[sid][1].wakeup_and_vad.history_num > min_vad_num and not interrupt_used:
                interrupt_used = True
                CONNECTED_USERS[sid][1].interrupt() 
        elif res['status'] == 'el':
            print("Sid: ", sid, " Vad stop")
            if interrupt_used:
                # CONNECTED_USERS[sid][1].output_audio_queue.put(res['history_feature'])
                save_wav(res['history_feature'])
                CONNECTED_USERS[sid][1].output_audio_queue.clear()
                SOCKET_IO.emit('stop_tts', to=sid)

                if CONNECTED_USERS[sid][1].pending_audio is not None:
                    full_audio = np.concatenate([CONNECTED_USERS[sid][1].pending_audio, res['history_feature']])
                    CONNECTED_USERS[sid][1].pending_audio = None
                else:
                    full_audio = res['history_feature']
    
                ipt_res = CONNECTED_USERS[sid][1].ipt_obj.s2t_infer((SR_IN, full_audio), wav_type="numpy")
                ipt_res = ipt_res.replace("<|endoftext|>", "")
                
                if "<incomplete>" in ipt_res:
                    CONNECTED_USERS[sid][1].pending_audio = full_audio
                    SOCKET_IO.emit('interrupt_result', {'text': ipt_res + "|检测到不完整内容，保存音频并继续聆听..."}, to=sid)
                    print("检测到不完整内容，保存音频并继续聆听...")
                elif "<complete>" in ipt_res:
                    SOCKET_IO.emit('interrupt_result', {'text': ipt_res + "|检测到完整内容，进行回答"}, to=sid)
                    print("检测到完整内容，进行回答")
                    res['history_feature'] = full_audio
                    CONNECTED_USERS[sid][1].generate_thread = threading.Thread(target=slm_generate, args=(res, sid))
                    CONNECTED_USERS[sid][1].generate_thread.start()
                elif "<backchannel>" in ipt_res:
                    SOCKET_IO.emit('interrupt_result', {'text': ipt_res + "|检测到用户backchannel，过滤掉"}, to=sid)
                    print("检测到backchannel，过滤掉")
                else:
                    SOCKET_IO.emit('interrupt_result', {'text': ipt_res + "|检测到未知状态"}, to=sid)
                    print("检测到未知状态")

def disconnect_user(sid):
    if sid in CONNECTED_USERS:
        print(f"Disconnecting user {sid} due to time out")
        SOCKET_IO.emit('out_time', to=sid) 
        CONNECTED_USERS[sid][0].cancel()
        CONNECTED_USERS[sid][1].interrupt()
        CONNECTED_USERS[sid][1].stop_pcm = True
        CONNECTED_USERS[sid][1].release()
        time.sleep(3)
        del CONNECTED_USERS[sid]

@APP.route('/')
def index():
    return render_template('demo.html')

@SOCKET_IO.on('connect')
def handle_connect():
    if len(CONNECTED_USERS) >= MAX_USERS:
        print('Too many users connected, disconnecting new user')
        emit('too_many_users')
        return

    sid = request.sid
    CONNECTED_USERS[sid] = []
    CONNECTED_USERS[sid].append(Timer(TIMEOUT, disconnect_user, [sid]))
    CONNECTED_USERS[sid].append(GlobalParams(IPT_POOL, SLM_POOL, TTS_POOL))
    CONNECTED_USERS[sid][1].vocoder = conf.vocoder
    CONNECTED_USERS[sid][0].start()
    pcm_thread = threading.Thread(target=send_pcm, args=(sid,))
    pcm_thread.start()
    IPT_POOL.print_info()
    SLM_POOL.print_info()
    TTS_POOL.print_info()
    print(f'User {sid} connected')

@SOCKET_IO.on('disconnect')
def handle_disconnect():
    time.sleep(3)
    sid = request.sid
    if sid in CONNECTED_USERS:
        CONNECTED_USERS[sid][0].cancel()
        CONNECTED_USERS[sid][1].interrupt()
        CONNECTED_USERS[sid][1].stop_pcm = True
        CONNECTED_USERS[sid][1].release()
        time.sleep(3)
        del CONNECTED_USERS[sid]
    IPT_POOL.print_info()
    SLM_POOL.print_info()
    TTS_POOL.print_info()
    print(f'User {sid} disconnected')

@SOCKET_IO.on('speaker-select')
def handle_speaker_select(speaker):
    sid = request.sid
    CONNECTED_USERS[sid][1].speaker=speaker

@SOCKET_IO.on('vocoder-select')
def handle_vocoder_select(vocoder):
    """Handles the vocoder selection event from the client."""
    sid = request.sid
    # if sid in CONNECTED_USERS:
    CONNECTED_USERS[sid][1].vocoder = vocoder

@SOCKET_IO.on('recording-started')
def handle_recording_started():
    sid = request.sid
    if sid in CONNECTED_USERS:
        SOCKET_IO.emit('stop_tts', to=sid)
        CONNECTED_USERS[sid][0].cancel()
        CONNECTED_USERS[sid][0] = Timer(TIMEOUT, disconnect_user, [sid])
        CONNECTED_USERS[sid][0].start()
        # CONNECTED_USERS[sid][1].interrupt()
        SOCKET_IO.emit('stop_tts', to=sid)
        CONNECTED_USERS[sid][1].reset()
    else:
        disconnect()
    print('Recording started')

@SOCKET_IO.on('recording-stopped')
def handle_recording_stopped():
    sid = request.sid
    if sid in CONNECTED_USERS:
        CONNECTED_USERS[sid][0].cancel()
        CONNECTED_USERS[sid][0] = Timer(TIMEOUT, disconnect_user, [sid])
        CONNECTED_USERS[sid][0].start()
        CONNECTED_USERS[sid][1].interrupt()
        SOCKET_IO.emit('stop_tts', to=sid)
        CONNECTED_USERS[sid][1].reset()
    else:
        disconnect()
    print('Recording stopped')

@SOCKET_IO.on('interrupt')
def handle_interrupt():
    sid = request.sid
    if sid in CONNECTED_USERS:
        CONNECTED_USERS[sid][1].interrupt()
    else:
        disconnect()
    print('interrupt')

@SOCKET_IO.on('audio')
def handle_audio(data):
    sid = request.sid
    if sid in CONNECTED_USERS:
        if not CONNECTED_USERS[sid][1].output_audio_queue.is_empty():
            CONNECTED_USERS[sid][0].cancel()
            CONNECTED_USERS[sid][0] = Timer(TIMEOUT, disconnect_user, [sid])
            CONNECTED_USERS[sid][0].start()
            output_data = CONNECTED_USERS[sid][1].output_audio_queue.get()
            if output_data is not None:
                print("Sid: ", sid, "Send TTS data")
                emit('audio', output_data.astype(np.int16).tobytes())

        data = json.loads(data)
        audio_data = np.frombuffer(bytes(data['audio']), dtype=np.int16)
        sample_rate = data['sample_rate']
        
        CONNECTED_USERS[sid][1].input_audio_queue.put(audio_data.astype(np.float32) / 32768.0)
    else:
        disconnect()

if __name__ == "__main__":
    print("Start OSUM sever") 
    
    # init parms
    conf = get_args()
    MAX_USERS = conf.max_users
    TIMEOUT = conf.timeout
    # init inference pool
    IPT_POOL = IPTObjectPool(
                            size=conf.max_users,
                            configs=conf,
                            )
    SLM_POOL = SLMObjectPool(
                            size=conf.max_users,
                            configs=conf,
                            )
    TTS_POOL = TTSObjectPool(
                            size=conf.max_users,
                            configs=conf,
                            )      
    cert_file = "web_flask/resources/cert.pem"
    key_file = "web_flask/resources/key.pem"
    if not os.path.exists(cert_file) or not os.path.exists(key_file):
        generate_self_signed_cert(cert_file, key_file)
    #SOCKET_IO.run(APP, host=conf.ip, port=conf.port, ssl_context=(cert_file, key_file))
    SOCKET_IO.run(APP, host=conf.ip, port=conf.port)
import queue
import torch
import threading
import time
import numpy as np
from copy import deepcopy
from threading import Timer
from web_flask.modules import PCMQueue, ThreadSafeQueue, BatchQueue, VAD

class GlobalParams:
    def __init__(self, ipt_pool, slm_pool, tts_pool):
        """
        Initialize the GlobalParams class with necessary components for managing global parameters and states.

        Parameters:
        - tts_pool: Pool of speech decoder.
        - slm_pool: Pool of speech llm.

        Returns:
        - None
        """
        self.ipt_pool = ipt_pool
        self.ipt_obj = self.ipt_pool.acquire()
        self.slm_pool = slm_pool
        self.slm_obj = self.slm_pool.acquire()
        self.tts_pool = tts_pool
        self.tts_obj = self.tts_pool.acquire()
        self.pending_audio = None
        # # init default prompt
        # init_outputs = self.slm_obj.infer_no_stream(None, 
        #                                                 stat='pre', 
        #                                                 role='You are a helpful voice assistant.\
        #                                                         Your answer should be coherent, natural, simple, complete.\
        #                                                         Your name is Xiao Yun.\
        #                                                         Your inventor is Tencent.')
        # self.system_role = deepcopy(init_outputs)
        self.wakeup_and_vad = VAD()
        self.speaker = "女性声优"
        self.reset()
    
    def set_prompt(self, prompt):
        self.system_role = self.slm_obj.infer_no_stream(None, stat='pre', role=prompt)

    def reset(self):
        self.stop_generate = False
        self.slm_obj.model.interrupt.stop = False
        self.generate_thread = None
        self.wakeup_and_vad.in_dialog = False
        self.stop_pcm = False
        # self.generate_outputs = deepcopy(self.system_role)
        self.output_audio_queue = ThreadSafeQueue()
        self.input_audio_queue = PCMQueue()
        # self.speech_token_buffer = BatchQueue()
        self.pending_audio = None
    
    def interrupt(self):
        self.stop_generate = True
        self.slm_obj.model.interrupt.stop = True
        if self.generate_thread is not None:
            self.generate_thread.join()
            self.generate_thread = None
            while True:
                self.output_audio_queue.clear()
                time.sleep(0.01)
                if self.output_audio_queue.is_empty():
                    print("############## success interrupt ##############")
                    break
        
    def release(self):
        self.ipt_pool.release(self.ipt_obj)
        self.slm_pool.release(self.slm_obj)
        self.tts_pool.release(self.tts_obj)

    def print(self):
        print("stop_generate:", self.stop_generate)
        print("in_dialog", self.wakeup_and_vad.in_dialog)
        print("stop_pcm", self.stop_pcm)

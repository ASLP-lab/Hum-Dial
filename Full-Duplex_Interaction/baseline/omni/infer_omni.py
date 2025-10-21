import soundfile as sf
import torch
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info
import os
from vocoder import Qwen2_5OmniToken2WavBigVGANModel,Qwen2_5OmniBigVGANConfig
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# default: Load the model on the available device(s)
# model = Qwen2_5OmniForConditionalGeneration.from_pretrained("/home/work_nfs11/asr_data/ckpt/Qwen2.5-Omni-3B", torch_dtype=torch.float16, device_map="auto")

# We recommend enabling flash_attention_2 for better acceleration and memory saving.
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "/home/work_nfs11/asr_data/ckpt/Qwen2.5-Omni-3B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)

print(model)  # 或者
for name, module in model.named_modules():
    print(name)

import pdb;pdb.set_trace()

code2wav_bigvgan_model = Qwen2_5OmniToken2WavBigVGANModel(
    Qwen2_5OmniBigVGANConfig()
)


# Qwen2_5OmniToken2WavBigVGANModel = torch.load("",type=torch.bfloat16)

code2wav_bigvgan_model.load_state_dict()













processor = Qwen2_5OmniProcessor.from_pretrained("/home/work_nfs11/asr_data/ckpt/Qwen2.5-Omni-3B")

conversation = [
    {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-Omni/draw.mp4"},
        ],
    },
]

# set use audio in video
USE_AUDIO_IN_VIDEO = True

# Preparation for inference
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
inputs = inputs.to(model.device).to(model.dtype)

# Inference: Generation of the output text and audio
text_ids, audio = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO)

text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(text)
sf.write(
    "output.wav",
    audio.reshape(-1).detach().cpu().numpy(),
    samplerate=24000,
)
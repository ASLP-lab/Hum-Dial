import os
import json
import argparse
import re
import glob

import soundfile as sf
import nemo.collections.asr as nemo_asr
from tqdm import tqdm
import torch

def get_time_aligned_transcription(data_path, gpu_id=0):
    # 设置使用的GPU
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
   # 定义文件名匹配规则
    pattern = re.compile(
        r'^(clean_)?\d{4}_\d{4}'
        r'(?:_add|_before)?'        #  <-- 这里允许 _add 或 _before 或什么都没有
        r'(_output)?\.wav$'
    )

    # 找到符合条件的 wav 文件
    audio_paths = []
    for file_path in glob.glob(os.path.join(data_path, "*.wav")):
        file_name = os.path.basename(file_path)
        if pattern.match(file_name):
            audio_paths.append(file_path)
    audio_paths.sort()
    
    if not audio_paths:
        print(f"[WARN] No matching .wav files found in {data_path}")
        return

    # 加载 NeMo ASR 模型并移动到指定GPU
    asr_model = nemo_asr.models.ASRModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-0.6b-v2"
    ).to(device)

    for audio_path in tqdm(audio_paths):
        print(f"[INFO] Processing {audio_path}")
        # 读音频
        waveform, sr = sf.read(audio_path)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, waveform, sr)
            asr_outputs = asr_model.transcribe([tmp.name], timestamps=True)
        os.unlink(tmp.name)

        # 结果
        result = asr_outputs[0]
        word_timestamps = result.timestamp["word"]

        chunks = []
        text = ""
        for w in word_timestamps:
            start_time = w["start"]
            end_time = w["end"]
            word = w["word"]

            text += word + " "
            chunks.append(
                {
                    "text": word,
                    "timestamp": [start_time, end_time],
                }
            )

        output_dict = {
            "text": text.strip(),
            "chunks": chunks,
        }

        # 输出 JSON 文件，和音频同目录同名
        result_path = os.path.splitext(audio_path)[0] + ".json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(output_dict, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Saved result to {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe specific wav files with timestamps"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/home/work_nfs17/gjli/full-duplex-data/en_dev/talk_to_others_process_moshi_eval",
        help="Folder containing .wav files",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=6,
        help="GPU ID to use (default: 0)"
    )
    args = parser.parse_args()

    get_time_aligned_transcription(args.root_dir, args.gpu)
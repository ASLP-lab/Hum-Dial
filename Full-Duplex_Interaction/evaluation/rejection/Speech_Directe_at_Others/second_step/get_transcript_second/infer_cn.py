import os
import json
import argparse
import re
import glob
import soundfile as sf
from tqdm import tqdm
from funasr import AutoModel

def get_time_aligned_transcription(data_path, gpu_id=0):
    """
    获取指定路径下所有匹配命名规则的音频文件路径：
      - 1234_5678.wav
      - 1234_5678_add.wav
      - clean_1234_5678.wav
      - clean_1234_5678_add_output.wav
    """
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

    # 加载 FunASR 模型
    model = AutoModel(
        model="paraformer-zh",
        model_revision="v2.0.4",
        vad_model="fsmn-vad",
        vad_model_revision="v2.0.4"
    )

    for audio_path in tqdm(audio_paths):
        print(f"[INFO] Processing {audio_path}")
        # 读音频（转单声道）
        waveform, sr = sf.read(audio_path)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        # 推理
        res = model.generate(
            input=audio_path,
            batch_size_s=300
        )

        text_raw = res[0]["text"]                  # 原始输出 (带空格)
        timestamps = res[0]["timestamp"]           # 每个 token 的时间戳
        #text_no_space = text_raw.replace(" ", "")  # 去空格文本
        tokens = text_raw.split()                  # token list

        # FunASR 支持标点输出
        #text_with_punc = res[0].get("text_with_punc", text_no_space)

        # ====== 构造 JSON ======
        output_dict = {
            "text": text_raw,
            "chunks": [
                {
                    "text": tok,
                    "timestamp": [start / 1000, end / 1000]
                }
                for tok, (start, end) in zip(tokens, timestamps)
            ]
        }

        # 输出 JSON 文件（与音频同名）
        result_path = os.path.splitext(audio_path)[0] + ".json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(output_dict, f, ensure_ascii=False, indent=4)
        print(f"[INFO] Saved result to {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe wav files with FunASR and save JSON (text + timestamps)"
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="/home/work_nfs17/gjli/full-duplex-data/en_dev/talk_to_others_process_osum",
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

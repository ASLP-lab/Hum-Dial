import json
import shutil
from pathlib import Path
from pydub import AudioSegment

def process_audio_folder(input_dir: Path, output_dir: Path):
    """
    批量处理音频文件夹：
    1. xxxx_xxxx.wav：删掉 speech_segments[0].xmin 之前的音频
    2. xxxx_xxxx_output.wav：将 speech_segments[0] 段置静音，并删掉 xmin 之前的部分
    3. 将处理结果及 clean_*.json 一并保存到 output_dir
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for sentence_json in input_dir.glob("*_sentence.json"):
        base_name = sentence_json.stem.replace("_sentence", "")
        wav_path = input_dir / f"{base_name}.wav"
        out_wav_path = input_dir / f"{base_name}_output.wav"
        clean_json = input_dir / f"clean_{base_name}.json"
        clean_output_json = input_dir / f"clean_{base_name}_output.json"

        # 检查文件是否齐全
        if not (wav_path.exists() and out_wav_path.exists() and clean_json.exists() and clean_output_json.exists()):
            print(f"⚠️ 缺少相关文件，跳过: {base_name}")
            continue

        # 读取 sentence.json
        with open(sentence_json, "r", encoding="utf-8") as f:
            meta = json.load(f)
        speech_segments = meta.get("speech_segments", [])
        if len(speech_segments) == 0:
            print(f"⚠️ speech_segments为空，跳过: {base_name}")
            continue

        first_seg = speech_segments[1]
        xmin = float(first_seg["xmin"])
        xmax = float(first_seg["xmax"])

        # ========== 处理 xxxx_xxxx.wav ==========
        audio = AudioSegment.from_wav(wav_path)
        cut_start = int(xmin * 1000)
        processed_audio = audio[cut_start:]
        out_wav_new = output_dir / wav_path.name
        processed_audio.export(out_wav_new, format="wav")

        # ========== 处理 xxxx_xxxx_output.wav ==========
        audio_out = AudioSegment.from_wav(out_wav_path)
        silent_part = AudioSegment.silent(duration=int((xmax - xmin) * 1000))
        before = audio_out[:int(xmin * 1000)]
        after = audio_out[int(xmax * 1000):]
        # 替换为静音
        audio_silenced = before + silent_part + after
        # 删除 xmin 前的部分
        audio_silenced = audio_silenced[int(xmin * 1000):]
        out_wav_output_new = output_dir / out_wav_path.name
        audio_silenced.export(out_wav_output_new, format="wav")

        # ========== 复制 clean_xxxx_xxxx.json 文件 ==========
        shutil.copy2(clean_json, output_dir / clean_json.name)
        shutil.copy2(clean_output_json, output_dir / clean_output_json.name)

        print(f"✅ 处理完成: {base_name}")

if __name__ == "__main__":
    input_dir = Path("./dev/Speech_Directe_at_Others")
    output_dir = Path("./dev/Speech_Directe_at_Others_for_eval")
    process_audio_folder(input_dir, output_dir)

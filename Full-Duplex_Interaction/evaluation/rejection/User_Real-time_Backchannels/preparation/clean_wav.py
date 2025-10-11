import os
import json
import soundfile as sf
import numpy as np
from pathlib import Path
from tqdm import tqdm

def process_wav_file(wav_path: Path):
    """
    处理单个WAV文件：只保留第一个语音段，其他部分置为静音
    
    Args:
        wav_path (Path): WAV文件路径
    """
    # 获取文件名（不含扩展名）
    file_stem = wav_path.stem
    
    # 检查文件名是否包含字母
    #if any(char.isalpha() for char in file_stem):
        #print(f"跳过包含字母的文件名: {wav_path}")
        #return
    
    # 获取对应的JSON文件路径（音频名_origin.json）
    json_path = wav_path.with_name(f"{file_stem}_sentence.json")
    
    # 检查JSON文件是否存在
    if not json_path.exists():
        print(f"JSON文件不存在: {json_path}")
        return
    
    try:
        # 读取音频文件
        data, sr = sf.read(wav_path)
        if data.ndim == 2:
            data = data.mean(axis=1)
        
        # 读取JSON文件
        with open(json_path, "r") as f:
            json_data = json.load(f)
        
        # 获取总时长和语音段信息
        final_duration = json_data.get("final_duration", 0)
        speech_segments = json_data.get("speech_segments", [])
        
        # 计算总样本数
        total_samples = int(round(final_duration * sr))
        
        # 创建全零数组（静音）
        clean_data = np.zeros(total_samples, dtype=data.dtype)
        
        # 如果有语音段，保留第一个语音段
        if speech_segments:
            first_segment = speech_segments[0]
            xmin = first_segment.get("xmin", 0)
            xmax = first_segment.get("xmax", 0)
            
            # 转换为样本索引
            start_idx = int(round(xmin * sr))
            end_idx = int(round(xmax * sr))
            
            # 确保索引在范围内
            start_idx = max(0, min(start_idx, len(data) - 1))
            end_idx = max(0, min(end_idx, len(data) - 1))
            
            # 复制第一个语音段的数据
            segment_length = end_idx - start_idx
            if segment_length > 0:
                clean_data[start_idx:end_idx] = data[start_idx:end_idx]
        
        # 创建输出文件名（clean_原始文件名）
        output_path = wav_path.with_name(f"clean_{wav_path.name}")
        
        # 保存处理后的音频
        sf.write(output_path, clean_data, sr)
        print(f"已处理并保存: {output_path}")
        
    except Exception as e:
        print(f"处理文件 {wav_path} 时出错: {e}")

def process_directory(directory):
    """
    处理目录中的所有WAV文件
    
    Args:
        directory (str): 目录路径
    """
    dir_path = Path(directory)
    
    # 查找所有WAV文件
    wav_files = list(dir_path.rglob("*.wav"))
    
    print(f"在 {directory} 中找到 {len(wav_files)} 个WAV文件")
    
    # 处理每个文件
    for wav_path in tqdm(wav_files, desc="处理文件"):
        process_wav_file(wav_path)

if __name__ == "__main__":
    # 设置要处理的目录路径
    input_directory = "./dev/User_Real-time_Backchannels"  # 请替换为实际目录路径
    
    # 处理目录
    process_directory(input_directory)
    print("处理完成！")
import json
import os
import re
import time
from collections import Counter
from typing import Dict, Any, Union, List, Tuple
from openai import OpenAI

# 设置DeepSeek API
DEEPSEEK_API_KEY = "your_api_key"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"

def json_dict_to_compact_text(json_list):
    """
    Convert list of dicts to a compact plain text string with minimal spaces.
    """
    return json.dumps(json_list, separators=(",", ":"), ensure_ascii=False)

def extract_json(text: str, key: str = "behaviour"):
    """
    Extract JSON object from text containing the specified key.
    """
    decoder = json.JSONDecoder()
    pos = text.find("{")
    while pos != -1:
        try:
            obj, end = decoder.raw_decode(text, pos)
            if key in obj:
                return obj
            pos = text.find("{", end)
        except json.JSONDecodeError:
            pos = text.find("{", pos + 1)
    raise ValueError(f"No JSON object with key '{key}' found.")

def parse_eval(data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Parse evaluation result from JSON string or dict.
    """
    if isinstance(data, str):
        data = extract_json(data)
    elif not isinstance(data, dict):
        raise ValueError("Input must be a JSON string or dict.")
    return data

def eval_behavior(system_msg, user_msg, client, overlap=1):
    """
    Evaluate behavior using DeepSeek API.
    """
    finished = False
    seed = 1
    while not finished:
        try:
            # 使用DeepSeek API
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                stream=False
            )
            
            prediction = response.choices[0].message.content
            # 清理响应内容
            prediction = prediction.strip().replace("\n", "")
            print("Prediction:", prediction)
            
            result = parse_eval(prediction)
            finished = True
            return result

        except Exception as e:
            print(f"Error: {e}. Retrying...")
            seed += 1
            time.sleep(5)
            continue

def stats_by_axis(records):
    """
    Calculate statistics by behavior axis.
    """
    axes = {"C": Counter()}

    for rec in records:
        for tag in rec.get("behaviour", []):
            prefix = tag[0]
            axes[prefix][tag] += 1

    totals = {ax: sum(cnt.values()) for ax, cnt in axes.items()}
    ratios = {
        ax: {tag: count / totals[ax] for tag, count in cnt.items()}
        for ax, cnt in axes.items()
    }
    return axes, totals, ratios

def read_instruction(task):
    """
    Read instruction from file.
    """
    file_path = f"./evaluation/instruction/{task}.txt" #指定prompt文件（behavior.txt）
    with open(file_path, "r", encoding="utf-8") as f:
        instruction_text = f.read()
    return instruction_text


def get_file_group(file_name: str) -> Tuple[str, str]:
    """
    Extract file group and type from filename.
    Match patterns like:
      - xxxx_xxxx.json
      - xxxx_xxxx_add.json
      - clean_xxxx_xxxx.json
      - clean_xxxx_xxxx_add_output.json
    """
    pattern = r'^(clean_)?(\d{4}_\d{4}(?:_add)?)(_output)?\.json$'
    match = re.match(pattern, file_name)
    if match:
        prefix = match.group(2)           # 获取 xxxx_xxxx 或 xxxx_xxxx_add
        is_clean = bool(match.group(1))   # 是否是 clean 文件
        is_output = bool(match.group(3))  # 是否是 output 文件
        
        # 确定文件类型
        if is_clean and is_output:
            file_type = "output_clean"
        elif is_clean:
            file_type = "input_clean"
        elif is_output:
            file_type = "output_noisy"
        else:
            file_type = "input_noisy"
        
        return prefix, file_type
    return None, None


def eval_behavior_all(data_dir, client, task, output_dir):
    """
    Evaluate behavior for all file groups in the directory.
    """
    # 读取指令
    instruction = read_instruction(task)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 收集所有文件并按组分类
    file_groups = {}
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".json"):
            prefix, file_type = get_file_group(file_name)
            if prefix and file_type:
                if prefix not in file_groups:
                    file_groups[prefix] = {}
                file_groups[prefix][file_type] = os.path.join(data_dir, file_name)
    
    output_list = []
    
    # 处理每个文件组
    for prefix, files in file_groups.items():
        # 检查是否包含所有必要的文件
        required_files = ["input_clean", "input_noisy", "output_clean", "output_noisy"]
        if not all(ft in files for ft in required_files):
            print(f"Skipping group {prefix} due to missing files.")
            continue
        
        print(f"Processing group: {prefix}")
        
        # 读取所有文件
        data = {}
        for file_type, file_path in files.items():
            with open(file_path, "r") as f:
                data[file_type] = json.load(f)
        
        # 检查是否有重叠
        overlap = check_overlap(
            data["input_noisy"].get("chunks", []),
            data["output_noisy"].get("chunks", [])
        )
        
        # 准备输入文本
        final_input = {
            "input_clean": json_dict_to_compact_text(data["input_clean"]),
            "input_noisy": json_dict_to_compact_text(data["input_noisy"]),
            "output_clean": json_dict_to_compact_text(data["output_clean"]),
            "output_noisy": json_dict_to_compact_text(data["output_noisy"])
        }
        final_input_text = json.dumps(final_input, separators=(",", ":"), ensure_ascii=False)
        
        # 评估行为
        result = eval_behavior(
            system_msg=instruction,
            user_msg=final_input_text,
            client=client,
            overlap=overlap
        )
        
        print(f"Result for {prefix}: {result}")
        
        # 添加到输出列表
        output_list.append((prefix, result))
    
    # 保存所有结果到一个JSON文件
    output_path = os.path.join(output_dir, "User_Real-time_Backchannels_content_tags.json") #不同子任务的输出结果
    with open(output_path, "w", encoding="utf-8") as f:
        for prefix, result in output_list:
            # 创建新的条目格式
            entry = {
                "key": f"User_Real-time_Backchannels_{prefix}", #不同子任务的key
                "behaviour": result.get("behaviour", [])
            }
            # 写入一行JSON
            json_line = json.dumps(entry, ensure_ascii=False)
            f.write(json_line + "\n")
    
    print(f"所有结果已保存到: {output_path}")
    
    # 计算统计信息
    counts, totals, ratios = stats_by_axis([r for _, r in output_list])
    fmt_ratios = {
        ax: {k: round(v, 2) for k, v in sorted(ratios[ax].items())} for ax in ["C"]
    }
    
    return fmt_ratios

def check_overlap(list_a: List[Dict], list_b: List[Dict]) -> int:
    """
    Check if there is overlap between two lists of time segments.
    """
    for seg_a in list_a:
        start_a, end_a = seg_a["timestamp"]
        for seg_b in list_b:
            start_b, end_b = seg_b["timestamp"]
            if max(start_a, start_b) < min(end_a, end_b):
                return 1
    return 0

if __name__ == "__main__":
    # 初始化DeepSeek客户端
    client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    
    # 设置参数
    data_dir = "./dev/User_Real-time_Backchannels"  # 输入：存放wav音频数据的目录
    output_dir = "./dev/json_group"  # 输出：User_Real-time_Backchannels_content_tags.json文件的保存目录
    task = "behavior"  # 任务名称，这里不用修改
    
    # 执行评估
    ratios = eval_behavior_all(data_dir, client, task, output_dir)
    print("Final ratios:", ratios)
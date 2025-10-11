import os
import json
import statistics

def process_pair(origin_file, output_file):
    with open(origin_file, "r", encoding="utf-8") as f:
        origin_data = json.load(f)
    with open(output_file, "r", encoding="utf-8") as f:
        output_data = json.load(f)

    # 检查 chunks 是否为空
    if not output_data.get("chunks"):
        return None

    # 获取时间
    first_chunk_start = output_data["chunks"][0]["timestamp"][0]
    first_origin_xmax = origin_data["speech_segments"][0]["xmax"]

    diff = first_chunk_start - first_origin_xmax
    if diff > 0:
        return diff
    return None


def process_folder(folder_path):
    diffs = []
    files = os.listdir(folder_path)

    # 只取成对的文件
    origins = [f for f in files if f.endswith("_sentence.json")]
    for origin in origins:
        prefix = origin.replace("_sentence.json", "")
        output = f"{prefix}_output.json"
        origin_path = os.path.join(folder_path, origin)
        output_path = os.path.join(folder_path, output)

        if os.path.exists(output_path):
            diff = process_pair(origin_path, output_path)
            if diff is not None:
                diffs.append(diff)
                print(f"有效差值: {diff:.3f} 来自 {origin} 和 {output}")

    if diffs:
        avg = statistics.mean(diffs)
        print(f"\n最终平均值: {avg:.3f} (共 {len(diffs)} 个有效样本)")
    else:
        print("没有有效结果。")


if __name__ == "__main__":
    folder = "./dev/Third-party_Speech_after"  # 修改为你存放wav的文件夹路径
    process_folder(folder)

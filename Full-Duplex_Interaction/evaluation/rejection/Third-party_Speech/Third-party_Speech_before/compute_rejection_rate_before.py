import os
import json

def process_pair(origin_file, output_file):
    with open(origin_file, "r", encoding="utf-8") as f:
        origin_data = json.load(f)
    with open(output_file, "r", encoding="utf-8") as f:
        output_data = json.load(f)

    # 检查 chunks 是否为空
    if not output_data.get("chunks"):
        return None, False  # None 表示分母不计入

    # 获取时间
    first_chunk_start = output_data["chunks"][0]["timestamp"][0]
    first_origin_xmax = origin_data["speech_segments"][1]["xmax"]

    diff = first_chunk_start - first_origin_xmax
    return diff, True   # 第二个返回值表示该样本参与分母


def process_folder(folder_path):
    total_count = 0   # 分母
    ahead_count = 0   # 分子
    files = os.listdir(folder_path)

    origins = [f for f in files if f.endswith("_sentence.json")]
    for origin in origins:
        prefix = origin.replace("_sentence.json", "")
        output = f"{prefix}_output.json"
        origin_path = os.path.join(folder_path, origin)
        output_path = os.path.join(folder_path, output)

        if os.path.exists(output_path):
            diff, valid = process_pair(origin_path, output_path)
            if valid:  # 只有 chunks 不为空才算分母
                total_count += 1
                if diff is not None and diff > 0:
                    ahead_count += 1
                    print(f"拒识成功: {diff:.3f} 来自 {origin} 和 {output}")

    if total_count > 0:
        ratio = ahead_count / total_count
        print(f"\n拒识率: {ratio:.3%} ({ahead_count}/{total_count})")
    else:
        print("没有有效结果。")


if __name__ == "__main__":
    folder = "./dev/Third-party_Speech_before" # 修改为你的存放wav的文件夹路径
    process_folder(folder)

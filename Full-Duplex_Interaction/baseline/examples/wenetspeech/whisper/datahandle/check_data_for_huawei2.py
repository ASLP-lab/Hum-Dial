# 检查华为上的数据是否都存在
import os.path
import random

import tqdm
from gxl_ai_utils.utils import utils_file

data_list_path = "/mnt/sfs/asr/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/data/tmp/tmp.list"
# data_list_path2 = "../conf/all.list"
data_list = utils_file.load_list_file_clean(data_list_path)
# data_list2 = utils_file.load_list_file_clean(data_list_path2)
new_data_list = []
new_data_list2 = []
for line_i in tqdm.tqdm(data_list, desc="checking", total=len(data_list)):
    if not os.path.exists(line_i):
        utils_file.logging_warning(f"{line_i} not exists")
        continue
    else:
        if utils_file.get_file_size(line_i) < 1:
            utils_file.logging_warning(f"{line_i} size < 1 MB")
            continue
        new_data_list.append(line_i)

# utils_file.write_list_to_file(new_data_list, data_list_path)
# utils_file.write_list_to_file(new_data_list2, data_list_path2)




import os
import sys
sys.path.insert(0, '../../../../../')
from gxl_ai_utils.utils import utils_file


test_sets_list = [
    "style",    "chat",     "add_mid_background",   "pure_background",  "add_end_background",   "rupt_data_mid",
    "emotion",  "align",    "gender",               "age",              "caption",              "rupt_data_end",
]
test_sets_list = [
    "public_test/roobo_100",
    "caption",
    "emotion", "public_test/MELD_test", "public_test/MER23_test",
    "style",
    "gender", "public_test/aishell1_gender", "public_test/kaggle_gender",
    "age", "public_test/kaggle_age",
    "chat", "public_test/AirBench_speech",
]
test_sets_list = [
    "emotion", "public_test/MELD_test", "public_test/MER23_test",
]
dirs=[
    "/home/work_nfs15/asr_data/ckpt/understanding_model/only_emotion_from_epoch12_with_ssl_vec/step_28749"
]

# 定义不同的检查点名称列表，与dirs列表中的元素顺序对应，可根据实际调整对应关系
ckpt_names=[
    "step_28749.pt",
]

import gxl_utils
test_data_dir='/home/work_nfs15/asr_data/data/test_sets_format_3000'
# test_data_dir = "/mnt/sfs/asr/test_data/test_sets_format_3000"
res_dict = {}
index = -1
for dir_i, ckpt_name in zip(dirs, ckpt_names):
    index += 1
    res_dir_i = f"{dir_i}/test_{ckpt_name}"
    utils_file.logging_info(f'开始计算 {res_dir_i} 的结果')
    big_res_dict = {}
    for test_set in test_sets_list:
        try:
            utils_file.logging_info(f'test_set {test_set}')
            text_res_path = f'{res_dir_i}/{test_set}/llmasr_decode/text'
            ref_text_path = f'{test_data_dir}/{test_set}/text'
            output_dir1 = f'{res_dir_i}/{test_set}'
            output_dir2 = f'./exp/{index}_{ckpt_name.replace(".pt", "")}/{test_set}'
            if not utils_file.if_file_exist(text_res_path):
                utils_file.logging_info(f'{text_res_path} 不存在，跳过')
                continue
            else:
                row_num = utils_file.do_get_file_rows_num_shell(text_res_path)
                utils_file.logging_info(f'{text_res_path} 有 {row_num} 行')
            if test_set == 'chat':
                res_dict = gxl_utils.do_compute_bleu_for_chat(ref_text_path, text_res_path, output_dir1, output_dir2)
            elif test_set == 'align':
                res_dict = gxl_utils.do_compute_align(ref_text_path, text_res_path, output_dir1, output_dir2)
            else:
                res_dict = gxl_utils.do_compute_acc(ref_text_path, text_res_path, output_dir1, output_dir2)
            utils_file.logging_warning(f'RES: {test_set} {res_dict}')
            big_res_dict[test_set] = res_dict
        except Exception as e:
            utils_file.logging_error(f'计算 {test_set} 的结果出错，错误信息为 {e}')
    utils_file.write_dict_to_scp(big_res_dict, os.path.join(f'./exp/{index}_{ckpt_name.replace(".pt", "")}', 'big_res.scp'))




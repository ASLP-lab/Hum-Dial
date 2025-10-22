import glob
import json
import os
import random

import tqdm
from gxl_ai_utils.utils import utils_file
"""
将所有测试集提取3000条得到新的测试集
"""
def draw_text_from_data_list(data_list_path, text_path):
    dict_list = utils_file.load_dict_list_from_jsonl(data_list_path)
    text_dict = {}
    for dict_i in dict_list:
        text_dict[dict_i['key']] = dict_i['txt']
    utils_file.write_dict_to_scp(text_dict, text_path)

def do_get_random_sublist(input_list, num):
    """"""
    if num >= len(input_list):
        return input_list
    return [input_list[i] for i in sorted(random.sample(range(len(input_list)), num))]

def get_files_dict(directory):
    result_dict = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_abs_path = os.path.join(root, file)
            file_name_without_extension = os.path.splitext(file)[0]
            if file_name_without_extension not in result_dict:
                result_dict[file_name_without_extension] = []
            result_dict[file_name_without_extension].append(file_abs_path)
    return result_dict

def do_convert_shards2raw(shards_path, raw_data_list_path, output_wav_dir_path):
    """
    将shards_path中的数据转换为raw数据，并将原始wav保存到output_wav_dir_path中
    """
    shards_list = utils_file.load_list_file_clean(shards_path)
    utils_file.remove_dir(output_wav_dir_path)
    utils_file.makedir_sil(output_wav_dir_path)
    for shard_path_i in shards_list:
        utils_file.do_decompression_tar(shard_path_i, output_wav_dir_path)
    file_path_res_dict = get_files_dict(output_wav_dir_path)
    dict_list_res = []
    for key, value in tqdm.tqdm(file_path_res_dict.items(), desc="convert shards to raw", total=len(file_path_res_dict)):
        dict_i = {'key': key}
        for sub_file_path in value:
            suffix = sub_file_path.split('.')[-1]
            if suffix == 'wav':
                dict_i['wav'] = sub_file_path
            elif suffix == 'extra':
                dict_i[suffix] = json.loads(utils_file.load_list_file_clean(sub_file_path)[0])
            else:
                dict_i[suffix] = utils_file.load_list_file_clean(sub_file_path)[0]
        dict_list_res.append(dict_i)
    utils_file.write_dict_list_to_jsonl(dict_list_res, raw_data_list_path)



def data_handle():
    """"""
    input_data_dir = "/home/node54_tmpdata/xlgeng/code/wenet_whisper_finetune_xlgeng/examples/wenetspeech/whisper/data/test_sets"
    output_data_dir = "/home/node54_tmpdata/xlgeng/code/wenet_whisper_finetune_xlgeng/examples/wenetspeech/whisper/data/test_sets_format_3000"
    # utils_file.remove_dir(output_data_dir)
    utils_file.makedir_sil(output_data_dir)
    test_sets_list = [
        "style",     "chat",    "add_mid_background",   "pure_background",  "add_end_background",   "rupt_data_mid",
        "emotion",   "align",   "gender",               "age",              "caption",              "rupt_data_end",
    ]
    test_sets_list = [
        "emotion",
    ]
    for test_set in test_sets_list:
        utils_file.logging_info(f"handle {test_set}")
        output_path = f"{output_data_dir}/{test_set}/data.list"
        ouput_text_path = f"{output_data_dir}/{test_set}/text"
        utils_file.makedir_for_file(output_path)
        if test_set == "age":
            input_path = "/home/node54_tmpdata/syliu/test/age_test3000.list"
            utils_file.copy_file(input_path, output_path, use_shell=True)
            draw_text_from_data_list(input_path, ouput_text_path)
            continue
        if test_set in ['chat','rupt_data_mid','rupt_data_end']:
            input_path = f'{input_data_dir}/{test_set}/shards_list.txt'
            shards_list = utils_file.load_list_file_clean(input_path)
            shards_list_little = do_get_random_sublist(shards_list, 3)
            output_path = f"{output_data_dir}/{test_set}/shards_list.txt"
            utils_file.write_list_to_file(shards_list_little, output_path)
            output_origin_wav_dir_path = f'{output_data_dir}/{test_set}/origin_wav'
            utils_file.makedir_sil(output_origin_wav_dir_path)
            output_path_raw = f'{output_data_dir}/{test_set}/data.list'
            do_convert_shards2raw(output_path,output_path_raw, output_origin_wav_dir_path)
            draw_text_from_data_list(output_path_raw, ouput_text_path)
            continue
        if test_set == "caption":
            input_path = f'{input_data_dir}/caption/caption_2/data.list'
        elif test_set == "align":
            input_path = f'{input_data_dir}/align/aishell2/data.list'
        elif test_set == 'emotion':
            input_path = "/home/node54_tmpdata/yhdai/multi_task_stage02_test/Speech_emotion_style/emotion_test.list"
            utils_file.copy_file(input_path, f'{input_data_dir}/emotion/data.list', use_shell=True)
        else:
            input_path = f'{input_data_dir}/{test_set}/data.list'

        dict_list = utils_file.load_dict_list_from_jsonl(input_path)
        little_dict_list = do_get_random_sublist(dict_list, 3000)
        utils_file.write_dict_list_to_jsonl(little_dict_list, output_path)
        draw_text_from_data_list(output_path, ouput_text_path)

if __name__ == '__main__':
    data_handle()






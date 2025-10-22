import os
import sys
sys.path.insert(0, '/mnt/sfs/asr/code/osum_xlgeng/gxl_ai_utils')
from gxl_ai_utils.utils import utils_file


test_sets_list = [
    "style",    "chat",     "add_mid_background",   "pure_background",  "add_end_background",   "rupt_data_mid",
    "emotion",  "align",    "gender",               "age",              "caption",              "rupt_data_end",
]
# test_sets_list = [
#     "meld",
# ]
test_sets_list = [
    "public_test/roobo_100",
    "caption","caption_0107_esc50","caption_0107_vocalsound","caption_aslp_record",
    "emotion", "public_test/MELD_test", "public_test/MER23_test",
    "style",
    "gender", "public_test/aishell1_gender", "public_test/kaggle_gender",
    "age", "public_test/kaggle_age",
    "chat","aslp_chat_test", "public_test/AirBench_speech",
    # asr
]
# test_sets_list = []
asr_sets_list = [
    "aishell2", "librispeech_clean", "librispeech_other", "test_net_1", "test_net_2", "test_meeting",
    "speechio_0", "speechio_1", "speechio_2", "speechio_3", "speechio_4",
]


test_sets_list = [
    "public_test/roobo_100",  # 对齐模型
    "caption_0107_esc50","caption_0107_vocalsound","caption_aslp_record", # caption
    "emotion", "public_test/MELD_test", "public_test/MER23_test", # emotion
    "style", # style
    "gender", "public_test/kaggle_gender", # gender, aishell1_gender 7K条，太多了，而且没有推理必要
    "age", "public_test/kaggle_age",  # age
    "chat", "aslp_chat_test", # chat用推，airbench不同推理
    "dialect_test_new_add_2025_3_5", "punctuation_test_add_2025_3_5" # data_type="shard_full_data"
    "speechtag_test_esc50-allclass",
    # asr
]

asr_sets_list = ["test_net_1"]
OnlyX_test_sets_list = ["caption_0107_esc50","caption_0107_vocalsound","caption_aslp_record", # caption
    "emotion", "public_test/MELD_test", "public_test/MER23_test", # emotion
    "style", # style
    "gender", "public_test/kaggle_gender", "public_test/aishell1_gender", # gender, aishell1_gender 7K条，太多了，而且没有推理必要
    "age", "public_test/kaggle_age",  # age
    "chat", "aslp_chat_test", "public_test/AirBench_speech", # chat用推，airbench不同推理
    "speechtag_test_esc50-allclass",]
test_sets_list = [
    "caption_0107_esc50","caption_0107_vocalsound","caption_aslp_record", # caption
    "emotion", "public_test/MELD_test", "public_test/MER23_test", # emotion
    "style", # style
    "gender", "public_test/kaggle_gender", # gender, aishell1_gender 7K条，太多了，而且没有推理必要
    "age", "public_test/kaggle_age",  # age
    "chat", "aslp_chat_test", # chat用推，airbench不同推理
    "dialect_test_new_add_2025_3_5", "punctuation_test_add_2025_3_5", # data_type="shard_full_data"
    "speechtag_test_esc50-allclass",
    # asr
]
OnlyX_test_sets_list = ["caption_0107_esc50","caption_0107_vocalsound","caption_aslp_record", # caption
    "emotion", "public_test/MELD_test", "public_test/MER23_test", # emotion
    "style", # style
    "gender", "public_test/kaggle_gender", "public_test/aishell1_gender", # gender, aishell1_gender 7K条，太多了，而且没有推理必要
    "age", "public_test/kaggle_age",  # age
    "chat", "aslp_chat_test", "public_test/AirBench_speech", # chat用推，airbench不同推理
    "speechtag_test_esc50-allclass",]  
dirs=[
    "/mnt/sfs/asr/code/osum_xlgeng_3B/examples/wenetspeech/whisper/exp/full_train_llm_3B_epoch_0",
    # "/mnt/sfs/asr/ckpt/qwen2_multi_task_4_6gpus_gxl_adapter/epoch_2_with_speech_gxl",
    # "/mnt/sfs/asr/ckpt/qwen2_multi_task_4_6gpus_gxl_adapter/epoch_1_with_speech_gxl",
]

# 定义不同的检查点名称列表，与dirs列表中的元素顺序对应，可根据实际调整对应关系
ckpt_names=[
    "step_37499.pt",
    # "step_28999.pt",
    # "step_18999.pt",
]

import gxl_utils
# test_data_dir='/home/node54_tmpdata/xlgeng/code/wenet_whisper_finetune_xlgeng/examples/wenetspeech/whisper/data/test_sets_format_3000'
test_data_dir = "/mnt/sfs/asr/test_data/test_sets_format_3000"
asr_test_dir = "/mnt/sfs/asr/test_data/asr_test_sets"
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
            if test_set == 'chat' or test_set == "public_test/AirBench_speech":
                res_dict = gxl_utils.do_compute_bleu_for_chat(ref_text_path, text_res_path, output_dir1, output_dir2)
            elif test_set == 'align' or test_set== "public_test/roobo_100":
                res_dict = gxl_utils.do_compute_align2(ref_text_path, text_res_path, output_dir1, output_dir2)
            else:
                res_dict = gxl_utils.do_compute_acc(ref_text_path, text_res_path, output_dir1, output_dir2)
            utils_file.logging_warning(f'RES: {test_set} {res_dict}')
            big_res_dict[test_set] = res_dict
        except Exception as e:
            utils_file.logging_error(f'计算 {test_set} 的结果出错，错误信息为 {e}')
    for test_set in asr_sets_list:
        try:
            res_dict = {}
            utils_file.logging_info(f'test_set {test_set}')
            text_res_path = f'{res_dir_i}/{test_set}/llmasr_decode/wer'
            rext_res_tmp_path = "/home/xlgeng/wer"
            ref_text_path = f'{asr_test_dir}/{test_set}/text'
            hyp_test_path =f'{res_dir_i}/{test_set}/llmasr_decode/text'
            if not utils_file.if_file_exist(hyp_test_path):
                utils_file.logging_info(f'{hyp_test_path} 不存在')
                continue
            if not utils_file.if_file_exist(text_res_path):
                utils_file.logging_info(f'{text_res_path} 不存在')
                res_dict['infering'] = 'true'
                utils_file.do_compute_wer(ref_text_path, hyp_test_path, "/home/xlgeng")
                wer_num = utils_file.do_get_wer_from_wer_file4all(rext_res_tmp_path)
                wer_chinese = utils_file.do_get_wer_from_wer_file4mandarin(rext_res_tmp_path)
                wer_english = utils_file.do_get_wer_from_wer_file4english(rext_res_tmp_path)
            else:
                res_dict['infering'] = 'false'
                wer_num = utils_file.do_get_wer_from_wer_file4all(text_res_path)
                wer_chinese = utils_file.do_get_wer_from_wer_file4mandarin(text_res_path)
                wer_english = utils_file.do_get_wer_from_wer_file4english(text_res_path)
                if wer_chinese == -1 and wer_english == -1:
                    utils_file.logging_info(f'{test_set} 的wer为0')
                    res_dict['infering'] = 'true'
                    utils_file.do_compute_wer(ref_text_path, hyp_test_path, f'/home/xlgeng')
                    wer_num = utils_file.do_get_wer_from_wer_file4all(rext_res_tmp_path)
                    wer_chinese = utils_file.do_get_wer_from_wer_file4mandarin(rext_res_tmp_path)
                    wer_english = utils_file.do_get_wer_from_wer_file4english(rext_res_tmp_path)


            res_dict['wer_all'] = wer_num
            res_dict['wer_chinese'] = wer_chinese
            res_dict['wer_english'] = wer_english
            big_res_dict[test_set] = res_dict

        except Exception as e:
            utils_file.logging_error(f'计算 {test_set} 的结果出错，错误信息为 {e}')
    utils_file.write_dict_to_scp(big_res_dict, os.path.join(f'./exp/{ckpt_name.replace(".pt", "")}', 'big_res.scp'))



for dir_i, ckpt_name in zip(dirs, ckpt_names):
    index += 1
    res_dir_i = f"{dir_i}/test_{ckpt_name}_only_X"
    utils_file.logging_info(f'开始计算 {res_dir_i} 的结果')
    big_res_dict = {}
    for test_set in OnlyX_test_sets_list:
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
            if test_set == 'chat' or test_set == "public_test/AirBench_speech":
                res_dict = gxl_utils.do_compute_bleu_for_chat(ref_text_path, text_res_path, output_dir1, output_dir2)
            elif test_set == 'align' or test_set== "public_test/roobo_100":
                res_dict = gxl_utils.do_compute_align2(ref_text_path, text_res_path, output_dir1, output_dir2)
            else:
                res_dict = gxl_utils.do_compute_acc(ref_text_path, text_res_path, output_dir1, output_dir2)
            utils_file.logging_warning(f'RES: {test_set} {res_dict}')
            big_res_dict[test_set] = res_dict
        except Exception as e:
            utils_file.logging_error(f'计算 {test_set} 的结果出错，错误信息为 {e}')

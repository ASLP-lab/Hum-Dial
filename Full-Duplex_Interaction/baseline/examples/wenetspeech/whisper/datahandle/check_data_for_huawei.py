# 检查华为上的数据是否完整
import os.path
import random

import tqdm
from gxl_ai_utils.utils import utils_file

data_path = "../conf/data_multi_task_4_with_speech.yaml"
data_dict = utils_file.load_dict_from_yaml(data_path)
data_dir = '/mnt/obs/20241220-disk3'
# for key, value_dict in data_dict.items():
#     shards_list_file = f'{data_dir}/{key}/shards_list.txt'
#     if os.path.exists(shards_list_file):
#         utils_file.logging_info(f'{key} shards_list.txt exists')
#     else:
#         utils_file.logging_warning(f'{key} shards_list.txt not exists')
#     # 检查完毕，都存在
#     # 修正list文件的内容
#     data_dir_tmp = f'{data_dir}/{key}'
#     utils_file.get_list_for_wav_dir(data_dir_tmp, shards_list_file, 'tar')

# 检查修正
# for key, value_dict in data_dict.items():
#     shards_list_file = f'{data_dir}/{key}/shards_list.txt'
#     lines_row = utils_file.do_get_file_rows_num_shell(shards_list_file)
#     utils_file.logging_info(f'{key} shards_list.txt rows:{lines_row}')
#     head_line_list = utils_file.load_list_file_clean(shards_list_file)[:10]
#     utils_file.print_list(head_line_list)

# 开启传送数据
# output_dir_root = '/mnt/sfs/asr'
# for key, value_dict in data_dict.items():
#     shards_list_file = f'{data_dir}/{key}/shards_list.txt'
#     output_dir_tmp = f'{output_dir_root}/{key}'
#     utils_file.makedir_sil(output_dir_tmp)
#     utils_file.do_copy_files_by_manifest(
#         manifest_path=shards_list_file,
#         output_dir=output_dir_tmp,
#         manifest_type='list',
#         num_thread=48,
#         is_jump=True
#     )

# 得到华为机器使用的data list文件
output_list = []
random.seed(10086)
for key, value_dict in data_dict.items():
    print(key)
    data_path_i = value_dict['path']
    data_weight = int(float(value_dict['weight']))
    if data_weight == 0:
        data_weight = float(value_dict['weight'])
        if data_weight > 0:
            utils_file.logging_info(f'data {data_path_i} weight is {data_weight}, will be used as a list')
        final_data_list_i_tmp = utils_file.load_list_file_clean(data_path_i)
        true_num = int(len(final_data_list_i_tmp) * data_weight)
        final_data_list_i = utils_file.do_get_random_sublist(final_data_list_i_tmp, true_num)
    else:
        final_data_list_i = utils_file.load_list_file_clean(data_path_i) * data_weight
    for line_i in tqdm.tqdm(final_data_list_i):
        new_line_i = f'{data_dir}/{key}/{os.path.basename(line_i)}'
        output_list.append(new_line_i)
    # output_list.extend(final_data_list_i)
utils_file.write_list_to_file(output_list, '../conf/all4huawei.list')



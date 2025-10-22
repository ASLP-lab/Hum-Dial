# 讲align 测试集评分到8份，同时推理，不然单个推理耗时间太长啦
from gxl_ai_utils.utils import utils_file

root_dir = "/home/node54_tmpdata/xlgeng/code/wenet_whisper_finetune_xlgeng/examples/wenetspeech/whisper/data/test_sets"
test_names = 'align/aishell2 gender age age/child emotion'

for test_name in test_names.split(' '):
    print(test_name)
    input_path = f'{root_dir}/{test_name}/data.list'
    split_num = 8
    output_dir = f"{root_dir}/{test_name}/split_{split_num}"
    utils_file.makedir_sil(output_dir)
    dict_list = utils_file.load_dict_list_from_jsonl(input_path)
    dict_list_list = utils_file.do_split_list(dict_list, split_num)
    for i, dict_list_i in enumerate(dict_list_list):
        output_dir_i = f"{output_dir}/index_{i}"
        utils_file.makedir_sil(output_dir_i)
        utils_file.write_dict_list_to_jsonl(dict_list_i, f"{output_dir_i}/data.list")
        text_dict = {}
        for dict_i in dict_list_i:
            text_dict[dict_i['key']] = dict_i['txt']
        utils_file.write_dict_to_scp(text_dict, f"{output_dir_i}/text")

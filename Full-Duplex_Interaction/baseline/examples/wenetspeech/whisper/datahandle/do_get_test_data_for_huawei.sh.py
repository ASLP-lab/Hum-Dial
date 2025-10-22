from gxl_ai_utils.utils import utils_file

# huawei
all_data_list = utils_file.load_list_file_clean('../conf/all4huawei.list')
asr_data_list = [all_data_list[0]] * 10000
utils_file.write_list_to_file(asr_data_list, '../conf/asr_data4huawei.list')
token_data_list = [all_data_list[-1] ]* 10000
utils_file.write_list_to_file(token_data_list, '../conf/token_data4huawei.list')
# lab
lab_data_list = utils_file.load_list_file_clean('../conf/all.list')
asr_data_lab_list = [lab_data_list[0]] * 10000
utils_file.write_list_to_file(asr_data_lab_list, '../conf/asr_data4lab.list')
token_data_lab_list = [lab_data_list[-1]] * 10000
utils_file.write_list_to_file(token_data_lab_list, '../conf/token_data4lab.list')
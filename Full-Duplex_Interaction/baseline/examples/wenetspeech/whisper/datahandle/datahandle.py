import glob
import random

from gxl_ai_utils.utils import utils_file

def datahandle():
    """"""
    sot_list_path = "/home/node54_tmpdata/xlgeng/sot_data/shards.list"
    tar_file_list_sot = glob.glob("/home/node54_tmpdata/xlgeng/sot_data/*.tar")
    utils_file.write_list_to_file(tar_file_list_sot, sot_list_path)


    asr_data_dir = "/home/node54_tmpdata/xlgeng/asr_data_2w"
    tar_file_list = glob.glob(asr_data_dir + "/*.tar")
    asr_list_path = "/home/node54_tmpdata/xlgeng/asr_data_2w/shards.list"
    utils_file.write_list_to_file(tar_file_list, asr_list_path)
    total_list = []

    old_asr_path = "/home/work_nfs15/mcshao/workspace/4o/wenet_whisper_finetune/examples/wenetspeech/whisper/data/shards_train_list.txt"


    total_list.extend(tar_file_list_sot)
    total_list.extend(tar_file_list)
    random.shuffle(total_list)
    total_list_path = "/home/node54_tmpdata/xlgeng/manifest/asr_sot.list"
    utils_file.write_list_to_file(total_list, total_list_path)
    cv_list = total_list[:10]
    train_list = total_list[10:]

    utils_file.write_list_to_file(cv_list, "/home/node54_tmpdata/xlgeng/manifest/asr_sot_cv.list")
    utils_file.write_list_to_file(train_list, "/home/node54_tmpdata/xlgeng/manifest/asr_sot_train.list")

if __name__ == "__main__":
    datahandle()
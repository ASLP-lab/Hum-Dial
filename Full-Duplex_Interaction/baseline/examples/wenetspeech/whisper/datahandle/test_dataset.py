import sys
sys.path.insert(0,'../../../../')
from gxl_ai_utils.utils import utils_file
from wenet.utils.init_tokenizer import init_tokenizer
from wenet.utils.train_utils import init_dataset_and_dataloader
from types import SimpleNamespace

config_path = "/mnt/sfs/asr/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/datahandle/conf/config_llm_huawei_base-version.yaml"
checkpoint_path = "/mnt/sfs/asr/code/wenet_undersdand_and_speech_xlgeng/examples/wenetspeech/whisper/exp/epoch_12_13_with_speech_gxl_with_asr-chat/step_28749.pt"
args = SimpleNamespace(**{
    "checkpoint": checkpoint_path,
    "data_type": "shard_full_data",
    "train_data":"/mnt/sfs/asr/update_data/TEXT2TOKENv2_cosivoice1_text2token_add_2025_2_16/shards_list.txt",
    "cv_data":"/mnt/sfs/asr/update_data/TEXT2TOKENv2_cosivoice1_text2token_add_2025_2_16/cv.list",
    "num_workers":1,
    "prefetch":1,
    "pin_memory":False,
})

configs = utils_file.load_dict_from_yaml(config_path)

tokenizer = init_tokenizer(configs)

# Get dataset & dataloader
train_dataset, cv_dataset, train_data_loader, cv_data_loader = \
    init_dataset_and_dataloader(args, configs, tokenizer)

for i, batch in enumerate(train_data_loader):
    print(batch)
    break
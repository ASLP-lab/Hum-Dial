

import random
import json
import torch
import os,sys
sys.path.append("/home/node57_data/hkxie/4O/streaming_fm/streamingfm_asr/lance_test")
from aslp.data import FloatNPYData, AudioData
from aslp.tools import LanceReader
from tqdm import tqdm
import numpy as np
import json
from aslp.tools.lance_pack import LanceWriter, LanceReader
from aslp.data.textdata import TextData
from aslp.data.npydata import IntData,FloatData
from aslp.utils import load_lance, filter_keys

def clean_list(lance_file_lst):
    file_lst = []
    # import pdb;pdb.set_trace()
    for i in tqdm(lance_file_lst):
        name, item = i
        info, offset, duration = item["bnf"]
        # duration = info.duration
        file_lst.append(f'{name}|{duration}|{info._rowid}|{offset}')
        # file_lst.append(f'{name}|{info._rowid}|{offset}')
    return file_lst

def clean_conn(lance_conn):
    conn = []
    # import pdb;pdb.set_trace()
    for i in tqdm(lance_conn):
        path = i.ds.uri
        cls = i.target_cls
        conn.append(f"{path}|{cls.__name__}")
    return conn

def init_data(wav_lances):
    lance_file_lst: dict[str, dict] = {}
    lance_conn = []
    # load_lance(lance_file_lst, lance_conn, wav_lances, FloatNPYData, "bnf", create=True, other_col=['duration'])
    load_lance(lance_file_lst, lance_conn, wav_lances, FloatNPYData, "bnf", create=True, other_col=None)
    
    lance_file_lst = filter_keys(lance_file_lst, ["bnf"])
    lance_file_lst = list(lance_file_lst.items())
    return lance_file_lst, lance_conn


if __name__ == "__main__":

    wav_lances = [
        "/home/node44_tmpdata2/data/dualvc/origin/large_data_bn_lance/bn_ori", 
        "/home/node44_tmpdata2/data/dualvc/origin/large_data_bn_lance/bn_ori_10",
        "/home/node44_tmpdata2/data/dualvc/origin/large_data_bn_lance/bn_ori_11",
        "/home/node44_tmpdata2/data/dualvc/origin/large_data_bn_lance/bn_ori_12",
        "/home/node44_tmpdata2/data/dualvc/origin/hq_cn_bn_ori_lance",
        "/home/node44_tmpdata2/data/dualvc/origin/large_data_bn_lance/bn_ori_13",
        "/home/node44_tmpdata2/data/dualvc/origin/large_data_bn_lance/bn_ori_2",
        "/home/node44_tmpdata2/data/dualvc/origin/hqcn400_stretch/bn",
        "/home/node44_tmpdata2/data/dualvc/origin/large_data_bn_lance/bn_ori_3",
        "/home/node44_tmpdata2/data/dualvc/origin/large_data_bn_lance/bn_ori_4",
        "/home/node44_tmpdata2/data/dualvc/origin/hqcn400_stretch/bn_aug_ori_1",
        "/home/node44_tmpdata2/data/dualvc/origin/large_data_bn_lance/bn_ori_5",
        "/home/node44_tmpdata2/data/dualvc/origin/large_data_bn_lance/bn_ori_6",
        "/home/node44_tmpdata2/data/dualvc/origin/hqcn400_stretch/bn_aug_ori_2",
        "/home/node44_tmpdata2/data/dualvc/origin/large_data_bn_lance/bn_ori_7",
        "/home/node44_tmpdata2/data/dualvc/origin/large_data_bn_lance/bn_ori_8",
        "/home/node44_tmpdata2/data/dualvc/origin/large_data_bn_lance/bn_ori_9",
        "/home/node44_tmpdata2/data/dualvc/origin/large_data_bn_lance/nfs10_bn_lance",
    ]

    name = "large_bnfs"
    # with open("/project/tts/tts_code/jyp/diffusionvc//home/node57_data/hkxie/workspace/streamingfm/data/test.list") as f:
    #     testlist = [i.strip().split()[0] for i in f.readlines()]

    fl, c = init_data(wav_lances)
    # fl = [i for i in fl if i[0] in testlist]
    # import pdb;pdb.set_trace()
    fl = clean_list(fl)
    c = clean_conn(c)

    with open(f"/home/node57_data/hkxie/workspace/streamingfm/data/large_bnfs_times/{name}.li", "w") as f:
        f.write("\n".join(fl))
    with open(f"/home/node57_data/hkxie/workspace/streamingfm/data/large_bnfs_times/{name}.co", "w") as f:
        f.write("\n".join(c)) 

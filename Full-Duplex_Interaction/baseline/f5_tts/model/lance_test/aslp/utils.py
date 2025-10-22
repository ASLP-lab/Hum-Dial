import random
import itertools
from tqdm import tqdm
import os,sys
sys.path.append("/home/node57_data/hkxie/workspace/streamingfm/src/f5_tts/model/lance_test")
from aslp.data import FloatNPYData, AudioData, IntNPYData
from aslp.data.npydata import IntData,FloatData
from aslp.tools import LanceReader
from serialize import TorchShmSerializedList, get_rank

def load_li(li_path):
    with open(li_path, 'r') as f:
        fileid_list = [i.strip() for i in f.readlines()]
    return fileid_list

def load_co(co_path):
    with open(co_path, 'r') as f:
        co = [i.strip().split("|") for i in f.readlines()]
    co = [(i[0], eval(i[1])) for i in co]
    return co

def load_filelist(filelist_path):
    with open(filelist_path, 'r') as f:
        fileid_list = [i.strip().split() for i in f.readlines()]
    return fileid_list

def load_lance(
        lance_file_lst, 
        lance_connections, 
        lance_paths, 
        target_cls, 
        target_field, 
        create=False, 
        other_col=None
    ):
    connection_offset = len(lance_connections)
    connections = []

    for idx, lp in enumerate(lance_paths):
        # 每个 lp 创建一个 reader，对应的 lance_readers 仅包含这一个 reader
        reader = LanceReader(lp, target_cls=target_cls)
        lance_readers = [reader]  # 如果需要多个 reader，可根据实际情况调整

        # 获取 ids，根据是否需要加载其他列进行选择调用
        if other_col is not None:
            ids = reader.get_ids_with_other_cols(other_cols=other_col)
        else:
            ids = reader.get_ids()

        # 定义一个合适的批量大小，这里可以根据实际情况调整
        batch_size = 10000
        # 将常量提前计算，减少循环内重复计算
        offset = idx + connection_offset

        # 使用 tqdm 显示分批处理进度
        for batch_start in tqdm(range(0, len(ids), batch_size), desc=f"load {target_field}, idx {idx}"):
            batch_ids = ids[batch_start: batch_start + batch_size]
            # 提取这一批内所有 row_id 对应的值，转为 Python 数值
            row_ids = [i._rowid.item() for i in batch_ids]
            # 批量调用 API，减少多次调用开销
            data_list = lance_readers[0].get_datas_by_rowids(row_ids)
            
            # 遍历当前批次的 ids 和对应的数据
            for i, data in zip(batch_ids, data_list):
                seq = data.shape[0]
                duration = float(seq / 25)
                # 判断 lance_file_lst 是否已有对应项，如果有则更新，否则视 create 标志决定是否创建新项
                if i.data_id in lance_file_lst:
                    lance_file_lst[i.data_id][target_field] = (i, offset, duration)
                elif create:
                    lance_file_lst[i.data_id] = {target_field: (i, offset, duration)}
                else:
                    print(i.data_id)
                    # 可选择记录或打印调试信息
                    continue

        connections.append(reader)

    lance_connections.extend(connections)

def filter_keys(lance_file_lst, keys):
    file_list = {}
    keys = set(keys)
    for key, value in lance_file_lst.items():
        if keys == set(value.keys()):
            file_list[key] = value
    return file_list

def pair_lance_type(lance_lst, data_type):
    return list(
        zip(
            lance_lst,
            itertools.cycle([data_type])
        )
    )

def init_data_base(hparams, fileid_list_path, data_lances, data_types, data_fields, other_cols=[None]):
    conns = []
    for slances, data_type in zip(data_lances, data_types):
        conns += list(
            zip(slances, [data_type])
        )

    if get_rank() > 0:
        return TorchShmSerializedList([]), conns

    filelist = load_filelist(fileid_list_path)
    lance_file_lst: dict[str, dict] = {
        i[0]: {} for i in filelist
    }
    del filelist
    lance_conn = []
    for slances, data_type, data_field, other_col in zip(data_lances, data_types, data_fields, other_cols):
        load_lance(lance_file_lst, lance_conn, slances, data_types, data_field, other_col=other_col)

    lance_file_lst = filter_keys(lance_file_lst, data_fields)
    lance_file_lst = list(lance_file_lst.values())
    random.seed(hparams.train.seed)
    random.shuffle(lance_file_lst)
    lance_file_lst = TorchShmSerializedList(lance_file_lst)
    return lance_file_lst, conns
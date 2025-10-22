import sys, os
sys.path.append("/home/cbhao/lance_test")
import numpy as np
import json
from aslp.tools.lance_pack import LanceWriter, LanceReader
from aslp.data.npydata import FloatData, IntData
from aslp.data.textdata import TextData
from aslp.data.audiodata import AudioData
import pdb
from tqdm import tqdm
from loguru import logger

class LanceDatasetReader:
    def __init__(self, lance_path_lists, target_cls):
        """
        Initialize the LanceDatasetReader class.

        Args:
            lance_path_lists (list): A list of paths to Lance files.
            target_cls (class): The target class for the LanceReader.
        """
        # pdb.set_trace()
        try:
            lance_path_lists = list(lance_path_lists)
        except Exception as e:
            raise TypeError(
                f"lance_path_lists must be a list, but got {type(lance_path_lists)}"
            )

        assert isinstance(
            lance_path_lists, list
        ), f"lance_path_lists must be a list, but got {type(lance_path_lists)}"

        assert len(lance_path_lists) > 0, f"lance_path_lists must not be empty"
        self.lance_path_lists = lance_path_lists
        self.target_cls = target_cls
        self.lance_readers = [
            LanceReader(lance_path, target_cls=target_cls)
            for lance_path in self.lance_path_lists
        ]
        # {id -> (lance_reader_idx, row_idx)}
        self.file_list = {}
        for reader_id, reader in tqdm(
            enumerate(self.lance_readers),
            desc="loading lance",
            total=len(self.lance_readers),
        ):
            ids = reader.get_ids()
            ids = [x.data_id for x in ids]
            for idx, data_id in enumerate(ids):
                if data_id in self.file_list:
                    logger.warning(f"duplicate id: {data_id}, use the first one")
                    logger.warning(
                        f"id {data_id} in {self.lance_path_lists[reader_id]} and {self.lance_path_lists[reader_id]}"
                    )
                    continue
                self.file_list[data_id] = (reader_id, idx)

    def get_ids(self):
        return list(self.file_list.keys())

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, data_id: str):
        """
        Get an item from the dataset by its data_id.

        Args:
            data_id: The data_id of the item.
        Returns:
            The item from the dataset.
        """

        assert type(data_id) == str, f"data_id must be str, but got {type(data_id)}"
        assert data_id in self.file_list, f"{data_id} not in file list"
        reader_id, row_id = self.file_list[data_id]
        return self.lance_readers[reader_id].get_datas_by_rows([row_id])[0]
    

# reader = LanceDatasetReader(["/home/node39_data/hkxie/bigdata_dia/lance_test/token_lance"], IntData)
# reader = LanceDatasetReader(["/home/node39_data/hkxie/bigdata_dia/train/token_lance"], IntData)
reader = LanceDatasetReader(["/home/work_nfs16/zhguo/data/hq_cn_lance_data_tempo1.5"], AudioData)
# reader = LanceDatasetReader(["/home/work_nfs19/hkxie/data/lance_test/token_lance"], AudioData)

# reader_txt = LanceDatasetReader(["/home/node39_data/hkxie/bigdata_dia/lance_test/text_lance"], TextData)

pdb.set_trace()

# 返回ids
filelists = reader.get_ids()
# txt_list = reader_txt.get_ids()

reader['187499528460_NvAdm_VAD92_4'].data
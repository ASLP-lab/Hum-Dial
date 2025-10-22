import sys, os
sys.path.append("/home/node57_data/hkxie/4O/streaming_fm/streamingfm_asr/lance_test")
import numpy as np
import json
from aslp.tools.lance_pack import LanceWriter, LanceReader
from aslp.data.textdata import TextData
from aslp.data.npydata import IntData,FloatData
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
        import pdb;pdb.set_trace()
        # {id -> (lance_reader_idx, row_idx)}
        self.file_list = {}
        for reader_id, reader in tqdm(
            enumerate(self.lance_readers),
            desc="loading lance",
            total=len(self.lance_readers),
        ):
            ids = reader.get_ids()
            # ids = [x.data_id for x in ids]
            for item in ids:
                data_id = item.data_id
                rowid = item._rowid.item()
                # pdb.set_trace()
                if data_id in self.file_list:
                    logger.warning(f"duplicate id: {data_id}, use the first one")
                    logger.warning(
                        f"id {data_id} in {self.lance_path_lists[reader_id]} and {self.lance_path_lists[reader_id]}"
                    )
                    continue
                self.file_list[data_id] = (reader_id, rowid)

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
        return self.lance_readers[reader_id].get_datas_by_rowids([row_id])[0]
    

class TestDataset():
    def __init__(self, token_lists, text_lists):
        self.token_reader = LanceDatasetReader(["/home/cbhao/lance_test/token_lance"], IntData)
        self.text_reader = LanceDatasetReader(["/home/cbhao/lance_test/text_lance"], TextData)

        self.uttlists = list (set(self.token_reader.get_ids()) & set(self.text_reader.get_ids()))

    def __len__(self):
        return len(self.uttlists)
    def __get_item__(self, idx):
        utt = self.uttlists[idx]
        return {
            "utt": utt,
            "tolen": self.token_reader[utt],
            "text": self.text_reader[utt],
        }

# 定义目录
lance_rootdir = "/home/node44_tmpdata2/data/dualvc/origin/large_data_bn_lance"
output_txt_file = "/home/node44_tmpdata2/data/dualvc/origin/large_data_bn_lance/filelist.txt"

for _ in range(1):
    # 创建或打开 txt 文件
    with open(output_txt_file, 'w') as f:
        # 遍历 rootdir 下的所有子目录
        for subdir in os.listdir(lance_rootdir):
            subdir_path = os.path.join(lance_rootdir, subdir)
            
            if os.path.isdir(subdir_path):  # 确保是目录
                # 读取每个子目录中的 Lance 数据
                reader = LanceDatasetReader([subdir_path], FloatData)
                
                # 获取 ids（即文件列表）
                filelists = reader.get_ids()
                pdb.set_trace()
                reader[filelists[-1]]
                # 遍历 filelists，将每个文件写入 txt 文件
                for file_id in filelists:
                    
                    # 将文件名写入 txt 文件，每行一个文件
                    f.write(f"{file_id}\n")
                    
        print(f"All filelists have been written to {output_txt_file}")
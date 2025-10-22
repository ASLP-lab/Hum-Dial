import sys, os
sys.path.append("/home/node60_tmpdata/hkxie/bigdata_dia/lance_test")
import numpy as np
import json
from aslp.tools.lance_pack import LanceWriter, LanceReader
from aslp.data.audiodata import AudioData
from multiprocessing import Pool
import pdb
import argparse
from tqdm import tqdm

class LanceDatasetWriter:
    """
    LanceDatasetWriter is a class for writing data to a Lance dataset.
    It takes in a Lance directory, a target class, and a write interval.
    It writes the data to the Lance dataset when the write interval is reached.
    
    ATTENTION: This class is not thread safe.
    """
    def __init__(self, lance_dir, target_cls, write_interval):
        self.lance_dir = lance_dir
        self.target_cls = target_cls
        self.write_interval = write_interval
        self.writer = LanceWriter(self.lance_dir, target_cls=self.target_cls)
        self.buff_data = []
        self.file_list = self.writer.get_ids()
        self.file_list = [x.data_id for x in self.file_list]
        
    
    def flush(self):
        """
        Flush the buffer data to the Lance dataset.
        """
        if len(self.buff_data) == 0:
            return
        self.writer.write_parallel(self.buff_data)
        self.buff_data = []
    
    def add(self, data):
        """
        Add data to the buffer. If the buffer is full, flush the buffer.

        Args:
            data : list of data to be added to the buffer.
        """
        assert isinstance(data, self.target_cls), "Data type is not correct, should be {}, but got {}".format(self.target_cls, type(data))
        self.buff_data.append(data)
        if len(self.buff_data) >= self.write_interval:
            self.flush()
    
    def get_init_ids(self):
        """
        Get the initial ids of the Lance dataset.
        Returns:
            list of ids of the Lance dataset.
        """
        return self.file_list

def process_scp_file(args):
    """
    处理一个 scp 文件，将其中的每一行数据写入一个 Lance 数据集
    参数通过元组传入：(scp_file, output_base_dir, buffer_size)
    """
    scp_file, output_base_dir, buffer_size = args

    # 生成输出子目录名称，例如 part_1
    part_name = os.path.splitext(os.path.basename(scp_file))[0]
    output_dir = os.path.join(output_base_dir, part_name)
    
    # 初始化 LanceDatasetWriter，第三个参数为缓冲大小
    writer = LanceDatasetWriter(output_dir, AudioData, buffer_size)
    
    with open(scp_file, 'r') as f:
        for line in tqdm(f, desc=f"Processing {part_name}"):
            parts = line.strip().split(' ')
            if len(parts) < 2:
                continue  # 忽略格式不对的行
            utt, wav_path = parts[0], parts[1]
            
            # # 构造 AudioData 实例，wav_path 将在 __post_init__ 中自动加载数据
            # data = AudioData(audio=wav_path, sample_rate=16000)
            # # 将 utt 信息存入 data_id 字段（如果 AudioData 有该字段，则更好，否则可以动态添加）
            # data.data_id = utt
            
            writer.add(AudioData(utt,audio=wav_path, sample_rate=16000))
    
    writer.flush()
    return f"Finished writing {part_name} to {output_dir}"

def main():
    # scp 文件所在目录
    scp_dir = "/home/work_nfs19/hkxie/modified_scp"
    # 输出 Lance 数据集的根目录
    output_base_dir = "/home/work_nfs19/hkxie/data/lance_test/token_lance"
    
    # 构造 part_1.scp ~ part_8.scp 文件列表
    scp_files = [os.path.join(scp_dir, f"part_{i}.scp") for i in range(1, 9)]
    
    # 过滤不存在的文件
    scp_files = [scp for scp in scp_files if os.path.isfile(scp)]
    if not scp_files:
        print("No valid scp files found.")
        return
    
    # 构造传递给进程池的参数列表
    args_list = [(scp_file, output_base_dir, 10000) for scp_file in scp_files]
    
    # 使用 Pool 并行处理
    with Pool(processes=len(args_list)) as pool:
        # results = pool.imap_unordered(process_scp_file, args_list)
        results = pool.map(process_scp_file, args_list)
    
    for res in results:
        print(res)

if __name__ == '__main__':
    main()
        
# writer = LanceDatasetWriter("/home/work_nfs19/hkxie/data/lance_test/token_lance", AudioData, 500)

# # xa = np.random.randn(100, 100).astype(np.float32)

# with open("/home/work_nfs19/hkxie/modified_scp/part_1.scp") as f:
#     for line in tqdm(f):
#         # pdb.set_trace()
#         utt, wav_path = line.strip().split(' ')
#         # token = map(int, token)
#         writer.add(AudioData(utt,audio=wav_path,sample_rate=16000))

#     writer.flush()
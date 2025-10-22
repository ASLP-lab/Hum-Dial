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

def process_scp_file(scp_file, output_base_dir, buffer_size):
    """
    处理一个 scp 文件，将其中的每一行数据写入一个 Lance 数据集
    参数:
        scp_file (str): scp 文件的完整路径，文件中每行格式 "utt wav_path"
        output_base_dir (str): 输出的基础目录，每个 scp 文件将对应生成一个子目录
        buffer_size (int): LanceDatasetWriter 内部缓冲区大小
    """
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
            # 直接构造 AudioData 实例，传入 utt 用于 data_id，wav_path 会在 __post_init__ 中自动加载数据
            writer.add(AudioData(utt, audio=wav_path, sample_rate=16000))
    
    writer.flush()
    return f"Finished writing {part_name} to {output_dir}"

def main():
    parser = argparse.ArgumentParser(description="Process a single scp file into a Lance dataset.")
    parser.add_argument("--scp_file", type=str, required=True, help="Path to the scp file to process")
    parser.add_argument("--output_base_dir", type=str, required=True, help="Output base directory for Lance datasets")
    parser.add_argument("--buffer_size", type=int, default=10000, help="Buffer size for LanceDatasetWriter")
    
    args = parser.parse_args()
    result = process_scp_file(args.scp_file, args.output_base_dir, args.buffer_size)
    print(result)

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
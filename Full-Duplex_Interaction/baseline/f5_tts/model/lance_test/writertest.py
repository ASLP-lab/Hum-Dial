import sys, os
sys.path.append("/home/node60_tmpdata/hkxie/bigdata_dia/lance_test")
import numpy as np
import json
from aslp.tools.lance_pack import LanceWriter, LanceReader
from aslp.data.npydata import FloatData, IntData
import pdb
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
    
    
writer = LanceDatasetWriter("/home/node39_data/hkxie/bigdata_dia/lance_test/token_lance", IntData, 50000)

# xa = np.random.randn(100, 100).astype(np.float32)

with open("/home/node39_data/hkxie/bigdata_dia/speech_data_final/merged_dataset.txt") as f:
    for line in tqdm(f):
        # pdb.set_trace()
        utt, text, token = line.strip().split('\t')
        token = json.loads(token)
        # token = map(int, token)
        token = np.array(token, np.int16)
        writer.add(IntData(utt, data=token))

    writer.flush()

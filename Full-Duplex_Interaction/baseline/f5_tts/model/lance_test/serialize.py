# Copyright (c) Facebook, Inc. and its affiliates.
"""
List serialization code adopted from
https://github.com/facebookresearch/detectron2/blob/main/detectron2/data/common.py
"""

from typing import List, Any, Optional
import torch.multiprocessing as mp
import pickle
import numpy as np
import torch

import torch.distributed as dist


def get_rank():
    global local_rank
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_world_size():
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = dist.get_world_size()
    if world_size == 1:
        return [data]

    output = [None for _ in range(world_size)]
    dist.all_gather_object(output, data, group=group)
    return output


class NumpySerializedList():
    def __init__(self, lst: list):
        def _serialize(data):
            buffer = pickle.dumps(data, protocol=-1)
            return np.frombuffer(buffer, dtype=np.uint8)

        print(
            "Serializing {} elements to byte tensors and concatenating them all ...".format(
                len(lst)
            )
        )
        self._lst = [_serialize(x) for x in lst]
        self._addr = np.asarray([len(x) for x in self._lst], dtype=np.int64)
        self._addr = np.cumsum(self._addr)
        self._lst = np.concatenate(self._lst)
        print("Serialized dataset takes {:.2f} MiB".format(len(self._lst) / 1024**2))

    def __len__(self):
        return len(self._addr)

    def __getitem__(self, idx):
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        bytes = memoryview(self._lst[start_addr:end_addr])
        return pickle.loads(bytes)


class TorchSerializedList(NumpySerializedList):
    def __init__(self, lst: list):
        super().__init__(lst)
        self._addr = torch.from_numpy(self._addr)
        self._lst = torch.from_numpy(self._lst)

    def __getitem__(self, idx):
        start_addr = 0 if idx == 0 else self._addr[idx - 1].item()
        end_addr = self._addr[idx].item()
        bytes = memoryview(self._lst[start_addr:end_addr].numpy())
        return pickle.loads(bytes)


def local_scatter(array: Optional[List[Any]]):
    """
    Scatter an array from local leader to all local workers.
    The i-th local worker gets array[i].

    Args:
        array: Array with same size of #local workers.
    """
    if get_world_size() == 1:
        # Just one worker. Do nothing.
        return array[0]
    if get_rank() == 0:
        assert len(array) == get_world_size()
        all_gather(array)
    else:
        all_data = all_gather(None)
        array = all_data[get_rank() - get_rank()]
    return array[get_rank()]


# NOTE: https://github.com/facebookresearch/mobile-vision/pull/120
# has another implementation that does not use tensors.
class TorchShmSerializedList(TorchSerializedList):
    def __init__(self, lst: list):
        if get_rank() == 0:
            super().__init__(lst)
            # Move data to shared memory, obtain a handle to send to each local worker.
            # This is cheap because a tensor will only be moved to shared memory once.
            handles = [None] + [
              bytes(mp.reductions.ForkingPickler.dumps((self._addr, self._lst)))
              for _ in range(get_world_size() - 1)]
        else:
            handles = None
        # Each worker receives the handle from local leader.
        handle = local_scatter(handles)

        if get_rank() > 0:
            # Materialize the tensor from shared memory.
            self._addr, self._lst = mp.reductions.ForkingPickler.loads(handle)
            print(f"Worker {get_rank()} obtains a dataset of length="
                  f"{len(self)} from its local leader.")

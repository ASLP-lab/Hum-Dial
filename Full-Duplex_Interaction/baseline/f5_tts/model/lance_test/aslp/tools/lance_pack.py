import lance
import sys
import datetime
import pyarrow
import pandas

from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.base import Base
from typing import List
def to_pyarrow_table_with_schema(data_list: List[Base], schema: pyarrow.Schema):
    """
    根据指定schema生成table
    """
    dataframe = pandas.DataFrame.from_dict([i.to_dict() for i in data_list])
    table = pyarrow.Table.from_pandas(dataframe, schema=schema)
    return table

def load(uri: str, auto_create: bool=False, target_cls: Base=Base) -> lance.LanceDataset:
    """
    加载数据集
    """
    if auto_create and not os.path.exists(uri):
        # print(target_cls)
        ds = create_new_dataset(uri, target_cls.get_example)
    else:
        ds = lance.dataset(uri)
    return ds

def create_new_dataset(uri: str, example_func) -> lance.LanceDataset:
    """
    按照示例数据创建新的lance dataset，需要提供创建示例数据的函数
    """
    example_data: Base = example_func()
    table = to_pyarrow_table_with_schema([example_data], example_data.get_schema())
    ds = lance.write_dataset(table, uri)
    ds.create_scalar_index(
        "data_id",
        "BTREE"
    )
    ds.delete('(`data_id` IS NOT NULL) OR (`data_id` IS NULL)')
    return ds

def ceildiv(a, b):
    return -(a // -b)


class LanceReader:
    def __init__(self, uri: str, filter_str: str=None, target_cls: Base=Base) -> None:
        self.ds = load(uri)
        self.filter = filter_str
        self.target_cls = target_cls

    def get_ids(self, ids_col: str="data_id", filter_str: str=None, raw: bool=False, progress: bool=False) -> List[Base]:
        """
        获取id列
        
        Return
        --------
        pandas.DataFrame
                        data_id
        0  aslp_example_data_12345
        1  aslp_example_data_12346
        """
        items = self.ds.to_table([ids_col], filter=filter_str, with_row_id=True).to_pandas()
        if raw:
            return items
        return self.target_cls.from_pandas(items, progress)

    def get_datas_by_rows(self, rows: List[int], cols: List[str]=None, raw: bool=False) -> List[Base]:
        """
        获取指定的一组数据
        
        返回值为一组元组组成的列表，每个元组中包含行号和对应的数据。如果行不存在，则对应的数据为None
        """
        table = self.ds.take(rows, columns=cols, with_row_id=True)
        items = table.to_pandas()
        if raw:
            return items
        else:
            return self.target_cls.from_pandas(items)

    def get_datas_by_rowids(self, row_ids: List[int], cols: List[str]=None, raw: bool=False) -> List[Base]:
        """
        这个API不稳定。除非你知道在干啥不然不要使用！推荐从行号获取数据而非rowid！
        """
        data_list = []
        table = self.ds._take_rows(row_ids=row_ids, columns=cols, with_row_id=True)
        items = table.to_pandas()
        if raw:
            return items
        else:
            return self.target_cls.from_pandas(items)

    def get_datas_by_filter(self, cols: List[str]=None, filter_str: str=None, raw: bool=False) -> List[Base]:
        """
        过滤数据。

        如果存在某个字段过大（超过5KB），过滤会非常慢！
        建议在该API处获取data_id和row_id再从get_data_by_rowids读取数据。
        """
        table = self.ds.to_table(cols, filter_str, with_row_id=True)
        items = table.to_pandas()
        if raw:
            return items
        else:
            return self.target_cls.from_pandas(items)


class LanceWriter(LanceReader):
    def __init__(self, uri: str, filter_str: str=None, target_cls: Base=Base) -> None:
        self.ds = load(uri, auto_create=True, target_cls=target_cls)
        self.uri = uri
        self.filter = filter_str
        self.target_cls = target_cls

    def insert(self, data_list: List[Base], index: str = "data_id"):
        """
        非必要不使用。
        插入数据。如果index不存在，则插入。
        """
        schema = self.target_cls.get_schema()
        table = to_pyarrow_table_with_schema(data_list, schema)
        
        return self.ds.merge_insert(index) \
                .when_not_matched_insert_all() \
                .execute(table, schema=schema)
    
    def update(self, data_list: List[Base], index: str = "data_id"):
        """
        更新数据。如果index存在，则进行内容更新。
        """
        for item in data_list:
            if isinstance(item, Base):
                item_dict = item.to_dict(only_not_none=True, str_wrap=True)
            elif isinstance(item, dict):
                item_dict = {}
                for k, v in item.items():
                    if v is not None:
                        item_dict[k] = v
            else:
                raise TypeError(f"Excepted item of data list with type {type(Base)} or dict, but got {type(item)}")
            index_value = item_dict.pop(index, None)
            if index_value is None:
                continue
            self.ds.update(item_dict, where=f"{index}={index_value}")

    
    def delete(self, ids: List[str], index: str = "data_id"):
        """
        通过data_id删除数据
        """
        list_str = ", ".join([f"'{i}'" for i in ids])
        self.ds.delete(
            f"`{index}` IN ({list_str})"
        )
        
    
    def write_parallel(self, data_list: List[Base], n_jobs: int=5, max_items_per_frag: int = 10_000, progress: bool=False, **kwargs):
        """
        并行写入，仅推荐以该方式追加写
        """
        part_size = ceildiv(len(data_list), n_jobs)
        part_size = min(part_size, max_items_per_frag)
        data_parts = []
        for i in range(n_jobs):
            start = i * part_size
            end = min((i + 1) * part_size, len(data_list))
            data_parts.append(data_list[start:end])

        fragments = []
        def task(data, **kwargs):
            new_frags = lance.fragment.write_fragments(
                to_pyarrow_table_with_schema(
                    data, self.target_cls.get_schema()
                ), 
                self.uri, **kwargs
            )
            fragments.extend(new_frags)

        threads = []
        executor = ThreadPoolExecutor(n_jobs)
        for part in data_parts:
            if len(part) >= 1:
                threads.append(executor.submit(partial(task, part, **kwargs)))

        if progress:
            threads = tqdm(progress)

        for t in threads:
            t.result()

        operation = lance.LanceOperation.Append(fragments)
        self.ds = lance.LanceDataset.commit(self.uri, operation, read_version=self.ds.latest_version)
    
    def clean_up(self):
        last_verion = self.ds.latest_version
        self.ds.checkout_version(last_verion)
        delta = datetime.datetime.now() - self.ds.versions()[-1]['timestamp']
        return self.ds.cleanup_old_versions(delta)
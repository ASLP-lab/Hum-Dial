import pandas
from pyarrow import parquet
from tqdm import tqdm

from data.base import Data

def load_from_parquet(data_path: str) -> list[Data]:
    table = parquet.read_table(data_path)
    table: pandas.DataFrame = table.to_pandas()
    data_list = []
    for idx, data in tqdm(table.iterrows()):
        data_list.append(Data.from_kwargs(**data))
    return data_list

def dump_to_parquet(data_path: str, data_list: list[Data], append: bool=False):
    table = pandas.DataFrame(data_list)
    parquet.write_table(table, data_path)
import multiprocessing as mp
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from requests import Session
from typing import List, OrderedDict
from multiprocessing.connection import Connection
import pandas as pd
import numpy as np
from math import ceil
import json
from pandas.io.json import json_normalize

URL = "http://deepshadow.gsslab.rdu2.redhat.com:8080/predictions/my_tc"
JSON_HEADER = {"Content-Type": "application/json"}


def requests_session() -> Session:
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[507],
        method_whitelist=["POST"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    sesh = Session()
    sesh.mount("http://", adapter)
    return sesh


def request_post(data: str, sesh: Session, conn: Connection, num: int):
    r = sesh.post(URL, data=bytes(data, "utf-8"), headers=JSON_HEADER)
    print(f"{num} status_code: {r.status_code}")
    conn.send(r)


def split_and_format(arr: np.array, length: int) -> List[List[str]]:
    splits = ceil(len(arr) / length)
    arr_list = np.array_split(arr.flatten(), splits)
    data_list = [
        f'{{"instances":{json.dumps([{"data": text} for text in arr])}}}'
        for arr in arr_list
    ]
    return data_list


def batch_inference(
    responses: np.array, indexed_labels: OrderedDict, batch_size: int
) -> pd.DataFrame:
    data_list = split_and_format(responses, batch_size)
    processes = []
    r_list = []
    sesh = requests_session()
    for num, batch in enumerate(data_list):
        output_pipe, input_pipe = mp.Pipe(duplex=False)
        proc = mp.Process(
            target=request_post, args=(batch, sesh, input_pipe, num)
        )
        processes.append(proc)
        r_list.append(output_pipe)
        proc.start()
    [proc.join() for proc in processes]
    r_list = [
        json_normalize(r.recv().json()["predictions"], sep="_") for r in r_list
    ]
    return pd.concat(r_list, ignore_index=True)

import pytorch_quik as pq
import multiprocessing as mp
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from requests import Session
from typing import List, OrderedDict
from multiprocessing.connection import Connection
import pandas as pd
import numpy as np
from math import ceil
from pandas.io.json import json_normalize
import logging
import sys

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

URL = "http://deepshadow.gsslab.rdu2.redhat.com:8080/predictions/my_tc"
JSON_HEADER = {"Content-Type": "application/json"}


def requests_session() -> Session:
    """Create an API session that can queue and recieve multiple
    requests. It can also retry when a request returns a 507 instead
    of a 200.

    Returns:
        Session: A requests session
    """
    retry_strategy = Retry(
        total=10,
        backoff_factor=1,
        status_forcelist=[507],
        method_whitelist=["POST"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    sesh = Session()
    sesh.mount("http://", adapter)
    return sesh


def request_post(data_batch: str, sesh: Session, conn: Connection, num: int):
    """send a POST request based on a requests_session and a connection

    Args:
        data_batch (str): A batch of data to be predicted
        sesh (Session): A session from requests_session
        conn (Connection): The input_pipe from mp.Pipe
        num (int): the batch number for when the data is recombined.
    """
    r = sesh.post(URL, data=bytes(data_batch, "utf-8"), headers=JSON_HEADER)
    logger.info(f"Batch {num}, status_code: {r.status_code}")
    conn.send(r)


def split_and_format(arr: np.array, length: int) -> List[str]:
    """Taking a numpy array of text, split into batches for separate API
    posts, and format into the torch serve required "instances" and "data."

    Args:
        arr (np.array): An array of text to be predicted
        length (int): The length of the batch, or batch size

    Returns:
        List[str]: A list of strings formatted for a Transformer handler.
    """
    splits = ceil(len(arr) / length)
    arr_list = np.array_split(arr.flatten(), splits)
    data_list = [pq.utils.txt_format(arr) for arr in arr_list]
    return data_list


def batch_inference(
    responses: np.array, indexed_labels: OrderedDict, batch_size: int
) -> pd.DataFrame:
    """Take an array of text fields, and return predictions via API

    Args:
        responses (np.array): The set of text (or survey responses) to
        be predicted
        indexed_labels (OrderedDict): An ordered dict of labels, for instance
        0: Negative, 1: Positive
        batch_size (int): The size of each batch request

    Returns:
        pd.DataFrame: A dataframe with the original text, logits, and
        predicted label.
    """
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

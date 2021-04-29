import pytest
import pytorch_quik as pq
from time import time, sleep
import re
from pathlib import Path


def test_sec_str():
    """print a sec_str"""
    start_time = time()
    sleep(2)
    msg = pq.utils.sec_str(start_time)
    assert re.match(r"^2.0\d? seconds$", msg)


def test_row_str(sample_data):
    """print a row_str"""
    assert pq.utils.row_str(sample_data.shape[0]) == "0.0M rows"


def test_id_str(args):
    """print an id str"""
    filename = "train_tensor_20210101.pt"
    filename = Path.cwd().joinpath("data", args.bert_type, filename)
    # assert(pq.utils.id_str("train", args) == output)
    output = pq.utils.id_str("train", args)
    print(output)
    assert filename == output

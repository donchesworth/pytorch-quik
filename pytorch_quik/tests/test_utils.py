import pytorch_quik as pq
from time import time, sleep
import re


def test_sec_str():
    """print a sec_str"""
    start_time = time()
    sleep(2)
    msg = pq.utils.sec_str(start_time)
    assert re.match(r"^2.0\d? seconds$", msg)


def test_row_str(sample_data):
    """print a row_str"""
    assert pq.utils.row_str(sample_data.shape[0]) == "0.0M rows"

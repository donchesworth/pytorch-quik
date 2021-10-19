import pytorch_quik as pq
from time import time, sleep
import re
from collections import OrderedDict


def test_sec_str():
    """print a sec_str"""
    start_time = time()
    sleep(2)
    msg = pq.utils.sec_str(start_time)
    assert re.match(r"^2.0\d? seconds$", msg)


def test_row_str(sample_data):
    """print a row_str"""
    assert pq.utils.row_str(sample_data.shape[0]) == "0.0M rows"


def test_indexed_classes(senti_classes):
    raw_classes = list(senti_classes.values())
    dir_classes = pq.utils.indexed_dict(raw_classes)
    assert isinstance(dir_classes, OrderedDict)
    assert dir_classes == senti_classes


def test_inv_classes(senti_classes, inv_senti_classes):
    inv_classes = pq.utils.inverse_dict(senti_classes)
    assert isinstance(inv_classes, OrderedDict)
    assert inv_classes == inv_senti_classes


def test_txt_format():
    raw_txt = ["Great company with fast support",
               "Not great company with slow support"]
    txt = pq.utils.txt_format(raw_txt)
    output = '{"instances":[{"data": "Great company with fast support"}, ' \
        '{"data": "Not great company with slow support"}]}'
    assert txt == output

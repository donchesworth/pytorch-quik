import pytorch_quik as pq
from pathlib import Path
import torch
import os


def test_id_str(args, test_file):
    """print an id str"""
    os.chdir(Path(__file__).parent)
    output = pq.io.id_str("test", args)
    assert test_file == output


def test_load_str(args, test_file):
    """print an id str"""
    os.chdir(Path(__file__).parent)
    output = pq.io.load_torch_object("test", args, location=True)
    assert test_file == output


def test_load_torch_tensor(args):
    """use load_torch_object to load a tensor"""
    os.chdir(Path(__file__).parent)
    tt = pq.io.load_torch_object("test", args)
    assert tt.equal(torch.tensor([[1, 2, 3], [4, 5, 6]]))

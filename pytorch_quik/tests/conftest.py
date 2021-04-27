from pathlib import Path
import pytest
import pandas as pd
import torch
import json
from argparse import Namespace
from os import system
import warnings

TESTDIR = Path(__file__).parent
SAMPLE = TESTDIR.joinpath("sample_data.json")
FINAL = TESTDIR.joinpath("final_data.json")


def pytest_generate_tests(metafunc):
    metafunc.parametrize("gpus", [0, 1])


@pytest.fixture
def args(gpus):
    """sample args namespace"""
    args = Namespace()
    args.gpus = gpus
    args.data_date = 20210101
    args.has_gpu = system("nvidia-smi -L") == 0
    if not args.has_gpu:
        warnings.warn(
            "GPU not found, setting has_gpu to False. \
            Some tests will be skipped"
        )
        args.device = torch.device('cpu')
    else:
        args.device = torch.device('cuda', 0)
    args.nr = 0
    args.nodes = 1
    args.bs = 12
    args.num_workers = 0
    args.learning_rate = 5e-5
    args.weight_decay = 0
    args.find_unused_parameters = False
    args.bert_type = "roberta"
    return args


@pytest.fixture(scope="session")
def sample_tensor():
    """sample tensor dataset"""
    return torch.rand(200, 64)


@pytest.fixture(scope="session")
def sample_labels():
    """labels tensor"""
    return (torch.rand(200) < 0.5).int()


@pytest.fixture(scope="session")
def sample_data():
    """sample user/item dataset"""
    with open(SAMPLE) as f:
        df = pd.DataFrame(json.load(f))
    return df


@pytest.fixture(scope="session")
def final_data():
    """final user/item dataset"""
    with open(FINAL) as f:
        df = pd.DataFrame(json.load(f))
    df.index.names = ["ui_index"]
    return df


@pytest.fixture(scope="session")
def ttypes():
    """tensor data types for transformation"""
    ttypes = {
        0: torch.int16,
        1: torch.int16
    }
    return ttypes

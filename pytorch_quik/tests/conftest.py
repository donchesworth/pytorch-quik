from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import torch
import json
from argparse import Namespace
from os import system
import warnings
from collections import OrderedDict

TESTDIR = Path(__file__).parent
SAMPLE = TESTDIR.joinpath("sample_data.json")
FINAL = TESTDIR.joinpath("final_data.json")


# def pytest_generate_tests(metafunc):
#     metafunc.parametrize("gpus", [0, 1])
@pytest.fixture(params=[0, 1])
def gpus(request):
    return request.param


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


@pytest.fixture(scope='session')
def senti_classes():
    """sentiment classes"""
    dir_classes = OrderedDict([(0, 'Negative'), (1, 'Neutral'), (2, 'Positive')])
    return dir_classes


@pytest.fixture(scope='session')
def inv_senti_classes():
    """inverse sentiment classes"""
    inv_classes = OrderedDict([('Negative', 0), ('Neutral', 1), ('Positive', 2)])
    return inv_classes


@pytest.fixture(scope="session")
def sample_tensor():
    """sample tensor dataset"""
    torch.manual_seed(0)
    return torch.rand(200, 64)


@pytest.fixture(scope="session")
def sample_preds():
    """labels tensor"""
    torch.manual_seed(0)
    p = torch.rand(200, 3)
    # https://stackoverflow.com/questions/59090533/how-do-i-add-some-gaussian
    # -noise-to-a-tensor-in-pytorch
    noise = torch.zeros(200, 3, dtype=torch.float64)
    noise = noise + (0.1**0.9)*torch.randn(200, 3)
    return p + noise


@pytest.fixture(scope="session")
def sample_labels():
    """labels tensor"""
    torch.manual_seed(0)
    labels = torch.rand(200, 3)
    labels = np.argmax(labels, axis=1).flatten()
    # torch.manual_seed(0)
    # return (torch.rand(200) < 0.5).int()
    return labels


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
def final_cmdf(senti_classes):
    """final confusion matrix data frame"""
    arr = np.array([[63,  8,  7], [6, 48,  4], [2,  8, 54]])
    classes = senti_classes.values()
    aidx = pd.MultiIndex.from_product([['Actual'], classes])
    pidx = pd.MultiIndex.from_product([['Predicted'], classes])
    df = pd.DataFrame(arr, aidx, pidx)
    return df


@pytest.fixture(scope="session")
def ttypes():
    """tensor data types for transformation"""
    ttypes = {
        0: torch.int16,
        1: torch.int16
    }
    return ttypes

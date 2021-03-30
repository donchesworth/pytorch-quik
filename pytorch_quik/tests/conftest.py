from pathlib import Path
import pytest
import pandas as pd
import json
from argparse import Namespace
from os import system
import warnings

TESTDIR = Path(__file__).parent
SAMPLE = TESTDIR.joinpath("sample_data.json")
FINAL = TESTDIR.joinpath("final_data.json")


@pytest.fixture
def args():
    """sample args namespace"""
    args = Namespace()
    args.data_date = 20210101
    args.has_gpu = system("nvidia-smi -L") == 0
    if not args.has_gpu:
        warnings.warn(
            "GPU not found, setting has_gpu to False. \
            Some tests will be skipped"
        )
    args.bert_type = 'roberta'
    return args


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

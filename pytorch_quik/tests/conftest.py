from pathlib import Path
import pytest
from pytorch_quik import arg
from pytorch_quik.mlflow import QuikMlflow
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd
import numpy as np
import torch
import json
from os import getenv
import warnings
from collections import OrderedDict
import sys


# bd = Path("/workspaces/rdp-vscode-devcontainer/pytorch-quik")
# TESTDIR = bd.joinpath("pytorch_quik", "tests")
TESTDIR = Path(__file__).parent
SAMPLE = TESTDIR.joinpath("sample_data.json")
FINAL = TESTDIR.joinpath("final_data.json")
ENCODING = TESTDIR.joinpath("sample_encoding.pt")
AMASK = TESTDIR.joinpath("sample_amask.pt")
TRACKING_URI = getenv("TRACKING_URI", "https://localhost:5000")
ENDPOINT_URL = getenv("ENDPOINT_URL", None)
MLUSER = getenv("MLUSER", None)
IS_CI = getenv("CI", "false")


def pytest_collection_modifyitems(items):
    skipif_mlflow = pytest.mark.skipif(
        IS_CI == "true", reason="no mlflow server access"
    )
    skipif_mlflow_partial = pytest.mark.skipif(
        IS_CI == "true", reason="no mlflow server access"
    )
    skipif_gpus = pytest.mark.skipif(
        IS_CI == "true", reason="no GPU for test version"
    )
    for item in items:
        if "skip_mlflow" in item.keywords:
            item.add_marker(skipif_mlflow)
        # [True- is when test_mlflow = True
        if "skip_mlflow_partial" in item.keywords and "[True-" in item.name:
            item.add_marker(skipif_mlflow_partial)
        # -0] is when gpu = 0
        if "skip_gpus" in item.keywords and "0]" in item.name:
            item.add_marker(skipif_gpus)


@pytest.fixture(params=[None, 0])
def gpu(request):
    return request.param


@pytest.fixture(params=[True, False])
def test_mlflow(request):
    return request.param


@pytest.fixture
def clean_run():
    def clean_run_function(mlf, gpu):
        mlf.client.delete_run(mlf.runid)
        if gpu == 0:
            mlf.client.delete_experiment(mlf.expid)

    return clean_run_function


@pytest.fixture
def create_qml():
    def create_qml_function(args):
        args.experiment = "pytest"
        mlf = QuikMlflow(args)
        exp = mlf.client.get_experiment(mlf.expid)
        if exp.lifecycle_stage == "deleted":
            mlf.client.restore_experiment(mlf.expid)
        return mlf

    return create_qml_function


@pytest.fixture
def args(gpu):
    """sample args namespace"""
    sys.argv = [""]
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    MLKWARGS = {
        "user": MLUSER,
        "tracking_uri": TRACKING_URI,
        "endpoint_url": ENDPOINT_URL,
    }
    parser = arg.add_ddp_args(parser)
    parser = arg.add_learn_args(parser)
    parser = arg.add_mlflow_args(parser, kwargs=MLKWARGS)
    parser = arg.add_ray_tune_args(parser)
    args = parser.parse_args()
    args.mixed_precision = False
    args.bert_type = "roberta"
    args.data_date = 20210101
    args.gpu = gpu
    if args.gpu is not None:
        args.has_gpu = True
        args.gpus = 1
    else:
        args.has_gpu = False
        args.gpus = 0
        warnings.warn(
            "GPU not found, setting has_gpu to False. \
            Some tests will be skipped"
        )
    args.num_workers = 2
    args.use_init_group = True
    args.use_mlflow = False
    args.use_ray_tune = False
    args.sched_kwargs = {
        "num_warmup_steps": 10,
    }
    return args


@pytest.fixture
def test_file(args):
    fn = f"test_tensor_{args.data_date}.pt"
    fn = Path(__file__).parent.joinpath("data", args.bert_type, fn)
    return fn


@pytest.fixture(scope="session")
def senti_classes():
    """sentiment classes"""
    dir_classes = OrderedDict(
        [(0, "Negative"), (1, "Neutral"), (2, "Positive")]
    )
    return dir_classes


@pytest.fixture(scope="session")
def two_classes():
    """sentiment classes"""
    two_classes = OrderedDict([(0, "Negative"), (1, "Positive")])
    return two_classes


@pytest.fixture(scope="session")
def inv_senti_classes():
    """inverse sentiment classes"""
    inv_classes = OrderedDict(
        [("Negative", 0), ("Neutral", 1), ("Positive", 2)]
    )
    return inv_classes


@pytest.fixture(scope="session")
def sample_tensor():
    """sample tensor"""
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
    noise = noise + (0.1 ** 0.9) * torch.randn(200, 3)
    return p + noise


@pytest.fixture(scope="session")
def sample_tds(sample_tensor, sample_preds):
    """sample tensor dataset"""
    return torch.utils.data.TensorDataset(
        sample_tensor, sample_tensor, sample_preds
    )


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
def eight_labels():
    """labels list for 9 categories"""
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    return labels


@pytest.fixture(scope="session")
def sample_data():
    """sample user/item dataset"""
    with open(SAMPLE) as f:
        df = pd.DataFrame(json.load(f))
    return df


@pytest.fixture(scope="session")
def sample_encoding():
    """sample encoded tensors from tokenizer"""
    enc = {
        "input_ids": torch.load(ENCODING),
        "attention_mask": torch.load(AMASK),
    }
    return enc


@pytest.fixture(scope="session")
def sample_ins(sample_encoding):
    """sample encoded tensor input_ids from tokenizer"""
    ins = sample_encoding["input_ids"]
    return ins


@pytest.fixture(scope="session")
def sample_amask(sample_encoding):
    """sample encoded tensor attention_masks from tokenizer"""
    amask = sample_encoding["attention_mask"]
    return amask


@pytest.fixture(scope="session")
def final_data():
    """final user/item dataset"""
    with open(FINAL) as f:
        df = pd.DataFrame(json.load(f))
    df.index.names = ["ui_index"]
    return df


@pytest.fixture(scope="session")
def tens_labels(final_data):
    """sample tensor labels from final user/item dataset"""
    tens = torch.LongTensor(final_data.label.values)
    return tens


@pytest.fixture(scope="session")
def batch(sample_ins, sample_amask, tens_labels):
    """sample batch for model training"""
    return [sample_ins, sample_amask, tens_labels]


@pytest.fixture(scope="session")
def final_cmdf(senti_classes):
    """final confusion matrix data frame"""
    arr = np.array([[63, 8, 7], [6, 48, 4], [2, 8, 54]])
    classes = senti_classes.values()
    aidx = pd.MultiIndex.from_product([["Actual"], classes])
    pidx = pd.MultiIndex.from_product([["Predicted"], classes])
    df = pd.DataFrame(arr, aidx, pidx)
    return df


@pytest.fixture(scope="session")
def ttypes():
    """tensor data types for transformation"""
    ttypes = {0: torch.int16, 1: torch.int16}
    return ttypes

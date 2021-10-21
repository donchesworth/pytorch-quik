from pytorch_quik.travel import (
    QuikTrek,
    QuikMlflow,
    DlKwargs,
    OptKwargs,
    World,
)
from mlflow.tracking import MlflowClient
from mlflow.entities import Run
from pytorch_quik.ddp import cleanup
import dask_quik as dq
import sys
import pytest


def test_quik_trek(args):
    if bool(args.gpus) and dq.utils.gpus() == 0:
        print("unable to test quik_trek portion")
        pytest.skip()
    qt = QuikTrek(args.gpu, args)
    assert isinstance(qt, QuikTrek)
    assert isinstance(qt.epochs, int)
    assert isinstance(qt.dlkwargs, DlKwargs)
    assert isinstance(qt.optkwargs, OptKwargs)
    assert isinstance(qt.world, World)
    if args.gpu is not None:
        cleanup()


def test_quik_trek_noargs(gpu):
    if gpu and dq.utils.gpus() == 0:
        print("unable to test quik_trek no args portion")
        pytest.skip()
    sys.argv = ['']
    qt = QuikTrek(gpu)
    assert isinstance(qt, QuikTrek)
    assert isinstance(qt.epochs, int)
    assert isinstance(qt.dlkwargs, DlKwargs)
    assert isinstance(qt.optkwargs, OptKwargs)
    assert isinstance(qt.world, World)
    if gpu is not None:
        cleanup()


def test_quik_trek_mlf(args, create_qml, clean_run):
    if bool(args.gpus) and dq.utils.gpus() == 0:
        print("unable to test quik_trek mlflow portion")
        pytest.skip()
    args.use_mlflow = True
    _ = create_qml(args)
    qt = QuikTrek(args.gpu, args)
    assert isinstance(qt, QuikTrek)
    assert isinstance(qt.mlflow, QuikMlflow)
    assert isinstance(qt.mlflow.client, MlflowClient)
    assert isinstance(qt.mlflow.run, Run)
    clean_run(qt.mlflow, args.gpu)
    if args.gpu is not None:
        cleanup()

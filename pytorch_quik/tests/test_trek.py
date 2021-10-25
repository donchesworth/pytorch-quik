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
import sys
import pytest


@pytest.mark.skip_gpus
def test_quik_trek(args):
    sys.argv = ['']
    qt = QuikTrek(args.gpu, args)
    assert isinstance(qt, QuikTrek)
    assert isinstance(qt.epochs, int)
    assert isinstance(qt.dlkwargs, DlKwargs)
    assert isinstance(qt.optkwargs, OptKwargs)
    assert isinstance(qt.world, World)
    if args.gpu is not None:
        cleanup()


@pytest.mark.skip_gpus
def test_quik_trek_noargs(gpu):
    sys.argv = ['']
    qt = QuikTrek(gpu)
    assert isinstance(qt, QuikTrek)
    assert isinstance(qt.epochs, int)
    assert isinstance(qt.dlkwargs, DlKwargs)
    assert isinstance(qt.optkwargs, OptKwargs)
    assert isinstance(qt.world, World)
    if gpu is not None:
        cleanup()


@pytest.mark.skip_mlflow
def test_quik_trek_mlf(args, create_qml, clean_run):
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

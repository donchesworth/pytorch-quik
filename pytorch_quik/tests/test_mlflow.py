from pytorch_quik.mlflow import QuikMlflow
from mlflow.tracking import MlflowClient


def test_quik_mlflow(args):
    mlf = QuikMlflow(args)
    assert isinstance(mlf, QuikMlflow)
    assert isinstance(mlf.client, MlflowClient)
    assert mlf.expid is not None


from pytorch_quik.mlflow import QuikMlflow
from pytorch_quik.travel import DlKwargs
from mlflow.tracking import MlflowClient
from mlflow.store.entities import PagedList
from mlflow.entities import Run
import pytest
import logging
from botocore.exceptions import NoCredentialsError

INFO0 = "please supply either a run_id or filter_string"
INFO1 = "query returned multiple runs, please update the filter"


@pytest.mark.skip_mlflow
def test_quik_mlflow(args, create_qml):
    # testing a new experiment, then an existing one
    mlf = create_qml(args)
    assert isinstance(mlf, QuikMlflow)
    assert isinstance(mlf.client, MlflowClient)
    assert mlf.expid is not None
    if args.gpu == 0:
        mlf.client.delete_experiment(mlf.expid)


@pytest.mark.skip_mlflow
def test_quik_mlflow_run(args, create_qml, clean_run):
    args.experiment = "pytest"
    dlk = DlKwargs()
    mlf = create_qml(args)
    mlf.create_run([dlk])
    mlf.end_run()
    mlf.update_run_status("RUNNING")
    mlf.end_run()
    assert isinstance(mlf.run, Run)
    clean_run(mlf, args.gpu)


@pytest.mark.skip_mlflow
def test_quik_mlflow_logging(args, create_qml, clean_run):
    args.experiment = "pytest"
    mlf = create_qml(args)
    dlk = DlKwargs()
    mlf.create_run([dlk])
    mlf.log_metric("metric1", 1)
    clean_run(mlf, args.gpu)


@pytest.mark.skip_mlflow
def test_quik_mlflow_search(args, create_qml, clean_run):
    args.experiment = "pytest"
    mlf = create_qml(args)
    dlk = DlKwargs()
    mlf.create_run([dlk])
    mlf.log_metric("metric1", 1)
    out = mlf.search_runs("metrics.metric1 = 1")
    assert isinstance(out, PagedList)
    assert isinstance(out[0], Run)
    assert out[0].data.metrics["metric1"] == 1.0
    clean_run(mlf, args.gpu)


@pytest.mark.skip_mlflow
def test_some_quik_mlflow_state_dict(args, create_qml, caplog):
    mlf = create_qml(args)
    runs = mlf.search_runs("metrics.metric1 = 1")
    run_id = runs[0].info.run_id
    with caplog.at_level(logging.INFO):
        # neither
        ret_none = mlf.get_state_dict(4)
        assert INFO0 == caplog.records[0].message
        assert ret_none is None
        # too many
        ret_none = mlf.get_state_dict(4, filter_string="metrics.metric1 = 1")
        assert INFO1 == caplog.records[1].message
        assert ret_none is None
        # error
        mlf.client.log_metric(run_id, "metric1", 2.5)
        with pytest.raises(NoCredentialsError) as e_info:
            ret_none = mlf.get_state_dict(
                4, filter_string="metrics.metric1 = 2.5"
                )
        assert e_info.value.fmt == "Unable to locate credentials"
        with pytest.raises(NoCredentialsError) as e_info:
            ret_none = mlf.get_state_dict(4, run_id=run_id)
        assert e_info.value.fmt == "Unable to locate credentials"
        mlf.client.log_metric(run_id, "metric1", 1)
    if args.gpu == 0:
        mlf.client.delete_experiment(mlf.expid)

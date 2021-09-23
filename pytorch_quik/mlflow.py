try:
    from mlflow.tracking import MlflowClient
except ImportError:
    pass
import os
from dataclasses import dataclass, asdict, is_dataclass
from typing import Dict, Optional, Union
from argparse import Namespace
from pathlib import Path
from . import io
from types import SimpleNamespace
import logging
import sys

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class MlfKwargs:
    """MLFlow keyword arguments"""

    tracking_uri: str
    endpoint_url: str
    experiment: str
    user: str
    use_ray: str
    is_parent: bool
    parent_run: str


class QuikMlflow:
    """A class to manage MLflow API calls. You'll have to add your own
    credentials either to ~/.aws/credentials, or store them in environment
    variables like:
        os.environ["AWS_ACCESS_KEY_ID"] = ''
        os.environ["AWS_SECRET_ACCESS_KEY"] = ''
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = ''
    """

    def __init__(
        self,
        args: Namespace,
    ):
        """Constructor, primarily adding MLFlow arguments"""
        self.mlfkwargs = MlfKwargs(
            tracking_uri=args.tracking_uri,
            endpoint_url=args.endpoint_url,
            experiment=args.experiment,
            user=os.environ.get("USER", "unknown"),
            use_ray=getattr(args, "use_ray", False),
            is_parent=getattr(args, "is_parent", False),
            parent_run=getattr(args, "parent_run", None),
        )
        if "MLFLOW_S3_ENDPOINT_URL" not in os.environ:
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = self.mlfkwargs.endpoint_url
        self.client = MlflowClient(tracking_uri=self.mlfkwargs.tracking_uri)
        exp = self.client.get_experiment_by_name(self.mlfkwargs.experiment)
        if exp is None:
            self.expid = self.client.create_experiment(
                self.mlfkwargs.experiment
            )
        else:
            self.expid = exp.experiment_id

    def create_run(self, params: list):
        """Using MLFLow Tracking create_fun, plus adding tags, parameters,
        and saving the run_id."""
        self.tags = self.add_tags(tags={})
        self.run = self.client.create_run(self.expid, tags=self.tags)
        self.runid = self.run.info.run_id
        if not self.mlfkwargs.is_parent:
            self.log_parameters(params)

    def add_tags(
        self,
        tags: Dict[str, str],
    ) -> Dict[str, str]:
        """Adding tags for MLFLow create run."""
        tags = {"mlflow.user": self.mlfkwargs.user}
        if self.mlfkwargs.parent_run is not None:
            tags["mlflow.parentRunId"] = self.mlfkwargs.parent_run
        if self.mlfkwargs.use_ray:
            tags["mlflow.source.name"] = "ray tune"
        return tags

    def log_parameters(self, dclasses):
        """Add parameters to MLFlow run using the dataclasses object"""
        for dclass in dclasses:
            if is_dataclass(dclass):
                dclass = asdict(dclass)
            _ = [
                self.client.log_param(self.runid, k, v)
                for k, v in dclass.items()
            ]

    def log_metric(
        self, key: str, metric: Union[str, int], step: Optional[int] = None
    ):
        """Log an MLFlow metric"""
        self.client.log_metric(self.runid, key, metric, step=step)

    def log_artifact(self, filename: Union[str, Path]):
        """Log an MLFlow artifact"""
        self.client.log_artifact(self.runid, str(filename))

    def end_run(self):
        """Mark an MLFlow run as completed"""
        self.client.set_terminated(self.runid)

    def update_run_status(self, status: str):
        """Use the MLFlow set_terminated function to instead mark an MLFlow
        run a new status"""
        self.client.set_terminated(self.runid, status)

    def search_runs(
        self,
        filter_string: Optional[str] = None,
        max_results: Optional[int] = None,
        order_by: Optional[str] = None,
    ) -> dict:
        """This allows you to search existing runs by a parameter (params.*)
        or a metric (metrics.*), for instance filter_string="params.lr =
        '1.6e-06'"""
        out = self.client.search_runs(
            self.expid,
            filter_string=filter_string,
            max_results=max_results,
            order_by=order_by,
        )
        return out

    def get_state_dict(
        self,
        test_epoch,
        run_id: Optional[str] = None,
        filter_string: Optional[str] = None,
        ioargs: Optional[Namespace] = None,
        serve_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """This function will pull a state_dict artifact from a completed
        MLFlow run."""
        if run_id is None:
            if filter_string is None:
                logger.info("please supply either a run_id or filter_string")
                return None
            runs = self.search_runs(filter_string)
            if len(runs) > 1:
                logger.info(
                    "query returned multiple runs, please update the filter"
                )
                return None
            else:
                run_id = runs[0].info.run_id
        if serve_path is None:
            if ioargs is None:
                ioargs = SimpleNamespace(bert_type="bert", source="nps")
                serve_path = io.id_str("", ioargs).parent.joinpath("serve")
        epoch_str = f"_e{test_epoch}_"
        arts = self.client.list_artifacts(run_id)
        for art in arts:
            if epoch_str in art.path:
                path = art.path
        self.client.download_artifacts(
            run_id, path, dst_path=serve_path.parent
        )
        return path.split(epoch_str)[1].split(".")[0]

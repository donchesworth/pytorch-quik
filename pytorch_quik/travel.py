import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
import torch.utils.data.distributed as ddist
import torch.utils.data as td
from torch import nn, optim
from pytorch_quik import arg, ddp, io, metrics
from contextlib import nullcontext
from argparse import ArgumentParser, Namespace
from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass, field, asdict, is_dataclass
import os

try:
    from mlflow.tracking import MlflowClient
except ImportError:
    pass


@dataclass
class World:
    """World related data."""

    device: torch.device = field(init=False)
    node_id: int = 0
    total_nodes: int = 1
    gpu_id: int = None
    total_gpus: int = None
    rank_id: int = field(init=False)
    world_size: int = field(init=False)
    is_ddp: bool = field(init=False)
    is_logger: bool = field(init=False)

    def __post_init__(self):
        if self.gpu_id is None:
            self.device = torch.device("cpu")
            self.rank_id = None
            self.world_size = None
            self.is_ddp = False
        else:
            self.device = torch.device("cuda", self.gpu_id)
            self.rank_id = self.node_id * self.total_gpus + self.gpu_id
            self.world_size = self.total_gpus * self.total_nodes
            self.is_ddp = True
        self.is_logger = not self.is_ddp
        if self.gpu_id == 0:
            self.is_logger = True


@dataclass
class DlKwargs:
    """Data loader keyword arguments."""

    batch_size: int = 24
    shuffle: bool = False
    pin_memory: bool = True
    num_workers: int = 0


@dataclass
class OptKwargs:
    """Optimizer keyword arguments"""

    lr: int
    weight_decay: int
    eps: int
    betas: tuple


class QuikTrek:
    """A class for maintaining the general data for the full trek to
    be shared between travelers.
    """

    def __init__(
        self, args: Optional[Namespace] = None, gpu: Optional[int] = None
    ):
        if args is None:
            parser = arg.add_learn_args(ArgumentParser())
            args = parser.parse_args()
        self.args = args
        self.epochs = args.epochs
        self.create_dataclasses(args, gpu)
        self.trek_prep(args)

    def create_dataclasses(self, args, gpu):
        self.world = World(args.nr, args.nodes, gpu, args.gpus)
        self.dlkwargs = DlKwargs(
            batch_size=args.bs,
            num_workers=args.num_workers,
        )
        self.optkwargs = OptKwargs(
            lr=args.lr,
            weight_decay=args.weight_decay,
            eps=args.eps,
            betas=args.betas,
        )
        self.args.device = self.world.device

    def trek_prep(self, args):
        if args.use_mlflow:
            self.mlflow = QuikMlflow(
                args.experiment,
                args.tracking_uri,
                args.endpoint_url,
                self.world,
                self.dlkwargs,
                self.optkwargs,
            )
        if self.world.device.type == "cuda":
            torch.cuda.empty_cache()
        if self.world.gpu_id is not None:
            torch.cuda.set_device(self.world.device)
            ddp.setup(self.world.gpu_id, self.world)


class QuikTraveler:
    """A class for traversing a model either in training, validation, or
    testing. Is always a singular run - but can be part of a multi-GPU run.
    """

    metrics = metrics.LossMetrics(0.99)

    def __init__(self, trek, type: Optional[str] = None):
        self.type = type
        self.world = trek.world
        self.args = trek.args
        self.trek = trek
        self.find_unused_parameters = trek.args.find_unused_parameters
        self.amp = QuikAmp(trek.args.mixed_precision)

    def set_criterion(
        self,
        criterion_fcn: Callable[..., nn.Module],
        kwargs: Optional[Dict[str, Any]] = {},
    ):
        if callable(criterion_fcn):
            self.criterion = criterion_fcn(**kwargs)
            self.criterion.to(self.world.device)

    def set_optimizer(
        self,
        optimizer_fcn: Callable[..., optim.Optimizer],
        kwargs: Optional[Dict[str, Any]] = {},
    ):
        if hasattr(self.model, "module"):
            self.optimizer = optimizer_fcn(
                self.model.module.parameters(),
                **asdict(self.trek.optkwargs),
                **kwargs,
            )
        else:
            self.optimizer = optimizer_fcn(
                self.model.parameters(),
                **asdict(self.trek.optkwargs),
                **kwargs,
            )

    def set_scheduler(
        self,
        scheduler_fcn: Callable[..., optim.Optimizer],
        kwargs: Optional[Dict[str, Any]] = {},
    ):
        self.scheduler = scheduler_fcn(
            self.optimizer,
            **kwargs,
        )

    def add_model(
        self,
        model: nn.Module,
        state_dict: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.model = model
        if state_dict is not None:
            self.model.load_state_dict(state_dict)
        self.model.to(self.world.device)
        if self.world.is_ddp:
            self.model = DDP(
                self.model,
                device_ids=[self.world.device],
                output_device=self.world.device,
                find_unused_parameters=self.find_unused_parameters,
            )

    def add_data(self, tensorDataset: td.TensorDataset):
        self.data = QuikData(
            tensorDataset, self.world, self.trek.dlkwargs, self.trek.epochs
        )
        if self.type == "train":
            self.metrics.steps = self.data.steps

    def backward(self, loss: torch.Tensor, clip: Optional[bool] = True):
        if hasattr(self.amp, "scaler"):
            self.amp.backward(self, loss, clip)
        else:
            loss.backward()
            if hasattr(self.amp, "optimizer"):
                self.optimizer.step()

    def add_loss(self, loss, step, epoch):
        self.metrics.add_loss(loss)
        if self.args.use_mlflow:
            loss = self.metrics.metric_dict["train_loss"]
            step = step + (self.metrics.steps * epoch)
            self.trek.mlflow.log_metric("train_loss", loss, step)

    def add_vloss(self, vlosses, nums, epoch):
        self.metrics.add_vloss(vlosses, nums)
        if self.args.use_mlflow:
            vloss = self.metrics.metric_dict["valid_loss"]
            step = self.metrics.steps * (epoch + 1)
            self.trek.mlflow.log_metric("valid_loss", vloss, step)

    def save_state_dict(self, epoch):
        sd_id = io.save_state_dict(self.model, self.args, epoch)
        if self.args.use_mlflow:
            self.trek.mlflow.log_artifact(str(sd_id))

    def record_results(
        self,
        label_names,
        accuracy: Optional[bool] = True,
        f1: Optional[bool] = True,
        confusion: Optional[bool] = True,
    ):
        cm_id = io.id_str("confusion", self.args)
        self.cm = metrics.build_confusion_matrix(
            self.data.predictions,
            self.data.labels,
            label_names,
            cm_id,
        )
        cr = metrics.build_class_dict(
            self.data.predictions,
            self.data.labels,
            label_names,
        )
        self.trek.mlflow.log_artifact(cm_id)
        {
            self.trek.mlflow.log_metric(metric, value, 0)
            for metric, value in cr.items()
        }


class QuikData:
    """A class for providing data to a traveler."""

    def __init__(
        self,
        tensorDataset: td.TensorDataset,
        world: World,
        dlkwargs: Dict[str, Any],
        epochs: int,
    ):
        self.dataset = tensorDataset
        self.labels = self.dataset.tensors[2].cpu().numpy()
        self.dlkwargs = dlkwargs
        self.add_data_loader(world)
        self.steps = len(self.data_loader)
        self.total_steps = self.steps * epochs

    def add_sampler(self, world, sampler_fcn=None, kwargs={}):
        if world.is_ddp:
            self.sampler = ddist.DistributedSampler(
                self.dataset,
                num_replicas=world.world_size,
                rank=world.rank_id,
            )
        elif callable(sampler_fcn):
            self.sampler = sampler_fcn(**kwargs)
        else:
            self.sampler = sampler_fcn

    def add_data_loader(self, world):
        if not hasattr(self, "sampler"):
            self.add_sampler(world)
        self.data_loader = td.DataLoader(
            dataset=self.dataset,
            sampler=self.sampler,
            **asdict(self.dlkwargs),
        )

    def add_results(self, predictions: torch.Tensor, labels: torch.Tensor):
        self.predictions = predictions
        self.labels = labels


class QuikAmp:
    """A class to manage automatic mixed precision. Provides
    a nullcontext to your forward function if it's not being
    used.
    """

    def __init__(self, mixed_precision: bool):
        if mixed_precision:
            self.scaler = GradScaler()
            self.caster = autocast()
        else:
            self.caster = nullcontext()

    def backward(
        self,
        trvlr: QuikTraveler,
        loss: torch.Tensor,
        clip: Optional[bool] = True,
    ):
        self.scaler.scale(loss).backward()
        # https://pytorch.org/docs/stable/notes/amp_examples.html#working-with-unscaled-gradients
        if clip:
            self.scaler.unscale_(trvlr.optimizer)
            # https://discuss.pytorch.org/t/about-torch-nn-utils-clip-grad-norm/13873
            if hasattr(trvlr.model, "module"):
                clip_grad_norm_(trvlr.model.module.parameters(), 1.0)
            else:
                clip_grad_norm_(trvlr.model.parameters(), 1.0)
        self.scaler.step(trvlr.optimizer)
        self.scaler.update()


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
        experiment,
        tracking_uri,
        endpoint_url,
        world,
        dlkwargs,
        optkwargs,
    ):
        if "MLFLOW_S3_ENDPOINT_URL" not in os.environ:
            os.environ["MLFLOW_S3_ENDPOINT_URL"] = endpoint_url
        self.client = MlflowClient(tracking_uri=tracking_uri)
        exp = self.client.get_experiment_by_name(experiment)
        if exp is None:
            self.expid = self.client.create_experiment(experiment)
        else:
            self.expid = exp.experiment_id
        self.run = self.client.create_run(self.expid)
        self.runid = self.run.info.run_id
        self.log_parameters([world, dlkwargs, optkwargs])

    def log_parameters(self, dclasses):
        for dclass in dclasses:
            if is_dataclass(dclass):
                dclass = asdict(dclass)
            _ = [
                self.client.log_param(self.runid, k, v)
                for k, v in dclass.items()
            ]

    def log_metric(self, key, metric, step):
        self.client.log_metric(self.runid, key, metric, step=step)

    def log_artifact(self, sd_id):
        self.client.log_artifact(self.runid, str(sd_id))

    def end_run(self):
        self.client.set_terminated(self.runid)

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_
import torch.utils.data.distributed as ddist
import torch.utils.data as td
from torch import nn, optim
from pytorch_quik import args, ddp, metrics
from contextlib import nullcontext
from argparse import ArgumentParser, Namespace
from typing import Any, Callable, Dict, Optional
from dataclasses import dataclass, field, asdict


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


class QuikTraveler:
    """A class for traversing a model either in training, validation, or
    testing. Is always a singular run - but can be part of a multi-GPU run.
    """

    metrics = metrics.LossMetrics(0.99)

    def __init__(
        self, argp: Optional[Namespace] = None, gpu: Optional[int] = None
    ):
        if argp is None:
            parser = args.add_learn_args(ArgumentParser())
            argp = parser.parse_args()
        self.args = argp
        self.epochs = argp.epochs
        self.world = World(argp.nr, argp.nodes, gpu, argp.gpus)
        self.dlkwargs = DlKwargs(
            batch_size=argp.bs,
            num_workers=argp.num_workers,
        )
        self.optkwargs = OptKwargs(
            lr=argp.lr,
            weight_decay=argp.weight_decay,
            eps=argp.eps,
            betas=argp.betas,
        )
        self.find_unused_parameters = argp.find_unused_parameters
        self.args.device = self.world.device
        self.dl = {}
        self.amp = QuikAmp(argp.mixed_precision)

    def run_prep(self):
        if self.world.device.type == "cuda":
            torch.cuda.empty_cache()
        if self.world.gpu_id is not None:
            torch.cuda.set_device(self.world.device)
            ddp.setup(self.world.gpu_id, self.world)

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
                **asdict(self.optkwargs),
                **kwargs,
            )
        else:
            self.optimizer = optimizer_fcn(
                self.model.parameters(),
                **asdict(self.optkwargs),
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
            tensorDataset, self.world, self.dlkwargs, self.epochs
        )

    def backward(self, loss: torch.Tensor, clip: Optional[bool] = True):
        if hasattr(self.amp, "scaler"):
            self.amp.backward(self, loss, clip)
        else:
            loss.backward()
            self.optimizer.step()


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

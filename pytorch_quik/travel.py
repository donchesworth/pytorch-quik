import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils import data
import torch.utils.data.distributed as ddist
import pytorch_quik as pq
from collections import namedtuple
from contextlib import nullcontext
from torch.cuda.amp import autocast


Gpus = namedtuple(
    "Gpus",
    [
        "device",
        "gpu",
        "node_rank",
        "rank_id",
        "total_gpus",
        "world_size",
        "mixed_precision",
        "precision_type",
    ],
)
DlKwargs = namedtuple(
    "DlKwargs", "batch_size, shuffle, pin_memory, num_workers"
)
OptKwargs = namedtuple("OptKwargs", "lr, weight_decay, eps, betas")


class QuikTraveler:
    """A class for traversing a model either in training, validation, or
    testing. Is always a singular run - but can be part of a multi-GPU run.
    """

    metrics = pq.metrics.LossMetrics(0.99)

    def __init__(self, args, gpu=None):
        # self.partition = partition
        self.args = args
        self.epochs = args.epochs
        if gpu is not None:
            device = torch.device("cuda", gpu)
            rank = args.nr * args.gpus + gpu
            world_size = args.gpus * args.nodes
            self.is_ddp = True
        else:
            device = torch.device("cpu")
            rank = None
            self.is_ddp = False
        if args.mixed_precision:
            pt = autocast()
        else:
            pt = nullcontext()
        self.gpus = Gpus(
            device=device,
            gpu=gpu,
            node_rank=args.nr,
            rank_id=rank,
            total_gpus=args.gpus,
            world_size=world_size,
            mixed_precision=args.mixed_precision,
            precision_type=pt,
        )
        self.is_logger = not self.is_ddp
        if gpu == 0:
            self.is_logger = True
        self.dlkwargs = DlKwargs(
            batch_size=args.bs,
            shuffle=False,
            pin_memory=True,
            num_workers=args.num_workers,
        )
        self.optkwargs = OptKwargs(
            lr=args.lr,
            weight_decay=args.weight_decay,
            eps=args.eps,
            betas=args.betas,
        )
        self.find_unused_parameters = args.find_unused_parameters
        self.dl = {}

    def run_prep(self, args):
        if self.gpus.device.type == "cuda":
            torch.cuda.empty_cache()
        if self.gpus.gpu is not None:
            torch.cuda.set_device(self.gpus.device)
            pq.ddp.setup(self.gpus.gpu, self.gpus)

    def set_criterion(self, criterion_fcn, kwargs={}):
        if callable(criterion_fcn):
            self.criterion = criterion_fcn(**kwargs)
            self.criterion.to(self.gpus.device)

    def set_optimizer(self, optimizer_fcn, kwargs):
        if self.is_ddp:
            self.optimizer = optimizer_fcn(
                self.model.module.parameters(),
                **self.optkwargs._asdict(),
                **kwargs,
            )
        else:
            self.optimizer = optimizer_fcn(
                self.model.parameters(),
                **self.optkwargs._asdict(),
                **kwargs,
            )

    def set_scheduler(self, scheduler_fcn, kwargs):
        self.scheduler = scheduler_fcn(
            self.optimizer,
            **kwargs,
        )

    def add_state_dict(self, args, epoch):
        sd_id = pq.utils.id_str("state_dict", args, epoch)
        # ideally we'd save as *module*, but not working
        # if self.is_ddp:
        #     self.model.load_state_dict(
        #         torch.load(sd_id, map_location=self.gpus.device)
        #     )
        # else:
        state_dict = torch.load(sd_id, map_location=self.gpus.device)
        ddp_state_dict = {
            k.replace("module.", ""): v for k, v in state_dict.items()
        }
        self.model.load_state_dict(ddp_state_dict)

    def add_model(self, model, args=None, epoch=None):
        self.model = model
        if args is not None:
            self.add_state_dict(args, epoch)
        self.model.to(self.gpus.device)
        if args is None and self.is_ddp:
            self.model = DDP(
                self.model,
                device_ids=[self.gpus.device],
                output_device=self.gpus.device,
                find_unused_parameters=self.find_unused_parameters,
            )

    def add_data(self, tensorDataset):
        self.data = self.QuikData(
            tensorDataset, self.gpus, self.is_ddp, self.dlkwargs, self.epochs
        )

    class QuikData:
        def __init__(self, tensorDataset, gpus, is_ddp, dlkwargs, epochs):
            self.dataset = tensorDataset
            self.labels = self.dataset.tensors[2].cpu().numpy()
            self.is_ddp = is_ddp
            self.dlkwargs = dlkwargs
            self.add_data_loader(gpus)
            self.steps = len(self.data_loader)
            self.total_steps = self.steps * epochs

        def add_sampler(self, gpus, sampler_fcn=None, kwargs={}):
            if self.is_ddp:
                self.sampler = ddist.DistributedSampler(
                    self.dataset,
                    num_replicas=gpus.world_size,
                    rank=gpus.rank_id,
                )
            elif callable(sampler_fcn):
                self.sampler = sampler_fcn(**kwargs)
            else:
                self.sampler = sampler_fcn

        def add_data_loader(self, gpus):
            if not hasattr(self, "sampler"):
                self.add_sampler(gpus)
            self.data_loader = data.DataLoader(
                dataset=self.dataset,
                sampler=self.sampler,
                **self.dlkwargs._asdict(),
            )

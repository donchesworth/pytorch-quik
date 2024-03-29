from pytorch_quik.utils import gpus
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from typing import Callable, NamedTuple, Union
from argparse import Namespace
import os
from tqdm import tqdm
import socket
from contextlib import closing
from pathlib import Path
from typing import Optional


def find_free_port() -> int:
    """Provide a free port for DDP setup.

    Returns:
        int: A free port number.
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def tq_bar(
    esteps: int,
    epoch: Optional[int] = 0,
    total_epochs: Optional[int] = 0,
    train: Optional[bool] = True,
) -> tqdm:
    """Create a progress bar (tqdm) to keep track of a train, valid, or
    test pass

    Args:
        esteps (int): The number of steps in the epoch for the bar.
        epoch (int, optional): Epoch number for description. Defaults to 0.
        total_epochs (int, optional): Total Epochs for description. Defaults
        to 0.
        train (bool, optional): Whether to use the epoch description. Defaults
        to True.

    Returns:
        tqdm: a tqdm progress bar
    """
    pbar = tqdm(total=esteps)
    if train:
        pbar.set_description(f"epoch: {epoch +1}/{total_epochs}")
    else:
        pbar.set_description("testing progress")
    return pbar


def traverse(train_fn: Callable, args: Namespace):
    """Determine whether to train using DistributedDataParallel
    or a single training instance

    Args:
        train_fn (Callable): The train function for this particular
        Neural Network
        args (Namespace): The argparse Namespace for this script
    """
    if bool(gpus()):
        ddp_traverse(train_fn, args)
    else:
        train_fn(gpu=None, args=args)


def ddp_traverse(train_fn: Callable, args: Namespace):
    """Using multiprocessing, initiate a DistributedDataParallel
    training process

    Args:
        train_fn (Callable): The train function for this particular
        Neural Network
        args (Namespace): The argparse Namespace for this script
    """
    os.environ["MKL_THREADING_LAYER"] = "GNU"
    mp.spawn(train_fn, nprocs=args.gpus, args=(args,))


def setup(gpu: str, args: Union[Namespace, NamedTuple]):
    """Setup of the distrubtion settings.

    Args:
        gpu (str): The current gpu
        args (Union[Namespace, NamedTuple]): The gpu parameters
    """
    master_port = find_free_port()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["NCCL_P2P_LEVEL"] = "0"

    if args.use_init_group:
        init_method = Path.cwd().joinpath("init")
        init_method.mkdir(parents=True, exist_ok=True)
    else:
        init_method = "env://"
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_method}/{master_port}",
        world_size=args.world_size,
        rank=args.rank_id,
    )
    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42)


def cleanup():
    """Deletion of the DistributedDataParallel process group"""
    dist.destroy_process_group()

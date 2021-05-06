import dask_quik as dq
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from typing import Callable, NamedTuple, Union
from argparse import Namespace
import os
from tqdm import tqdm


def tq_bar(esteps, epoch=0, total_epochs=0, train=True):
    pbar = tqdm(total=esteps)
    if train:
        pbar.set_description(f'epoch: {epoch +1}/{total_epochs}')
    else:
        pbar.set_description('testing progress')
    return pbar


def traverse(train_fn: Callable, args: Namespace):
    """Determine whether to train using DistributedDataParallel
    or a single training instance

    Args:
        train_fn (Callable): The train function for this particular
        Neural Network
        args (Namespace): The argparse Namespace for this script
    """
    if bool(dq.utils.gpus()):
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
    mp.spawn(train_fn, nprocs=args.gpus, args=(args,))


def setup(gpu: str, args: Union[Namespace, NamedTuple]):
    """Setup of the distrubtion settings.

    Args:
        gpu (str): The current gpu
        args (Union[Namespace, NamedTuple]): The gpu parameters
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["NCCL_P2P_LEVEL"] = "0"

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=args.world_size,
        rank=args.rank_id,
    )

    # Explicitly setting seed to make sure that models created in two processes
    # start from same random weights and biases.
    torch.manual_seed(42)


def cleanup():
    """Deletion of the DistributedDataParallel process group"""
    dist.destroy_process_group()

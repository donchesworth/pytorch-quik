from argparse import ArgumentParser, Namespace
import dask_quik as dq
from os import popen
from typing import Dict, Any


def update_args(args: Namespace, updates: Dict[str, Any]):
    dargs = vars(args)
    for key, new_value in updates.items():
        dargs[key] = new_value
    return args


def beta_type(strings: str) -> tuple:
    """An argparse type function that takes a string and returns
    a two part tuple for a low and high beta for an optimizer

    Args:
        strings (str): a comma separated string of numbers

    Returns:
        tuple: a tuple of floats
    """
    mapped_beta = map(float, strings.split(","))
    return tuple(mapped_beta)


def add_ddp_args(parser: ArgumentParser, kwargs={}) -> ArgumentParser:
    """Add ddp args will add arguments that are common to PyTorch
    DistributedDataParallel, and necessary for a pytorch-quik trek. These can
    be set by command line, or can be defaulted in a script, or the defaults
    here can be used.

    Args:
        parser (ArgumentParser): This is the parser from the PyTorch project.
        kwargs (dict, optional): Sometimes the individual project will want
        to have default ddp args. For instance, use_init_group could be
        set at the command line, if not by the PyTorch project, or if not it
        will be set here. If they are all set here, then kwargs defaults to {}.

    Returns:
        ArgumentParser: The same ArgumentParser but with loaded learning args.
    """

    parser.add_argument(
        "--use_init_group",
        dest="use_ray_tune",
        action="store_true",
    )
    parser.add_argument(
        "-nr",
        "--nr",
        default=0,
        type=int,
        help="ranking within the nodes",
    )
    parser.add_argument(
        "-n",
        "--nodes",
        default=1,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "-g",
        "--gpus",
        default=dq.utils.gpus(),
        type=int,
        help="number of gpus per node",
    )
    parser.add_argument(
        "-nw",
        "--num_workers",
        # default=torch.get_num_threads(),
        default=kwargs.get("num_workers", 0),
        type=int,
        help="number of workers",
    )
    return parser


def add_learn_args(parser: ArgumentParser, kwargs={}) -> ArgumentParser:
    """Add learn args will add arguments that are common to PyTorch, and
    necessary for a pytorch-quik traveler. These can be set by command
    line, or can be defaulted in a script, or the defaults here can be
    used.

    Args:
        parser (ArgumentParser): This is the parser from the PyTorch project.
        kwargs (dict, optional): Sometimes the individual project will want
        to have default learning args. For instance, learning rate could be
        set at the command line, if not by the PyTorch project, or if not it
        will be set here. If they are all set here, then kwargs defaults to {}.

    Returns:
        ArgumentParser: The same ArgumentParser but with loaded learning args.
    """
    parser.add_argument(
        "-e",
        "--epochs",
        default=kwargs.get("epochs", 5),
        type=int,
        metavar="N",
        help="number of total epochs to run (2, 3, 5)",
    )
    parser.add_argument(
        "-amp",
        "--mixed_precision",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "-bs",
        "--bs",
        default=kwargs.get("bs", 50000),
        type=int,
        help="batch size (16, 32, 150000)",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        dest="lr",
        default=kwargs.get("learning_rate", 1e-5),
        type=float,
        help="learning rate for an optimizer (1e-5, 2e-6)",
    )
    parser.add_argument(
        "-wd",
        "--weight_decay",
        default=kwargs.get("weight_decay", 0),
        type=float,
        help="weight decay for an optimizer",
    )
    parser.add_argument(
        "--betas",
        type=beta_type,
        default=kwargs.get("betas", (0.9, 0.99)),
        help="beta values for an optimizer. Enter them as \
            one string, such as '0.90, 0.99'.",
    )
    parser.add_argument(
        "-eps",
        default=kwargs.get("eps", 1e-08),
        type=float,
        help="weight decay for an optimizer",
    )
    parser.add_argument(
        "--find_unused_parameters",
        dest="find_unused_parameters",
        action="store_true",
    )
    return parser


def add_ray_tune_args(parser: ArgumentParser, kwargs={}) -> ArgumentParser:
    """Add ray tune args will add arguments that are common to ray tune,
    and necessary for a pytorch-quik traveler. These can be set by command
    line, or can be defaulted in a script, or the defaults here can be
    used.

    Args:
        parser (ArgumentParser): This is the parser from the PyTorch project.
        kwargs (dict, optional): Sometimes the individual project will want
        to have default ray tune args. For instance, num samples could be
        set at the command line, if not by the PyTorch project, or if not it
        will be set here. If they are all set here, then kwargs defaults to {}.

    Returns:
        ArgumentParser: The same ArgumentParser but with loaded ray tune args.
    """
    parser.add_argument(
        "--use_ray_tune",
        dest="use_ray_tune",
        action="store_true",
    )
    parser.add_argument(
        "--num_samples",
        default=kwargs.get("num_samples", 20),
        type=int,
        help="The number of ray tune hyperparameter permutations to train",
    )
    return parser


def add_mlflow_args(parser: ArgumentParser, kwargs={}) -> ArgumentParser:
    """Add mlflow args will add arguments that are common to mlflow, and
    necessary for a pytorch-quik traveler. These can be set by command
    line, or can be defaulted in a script, or the defaults here can be
    used.

    Args:
        parser (ArgumentParser): This is the parser from the PyTorch project.
        kwargs (dict, optional): Sometimes the individual project will want
        to have default learning args. For instance, tracking uri could be
        set at the command line, if not by the PyTorch project, or if not it
        will be set here. If they are all set here, then kwargs defaults to {}.

    Returns:
        ArgumentParser: The same ArgumentParser but with loaded mlflow args.
    """
    parser.add_argument(
        "--use_mlflow",
        dest="use_mlflow",
        action="store_true",
    )
    parser.add_argument(
        "-exp",
        "--experiment",
        default=kwargs.get("experiment", "Default"),
        type=str,
        help="The name of the mlflow experiment (default is Default)",
    )
    parser.add_argument(
        "-u",
        "--user",
        default=kwargs.get("user", popen("whoami").read().strip()),
        type=str,
        help="The name of the mlflow experiment (default is Default)",
    )
    parser.add_argument(
        "--tracking_uri",
        default=kwargs.get("tracking_uri", "http://localhost:5000"),
        type=str,
        help="The tracking uri for mlflow (default is localhost, port 5000)",
    )
    parser.add_argument(
        "--endpoint_url",
        default=kwargs.get("endpoint_url", "http://s3.amazonaws.com"),
        help="The S3 endpoint where MLFlow will send your model artifacts",
    )
    return parser

import time
from pathlib import Path
from typing import Optional, Union
from argparse import Namespace
import torch
from argparse import ArgumentParser
import dask_quik as dq


def sec_str(st: float) -> str:
    """Simple util that tells the amount of time for a codeset (in seconds)

    Args:
        st (time): The start time of the codeset

    Returns:
        str: time in seconds
    """
    return str(round(time.time() - st, 2)) + " seconds"


def row_str(dflen: int) -> str:
    """String wrapper for the million of rows in a dataframe

    Args:
        dflen (int): the length of a dataframe

    Returns:
        str: rows in millions
    """
    return str(round(dflen / 1000000, 1)) + "M rows"


def id_str(
    ftype: str,
    args: Namespace,
    epoch: Optional[str] = None,
    gpu: Optional[str] = None,
    suffix: Optional[str] = ".pt",
) -> str:
    """Method to determine an appropriate filename for tensors, models, and
    state_dicts

    Args:
        ftype (str): file type (state_dict, train, valid, test, preds, model)
        args (Namespace): the current set of arguments. requires date,
            optional bert_type, source
        epoch (Optional[str], optional): the current epoch to be saving.
            Defaults to None.
        gpu (Optional[str], optional): the current gpu (for preds). Defaults
            to None.
        suffix (Optional[str], optional): file extension. Defaults to ".pt".

    Returns:
        str: full filename for output object
    """
    lbls = "".join(map(str, getattr(args, "labels", "")))
    if ftype == "state_dict":
        epoch = "e" + str(epoch)
    elif ftype == "preds":
        suffix = ".csv"
    elif ftype != "model":
        ftype = ftype + "_tensor"
    id_list = [ftype, lbls, epoch, str(args.data_date), gpu]
    id_str = "_".join(filter(None, id_list))
    path_list = [
        "data",
        getattr(args, "bert_type", None),
        getattr(args, "source", None),
        id_str,
    ]
    path_list = filter(None, path_list)
    filename = Path.cwd().joinpath(*path_list).with_suffix(suffix)
    filename.parent.mkdir(parents=True, exist_ok=True)
    return filename


def tens_load(
    ttype: str, args: Namespace, loc_only: Optional[bool] = False
) -> Union[str, torch.Tensor]:
    """Load a PyTorch tensor

    Args:
        ttype (str): The type of tensor (train, valid, test)
        args (Namespace): The argparse namespace
        loc_only (bool, optional): If the tensor isn't wanted, but
        only the location (filename). Defaults to False.

    Returns:
        Union[str, torch.Tensor]: [description]
    """
    print("loading " + ttype + " " + str(args.device))
    file_id = id_str(ttype, args)
    if loc_only:
        return file_id
    else:
        return torch.load(file_id)


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


def add_learn_args(parser: ArgumentParser, kwargs={}) -> ArgumentParser:
    """Add learn args will add arguments that are common to PyTorch, and
    necessary for a pytorch-quik traveller. These can be set by command
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
        "-amp",
        "--mixed_precision",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "-nr",
        "--nr",
        default=0,
        type=int,
        help="ranking within the nodes",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=kwargs.get("epochs", 5),
        type=int,
        metavar="N",
        help="number of total epochs to run (2, 3, 5)",
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
        "-nw",
        "--num_workers",
        # default=torch.get_num_threads(),
        default=kwargs.get("num_workers", 0),
        type=int,
        help="number of workers",
    )
    parser.add_argument(
        "--find_unused_parameters",
        dest="find_unused_parameters",
        action="store_true",
    )
    return parser

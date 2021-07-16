import torch
from argparse import Namespace
from typing import Optional, Union
from pathlib import Path
from datetime import date
import json
import numpy as np


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
    elif ftype == "confusion":
        suffix = ".png"
    elif ftype == "test_array":
        suffix = ".npy"
    elif ftype != "model":
        ftype = ftype + "_tensor"
    data_date = getattr(args, "data_date", date.today().strftime("%Y%m%d"))
    id_list = [ftype, lbls, epoch, str(data_date), gpu]
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


def load_torch_object(
    torch_type: str,
    args: Namespace,
    epoch: Optional[int] = None,
    location: Optional[bool] = False,
) -> Union[str, torch.Tensor]:
    """Load a PyTorch object (tensor, model, state_dict)

    Args:
        ttype (str): The type of torch object (train, valid, test, state_dict)
        args (Namespace): The argparse namespace
        epoch (int, optional): Current epoch (for state_dict)
        location (bool, optional): If the object isn't wanted, but
        only the location (filename). Defaults to False.

    Returns:
        Union[str, torch.Tensor]: [description]
    """

    print("loading " + torch_type + " " + str(getattr(args, "device", "cpu")))
    filename = id_str(torch_type, args, epoch)
    if location:
        return filename
    pt = torch.load(filename)
    if torch_type == "state_dict":
        # DDP will leave module artifacts to be removed
        pt = {k.replace("module.", ""): v for k, v in pt.items()}
    return pt


def load_tensor(
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


# def load_state_dict():
#     sd_id = pq.utils.id_str("state_dict", args, epoch)
#     state_dict = torch.load(sd_id, map_location=self.gpus.device)
#     # DDP will leave module artifacts to be removed
#     ddp_state_dict = {
#         k.replace("module.", ""): v for k, v in state_dict.items()
#     }
#     self.model.load_state_dict(ddp_state_dict)


def save_state_dict(model, args: Namespace, epoch: int):
    sd_id = id_str("state_dict", args, epoch)
    print("saving epoch " + str(epoch) + " state")
    if hasattr(model, "module"):
        sd = model.module.state_dict()
    else:
        sd = model.state_dict()
    torch.save(sd, sd_id)
    return sd_id


def save_test_array(Xte: np.array, yte: np.array, args: Namespace):
    filename = id_str("test_array", args)
    with open(filename, 'wb') as f:
        np.save(f, Xte)
        np.save(f, yte)


def load_test_array(args):
    filename = id_str("test_array", args)
    with open(filename, 'rb') as f:
        Xte = np.load(f, allow_pickle=True)
        yte = np.load(f, allow_pickle=True)
    return Xte, yte


def json_write(serve_path, filename, data):
    file = Path.joinpath(serve_path, filename)
    with open(file, 'w') as outfile:
        json.dump(data, outfile)

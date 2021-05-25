import torch
from argparse import Namespace
from typing import Optional, Union
from pathlib import Path


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


def state_dict_save(model, args: Namespace, epoch: int):
    sd_id = id_str("state_dict", args, epoch)
    print("saving epoch " + str(epoch) + " state")
    if hasattr(model, "module"):
        sd = model.module.state_dict()
    else:
        sd = model.state_dict()
    torch.save(sd, sd_id)

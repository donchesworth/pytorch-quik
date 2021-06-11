import torch
import torch.utils.data as td
from typing import Optional, Dict, Union
from transformers import BatchEncoding
from argparse import Namespace
import numpy as np
import pandas as pd
from pytorch_quik import io

Tensor_Target = Union[str, np.ndarray]
Tensor_Data = Union[pd.DataFrame, torch.Tensor, BatchEncoding]


def make_TensorDataset(
    tens: Tensor_Data,
    labels: Tensor_Target,
    ttype: Optional[str] = "train",
    data_types: Optional[Dict[int, type]] = None,
    args: Optional[Namespace] = None,
) -> td.TensorDataset:
    """Will turn a set of data into tensors in a torch TensorDataset for use
    in a Neural Network. Also provides the option of saving the TensorDataset

    Args:
        tens (Union[torch.Tensor, BatchEncoding]): Either a torch.Tensor,
            or if from transformers, a BatchEncoding.
        labels (Union[str, np.ndarray]): Can either be a string of the label
            name found in tens, or the actual labels as an np.ndarray
        args (Namespace, optional): The argparse arguments for the job.
            Defaults to None.
        ttype (str, optional): The type of dataset (train, valid, test).
            Defaults to "train".
        data_types (Dict[int, type], optional): If the tensor data types
            need to be changed for space considerations. Defaults to None.

    Returns:
        td.TensorDataset: The final TensorDataset
    """
    if data_types is not None:
        for i, col in enumerate(tens):
            tens[col] = tens[col].astype(data_types[i])
    if isinstance(labels, str):
        ttar = tens.pop(labels)
    if isinstance(tens, BatchEncoding):
        tds = td.TensorDataset(
            tens["input_ids"], tens["attention_mask"], torch.LongTensor(labels)
        )
    else:
        tens = torch.tensor(tens.values)
        tens = tens.transpose(0, 1)
        ttar = torch.tensor(ttar.values)
        tds = td.TensorDataset(*tens, ttar)
    if args is not None:
        tds_id = io.id_str(ttype, args)
        torch.save(tds, tds_id)
    return tds


def transform_TensorDataset(
    tds: td.TensorDataset,
    pop: Optional[str] = None,
    torch_types: Optional[Dict[int, torch.dtype]] = None,
) -> td.TensorDataset:
    """Transforms a torch.utils.data.TensorDataset by either popping out
    a tensor, changing the data types, or both.

    Args:
        tds (td.TensorDataset): The original TensorDataset
        pop (str, optional): The ordinal of the tensor to pop. Defaults to
            None.
        torch_types (Dict[int, torch.dtype], optional): The ordinal and
            new data type of each tensor. Defaults to None.

    Returns:
        td.TensorDataset: The final transformed TensorDataset
    """
    tl = list(tds.tensors)
    if pop is not None:
        del tl[pop]
    if torch_types is not None:
        tl = [tens.type(torch_types[i]) for i, tens in enumerate(tl)]
    return td.TensorDataset(*tl)

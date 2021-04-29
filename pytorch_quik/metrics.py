from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.colors as mclr

# from collections import OrderedDict
from typing import Union, Optional, List, Dict, Tuple, OrderedDict
import time
from datetime import timedelta
import numpy as np
import pandas as pd
from torch import Tensor
import torch

try:
    import cudf
except ImportError:
    import dask_quik.dummy as cudf

ArrType = Union[
    np.ndarray, pd.Series, pd.DataFrame, cudf.DataFrame, torch.Tensor
]
LABELS = ["Actual", "Predicted"]


def direct_dict(classes: Tuple[str]) -> OrderedDict[int, str]:
    """Create an ordered dict of classes with indices as keys

    Args:
        classes (Tuple[str]): A list of classes

    Returns:
        OrderedDict[int, str]: A final ordered dict
    """
    class_keys = range(len(classes))
    return OrderedDict(zip(class_keys, classes))


def inverse_dict(
    dict_direct: Union[Dict, OrderedDict]
) -> OrderedDict[str, int]:
    """invert a dictionary

    Args:
        dict_direct (Union[Dict, OrderedDict]): the original
            dictionary to be inverted
    Returns:
        OrderedDict[str, int]: The inverted ordered dictionary
    """
    dict_inverse = {v: k for k, v in dict_direct.items()}
    return OrderedDict(dict_inverse)


def choose_a_class(preds_array: ArrType) -> ArrType:
    """Given an np.array with probabilities for multiple classes,
    provide the argmax class back as a np.array

    Args:
        preds_array (ArrType): 2d array with class probabilities

    Returns:
        np.array: 1d np.array with selected class
    """
    return np.argmax(preds_array, axis=1).flatten()


def numpize_array(arr: ArrType) -> np.array:
    """Given some type of array, try to convert it to a
    numpy array, and of one dimension

    Args:
        arr (ArrType): 1d or 2d array of some type

    Returns:
        np.array: 1d numpy array
    """
    if isinstance(arr, cudf.DataFrame):
        arr = arr.as_matrix()
    elif isinstance(arr, (pd.DataFrame, pd.Series)):
        arr = arr.to_numpy()
    elif isinstance(arr, torch.Tensor):
        arr = arr.numpy()
    if len(arr.shape) == 2:
        if arr.shape[1] == 1:
            arr = arr.squeeze()
        else:
            arr = choose_a_class(arr)
    return arr


def accuracy_per_class(preds_array: ArrType, true_array: ArrType, label_dict):
    """Accuracy per class function taken from Ali Anastassiou's
    BERT course
    https://www.coursera.org/projects/sentiment-analysis-bert

    Args:
        preds_array (ArrType): [description]
        true_array (ArrType): [description]
        label_dict ([type]): [description]
    """
    # preds_array = choose_a_class(preds_array)
    [parr, tarr] = [numpize_array(x) for x in [preds_array, true_array]]
    for nclass in np.unique(tarr):
        y_preds = parr[tarr == nclass]
        y_true = tarr[tarr == nclass]
        print(f"Class: {label_dict[nclass]}")
        print(f"Accuracy: {len(y_preds[y_preds==nclass])}/{len(y_true)}\n")


def f1_score_func(preds_array: ArrType, true_array: ArrType) -> np.float64:
    """F1 score function taken from Ali Anastassiou's BERT course
    https://www.coursera.org/projects/sentiment-analysis-bert


    Args:
        preds_array (ArrType): Prediction array
        true_array (ArrType): True labels array

    Returns:
        np.float64: F1 score
    """
    [parr, tarr] = [numpize_array(x) for x in [preds_array, true_array]]
    f1 = f1_score(tarr, parr, average="weighted")
    print(f"weighted f1 score {f1:.3f}")


def matplotlib_gist_heat():
    cmap = plt.get_cmap("gist_heat")
    new_cmap = mclr.LinearSegmentedColormap.from_list(
        "trunc({n},{a:.2f},{b:.2f})".format(n=cmap.name, a=0.01, b=0.9),
        cmap(np.linspace(0.01, 0.9, 100)),
    )
    return new_cmap


def matplotlib_confusion_matrix(
    conf_matrix: np.ndarray, dir_classes: OrderedDict[int, str]
):
    """Turn a confustion matrix into a matplotlib plot.

    Args:
        conf_matrix (np.ndarray): a 2d array of the confusion matrix
        dirclasses (OrderedDict[int, str]): A list of chart labels for each
            classification type
    """
    heat_cmap = matplotlib_gist_heat()
    ticks = list(dir_classes.keys())
    classes = list(dir_classes.values())
    lw = len(classes)
    fig, ax = plt.subplots(figsize=(lw, lw))
    plt.imshow(conf_matrix, cmap=heat_cmap)
    plt.xticks(ticks=ticks, labels=classes)
    plt.yticks(ticks=ticks, labels=classes, rotation=90)
    _ = [
        ax.text(j, i, conf_matrix[i, j], ha="center", va="center", color="w")
        for i in range(lw)
        for j in range(lw)
    ]
    plt.ylabel(LABELS[0])
    plt.xlabel(LABELS[1])
    plt.show()
    return plt


def show_confusion_matrix(
    preds_array: ArrType,
    true_array: ArrType,
    dir_classes: OrderedDict[int, str],
    plot: Optional[bool] = False,
):
    """Takes either a 2d classification array or a 1d array and true
    values, and creates a confusion matrix, either with seaborn or pandas.
    Taken from https://stackoverflow.com/questions/65390832/
    set-meta-name-for-rows-and-columns-in-pandas-dataframe

    Args:
        preds_array (ArrType): A 2d np.array to be argmaxed, a
            pd.Series, or a 1d np.array
        true_array (ArrType): A pd.Series or 1d np.array
        dir_classes: OrderedDict[int, str]: A dict of index keys and
            labels for each classification type
        plot (bool, optional): If this is in a notebook, then you may
        wish to create a seaborn plot. Defaults to False.
    """
    [parr, tarr] = [numpize_array(x) for x in [preds_array, true_array]]
    conf_mat = confusion_matrix(tarr, parr)
    if plot:
        cm = matplotlib_confusion_matrix(conf_mat, dir_classes)
    else:
        classes = dir_classes.values()
        labels = [
            pd.MultiIndex.from_product([[lbl], classes]) for lbl in LABELS
        ]
        cm = pd.DataFrame(conf_mat, *labels)
    return cm


class SmoothenLoss:
    """Taken from fast.ai, Create a smooth moving average with a beta
    value (https://github.com/fastai/fastai1/blob/master/fastai/callback.py)"""

    def __init__(self, beta: float):
        """Create the smoothen value instance for loss metrics

        Args:
            beta (float): The smoothing beta
        """
        self.beta = beta
        self.n = 0
        self.mov_avg = 0

    def add_value(self, val: float) -> None:
        """Add value to calculate updated smoothed value.

        Args:
            val (float): The value to be added
        """

        self.n += 1
        self.mov_avg = self.beta * self.mov_avg + (1 - self.beta) * val
        self.smooth = self.mov_avg / (1 - self.beta ** self.n)


class LossMetrics:
    """Don's method to store dict without callbacks, adapted from
    https://github.com/fastai/fastai/blob/master/fastai/metrics.py"""

    def __init__(self, beta: Optional[float] = 0.99):
        """Create the loss metrics structure

        Args:
            beta (float): A beta for smoothing the average. Default is 0.99
        """
        self.beta = beta
        self.smoothener = SmoothenLoss(self.beta)
        self.state_dict = {"epoch": 0, "train_loss": 0.0, "valid_loss": 0.0}
        self.results = pd.DataFrame()
        self.start_time = time.time()

    def add_loss(self, loss: Tensor):
        """Handle gradient calculation on loss.

        Args:
            loss (Tensor): The loss tensor from criterion
        """
        loss = loss.float().detach().cpu().item()
        self.smoothener.add_value(loss)
        self.state_dict["train_loss"] = self.smoothener.smooth

    def add_vloss(self, vlosses: List[Tensor], nums: List[int]):
        """Handle validation loss calculation

        Args:
            vlosses (List[Tensor]): A set of validation tensors
            nums (List[int]): The length of the groups
        """
        nums = np.array(nums, dtype=np.float32)
        fl = (
            torch.stack(vlosses).data.cpu().numpy() * nums
        ).sum() / nums.sum()
        self.state_dict["valid_loss"] = fl

    def calc_time(self):
        """Add time to the metrics DataFrame"""
        elapsed = time.time() - self.start_time
        self.duration = (
            str(timedelta(seconds=elapsed)).split(".")[0].partition(":")[2]
        )
        self.state_dict["time"] = self.duration

    def write(self):
        """write out the metrics dataframe when completed"""
        self.calc_time()
        self.results = pd.concat(
            [
                self.results,
                pd.DataFrame(
                    self.state_dict, index=[self.state_dict["epoch"]]
                ),
            ]
        )
        self.state_dict["epoch"] += 1

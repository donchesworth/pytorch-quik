from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
from typing import Union, Optional, List, Dict, Tuple
import numpy as np
import pandas as pd

try:
    import cudf
except ImportError:
    import dask_quik.dummy as cudf

ArrType = Union[np.ndarray, pd.Series, pd.DataFrame, cudf.DataFrame]
LABELS = ["Actual", "Predicted"]


def direct_dict(classes: Tuple[str]) -> OrderedDict[int, str]:
    """Create an ordered dict of classes with indices as keys

    Args:
        classes (Tuple[str]): A list of classes

    Returns:
        OrderedDict[int, str]: A final ordered dict
    """
    class_keys = range(len(classes))
    return OrderedDict(zip(class_keys), classes)


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


def choose_a_class(preds_array: ArrType) -> np.array:
    """Given an array with probabilities for multiple classes,
    convert to a np.array and provide the argmax class
    back as a np.array

    Args:
        preds_array (ArrType): 2d array with class probabilities

    Returns:
        np.array: 1d np.array with selected class
    """
    if isinstance(preds_array, cudf.DataFrame):
        preds_array = preds_array.as_matrix()
    elif isinstance(preds_array, (pd.DataFrame, pd.Series)):
        preds_array = preds_array.to_numpy()
    return np.argmax(preds_array, axis=1).flatten()


def accuracy_per_class(
    preds_array: ArrType, actual_array: ArrType, label_dict
):
    """Accuracy per class function taken from Ali Anastassiou's
    BERT course
    https://www.coursera.org/projects/sentiment-analysis-bert

    Args:
        preds_array (ArrType): [description]
        actual_array (ArrType): [description]
        label_dict ([type]): [description]
    """
    label_dict_inverse = {v: k for k, v in label_dict.items()}
    # preds_array = choose_a_class(preds_array)
    for nclass in np.unique(actual_array):
        y_preds = preds_array[actual_array == nclass]
        y_true = actual_array[actual_array == nclass]
        print(f"Class: {label_dict_inverse[nclass]}")
        print(f"Accuracy: {len(y_preds[y_preds==nclass])}/{len(y_true)}\n")


def f1_score_func(
    prediction_array: ArrType, actual_array: ArrType
) -> np.float64:
    """F1 score function taken from Ali Anastassiou's BERT course
    https://www.coursera.org/projects/sentiment-analysis-bert


    Args:
        preds (ArrType): Prediction array
        labels (ArrType): True labels array

    Returns:
        np.float64: F1 score
    """
    prediction_array = choose_a_class(prediction_array)
    return f1_score(actual_array, prediction_array, average="weighted")


def seaborn_confusion_matrix(conf_matrix: np.ndarray, classes: List[str]):
    """Turn a confustion matrix into a seaborn plot.

    Args:
        conf_matrix (np.ndarray): a 2d array of the confusion matrix
        classes (List[str]): A list of chart labels for each
            classification type
    """
    fig, ax = plt.subplots(figsize=(len(classes), len(classes)))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.ylabel(LABELS[0])
    plt.xlabel(LABELS[1])
    plt.show()


def show_confusion_matrix(
    predicted_array: ArrType,
    actual_array: ArrType,
    classes: List[str],
    plot: Optional[bool] = False,
):
    """Takes either a 2d classification array or a 1d array and true
    values, and creates a confusion matrix, either with seaborn or pandas.
    Taken from https://stackoverflow.com/questions/65390832/
    set-meta-name-for-rows-and-columns-in-pandas-dataframe

    Args:
        predicted_array (ArrType): A 2d np.array to be argmaxed, a
            pd.Series, or a 1d np.array
        actual_array (ArrType): A pd.Series or 1d np.array
        classes (List[str]): A list of chart labels for each
            classification type
        plot (bool, optional): If this is in a notebook, then you may
        wish to create a seaborn plot. Defaults to False.
    """
    if len(predicted_array.shape) == 2:
        predicted_array = choose_a_class(predicted_array)
    conf_mat = confusion_matrix(actual_array, predicted_array)
    if plot:
        seaborn_confusion_matrix(conf_mat, classes)
    else:
        labels = [
            pd.MultiIndex.from_product([[lbl], classes]) for lbl in LABELS
        ]
        df = pd.DataFrame(conf_mat, *labels)
        print(df)

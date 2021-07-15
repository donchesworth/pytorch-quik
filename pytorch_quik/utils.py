import time
from typing import Union, Dict, Tuple, OrderedDict


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


def indexed_dict(classes: Tuple[str]) -> OrderedDict[int, str]:
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

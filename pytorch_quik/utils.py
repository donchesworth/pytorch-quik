import time


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

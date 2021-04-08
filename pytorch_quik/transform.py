from typing import Optional, Union, Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler


def create_dicts(
    categories: np.array, labels: np.array
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Turns two lists of ints into two dictionaries with a index key and
    a value from the list. These help with a quick relabelling.

    Args:
        categories (List[int]): The categorical variable numbers
        labels (List[int]): The label numbers

    Returns:
        Tuple[Dict[int, int], Dict[int, int]]: A category indexed dictionary
        and a category keys labels values dictionary.
    """
    catd = dict(zip(categories, range(0, len(categories))))
    lbld = dict(zip(categories, labels))
    return catd, lbld


def create_labels(
    df: pd.DataFrame,
    labels: Optional[List[int]] = [0, 0, 1, 2, 2],
    category: Optional[str] = "category",
) -> pd.DataFrame:
    """Turn categorical variables into labels to be predicted. For instance,
    if your data is 1-10, and you want to predict 5-10 is positive, and 0-4
    is negative, then you would provide the category column name, and a list
    of [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]. This will label all rows as positive or
    negative.

    Args:
        df (pd.DataFrame): Original df to be labelled.
        category (str, optional): DataFrame column name containing the
            category. Defaults to 'category'.
        labels (List[int], optional): The labels to be assigned to each
            category. It must be the same length as the number of unique
            categories. Defaults to [0, 0, 1, 2, 2].

    Returns:
        pd.DataFrame: The original df with an additional column named
        label.
    """
    cats = np.sort(df[category].unique())
    labels = np.array(labels)
    print("Categories\t\t" + str(cats) + "\nNow match labels\t" + str(labels))
    cat_dict, label_dict = create_dicts(cats, labels)
    df["cat_index"] = df[category].replace(cat_dict)
    df["label"] = df[category].replace(label_dict)
    return df


def imbalanced_split(
    df: pd.DataFrame,
    label_name: Optional[str] = "label",
    test_size: Optional[int] = 0.1,
    valid_size: Optional[Union[int, None]] = None,
    random_state: Optional[int] = 42,
) -> Tuple[np.ndarray]:
    """When categories are imbalanced, create a balanced split of train,
    validation, and test sets.

    Args:
        df (pd.DataFrame): The original dataFrame to be split
            label_name (str, optional): the name of the
            column to be balanced. Defaults to "label".
            test_size (int, optional): The size of the test set.
            Defaults to 0.1.
        valid_size (Union[int, None], optional): The size of the
            validation set. Defaults to None.
        random_state (int, optional): The random state, or seed.
            Defaults to 42.

    Returns:
        Tuple[np.ndarray]: A tuple of four to six arrays
    """
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(label_name, axis=1).values,
        df[label_name].values,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_name].values,
    )
    arrlist = [X_test, y_test]
    if valid_size is not None:
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train,
            y_train,
            test_size=valid_size,
            random_state=random_state,
            stratify=y_train,
        )
        arrlist = [X_valid, y_valid] + arrlist
    X_over, y_over = RandomOverSampler().fit_resample(X_train, y_train)
    vc = np.unique(y_over, return_counts=True)
    print(
        "These labels \t\t" + str(vc[0]) + "\nare oversampled to " + str(vc[1])
    )
    arrlist = [X_over, y_over] + arrlist
    return arrlist

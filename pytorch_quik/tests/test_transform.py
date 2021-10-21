from pytorch_quik import transform
from collections import OrderedDict
import numpy as np
import pandas as pd


def test_create_dicts(sample_data, final_data, eight_labels):
    df = sample_data
    cats = np.sort(sample_data["category"].unique())
    labels = np.array(eight_labels)
    cat_dict, label_dict = transform.create_dicts(cats, labels)
    df["cat_index"] = df["category"].replace(cat_dict)
    df["label"] = df["category"].replace(label_dict)
    cols = ["text", "category", "label"]
    assert df[cols].equals(final_data.reset_index(drop=True)[cols])


def test_create_labels(sample_data, eight_labels):
    tdf = transform.create_labels(sample_data, eight_labels)
    exp_out = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    assert np.array_equal(exp_out, tdf.label.values)


def test_imbalanced_split_test(sample_data, eight_labels):
    eight_labels[6:8] = [2, 2]
    tdf = transform.create_labels(sample_data, eight_labels)
    data = transform.imbalanced_split(tdf, test_size=0.4)
    assert isinstance(data, OrderedDict)
    assert isinstance(data["train"], transform.DataSplit)
    assert data["train"].X.shape == (6, 3)
    assert data["train"].y.shape == (6,)
    assert set(np.bincount(data["train"].y)) == {2}


def test_imbalanced_split_valid(sample_data, eight_labels):
    eight_labels[6:8] = [2, 2]
    tdf = transform.create_labels(sample_data, eight_labels)
    tdf = pd.concat([tdf]*3)
    data = transform.imbalanced_split(
        tdf, valid_size=0.2, test_size=0.2
        )
    assert isinstance(data, OrderedDict)
    assert isinstance(data["valid"], transform.DataSplit)
    assert data["valid"].X.shape == (4, 3)
    assert set(np.bincount(data["train"].y)) == {7}

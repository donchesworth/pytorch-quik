import pytest
import pytorch_quik as pq
import torch.utils.data as td


def test_make_tds(final_data):
    df = final_data.drop('text', axis=1)
    tds = pq.tensors.make_TensorDataset(df, "label")
    assert(isinstance(tds, td.TensorDataset))


def test_transform_tds(final_data, ttypes):
    df = final_data.drop('text', axis=1)
    tds = pq.tensors.make_TensorDataset(df, "label")
    tds = pq.tensors.transform_TensorDataset(tds, 1, ttypes)

import pytorch_quik as pq
import torch.utils.data as td


def test_make_tds(final_data, args):
    df = final_data.drop('text', axis=1)
    tds = pq.tensor.make_TensorDataset(df, "label")
    assert(isinstance(tds, td.TensorDataset))


def test_transform_tds(final_data, ttypes, args):
    df = final_data.drop('text', axis=1)
    tds = pq.tensor.make_TensorDataset(df, "label")
    tds = pq.tensor.transform_TensorDataset(tds, 1, ttypes)
    assert(isinstance(tds, td.TensorDataset))

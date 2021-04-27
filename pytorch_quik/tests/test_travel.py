import pytest
from pytorch_quik.travel import QuikTraveler
from torch.utils import data
import torch.utils.data as td


def test_quik_traveler(args):
    myqt = QuikTraveler(0, args)
    assert(isinstance(myqt, QuikTraveler))
    assert(isinstance(myqt.epochs, int))


def test_quik_data(sample_tensor, sample_labels, args):
    if args.gpus == 0:
        pytest.skip()
    myqt = QuikTraveler(0, args)
    mytds = td.TensorDataset(sample_tensor, sample_tensor, sample_labels)
    myqt.add_data(mytds)
    assert(isinstance(myqt.data.dataset, td.TensorDataset))
    assert(isinstance(myqt.data.data_loader, data.DataLoader))

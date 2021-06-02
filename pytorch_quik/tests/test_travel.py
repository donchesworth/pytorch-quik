import pytest
from pytorch_quik.travel import QuikTraveler
from torch.utils import data
import torch.utils.data as td


def test_quik_traveler(args, gpus):
    myqt = QuikTraveler(args, 0)
    assert isinstance(myqt, QuikTraveler)
    assert isinstance(myqt.epochs, int)


def test_quik_data(gpus, sample_tensor, sample_labels, args):
    args.gpus = gpus
    if args.gpus == 0:
        pytest.skip()
    myqt = QuikTraveler(args, 0)
    mytds = td.TensorDataset(sample_tensor, sample_tensor, sample_labels)
    myqt.add_data(mytds)
    assert(isinstance(myqt.data.dataset, td.TensorDataset))
    assert(isinstance(myqt.data.data_loader, data.DataLoader))

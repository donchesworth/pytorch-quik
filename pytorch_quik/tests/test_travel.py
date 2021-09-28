from pytorch_quik.travel import QuikTrek, QuikTraveler
from torch.utils import data
import torch.utils.data as td


def test_quik_trek(args):
    myqt = QuikTrek(args.gpu, args)
    assert isinstance(myqt, QuikTrek)
    assert isinstance(myqt.epochs, int)


# def test_quik_traveler(args):
#     myqt = QuikTrek(args.gpu, args)
#     myqtr = QuikTraveler(myqt, "pytest")
#     assert isinstance(myqtr, QuikTraveler)
#     assert isinstance(myqtr.epochs, int)


# def test_quik_data(gpus, sample_tensor, sample_labels, args):
#     args.gpus = gpus
#     if args.gpus == 0:
#         pytest.skip()
#     myqt = QuikTrek(0, args)
#     myqtr = QuikTraveler(myqt, "pytest")
#     mytds = td.TensorDataset(sample_tensor, sample_tensor, sample_labels)
#     myqtr.add_data(mytds)
#     assert(isinstance(myqtr.data.dataset, td.TensorDataset))
#     assert(isinstance(myqtr.data.data_loader, data.DataLoader))

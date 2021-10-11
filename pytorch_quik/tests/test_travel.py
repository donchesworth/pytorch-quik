# from pytorch_quik.travel import (
#     QuikTrek,
#     QuikTraveler,
#     QuikAmp,
#     DlKwargs,
#     OptKwargs,
#     World,
# )
# from pytorch_quik.ddp import cleanup
# from pytorch_quik.bert import get_pretrained_model, BERT_MODELS
# from torch.utils import data
# import torch.utils.data as td
# from torch import nn
# from transformers import (
#     AdamW,
#     get_linear_schedule_with_warmup,
#     logging as tlog,
# )


# def test_quik_model_ml(args, sample_labels):
#     myqt = QuikTrek(args.gpu, args)
#     myqtr = QuikTraveler(myqt, "pytest")
#     tlog.set_verbosity_error()
#     model = get_pretrained_model(
#         labels=sample_labels,
#         bert_type=args.bert_type
#     )
#     myqtr.add_model(model)
#     myqtr.set_criterion(nn.CrossEntropyLoss)
#     myqtr.set_optimizer(AdamW)
#     args.sched_kwargs["num_training_steps"] = 100
#     myqtr.set_scheduler(get_linear_schedule_with_warmup, args.sched_kwargs)
#     assert(isinstance(model, BERT_MODELS[args.bert_type]["model"]))
#     assert(isinstance(myqtr.criterion, nn.CrossEntropyLoss))
#     assert(isinstance(myqtr.optimizer, AdamW))
#     if args.gpu is not None:
#         cleanup()


# def test_quik_trek(args):
#     myqt = QuikTrek(args.gpu, args)
#     assert isinstance(myqt, QuikTrek)
#     assert isinstance(myqt.epochs, int)
#     assert isinstance(myqt.dlkwargs, DlKwargs)
#     assert isinstance(myqt.optkwargs, OptKwargs)
#     assert isinstance(myqt.world, World)
#     if args.gpu is not None:
#         cleanup()


# def test_quik_traveler(args):
#     myqt = QuikTrek(args.gpu, args)
#     myqtr = QuikTraveler(myqt, "pytest")
#     assert isinstance(myqtr, QuikTraveler)
#     assert isinstance(myqtr.amp, QuikAmp)
#     if args.gpu is not None:
#         cleanup()


# def test_quik_data(sample_tensor, sample_labels, args):
#     myqt = QuikTrek(args.gpu, args)
#     myqtr = QuikTraveler(myqt, "pytest")
#     mytds = td.TensorDataset(sample_tensor, sample_tensor, sample_labels)
#     myqtr.add_data(mytds)
#     assert(isinstance(myqtr.data.dataset, td.TensorDataset))
#     assert(isinstance(myqtr.data.data_loader, data.DataLoader))
#     if args.gpu is not None:
#         cleanup()


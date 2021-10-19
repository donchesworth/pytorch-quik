from pytorch_quik.travel import (
    QuikTrek,
    QuikTraveler,
    QuikAmp,
    DlKwargs,
    OptKwargs,
    World,
)
from pytorch_quik.ddp import cleanup
from pytorch_quik.bert import get_pretrained_model, BERT_MODELS
import dask_quik as dq
from torch.utils import data
import torch.utils.data as td
from torch import nn
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    logging as tlog,
)
import torch
import sys
import pytest


def test_quik_trek(args):
    if bool(args.gpus) and dq.utils.gpus() == 0:
        print("unable to test quik_trek portion")
        pytest.skip()
    qt = QuikTrek(args.gpu, args)
    assert isinstance(qt, QuikTrek)
    assert isinstance(qt.epochs, int)
    assert isinstance(qt.dlkwargs, DlKwargs)
    assert isinstance(qt.optkwargs, OptKwargs)
    assert isinstance(qt.world, World)
    if args.gpu is not None:
        cleanup()


def test_quik_trek_noargs(gpu):
    sys.argv = ['']
    qt = QuikTrek(gpu)
    assert isinstance(qt, QuikTrek)
    assert isinstance(qt.epochs, int)
    assert isinstance(qt.dlkwargs, DlKwargs)
    assert isinstance(qt.optkwargs, OptKwargs)
    assert isinstance(qt.world, World)
    if gpu is not None:
        cleanup()


def test_quik_traveler(args):
    if bool(args.gpus) and dq.utils.gpus() == 0:
        print("unable to test quik_traveler portion")
        pytest.skip()
    qt = QuikTrek(args.gpu, args)
    qtr = QuikTraveler(qt, "pytest")
    assert isinstance(qtr, QuikTraveler)
    assert isinstance(qtr.amp, QuikAmp)
    if args.gpu is not None:
        cleanup()


def test_quik_data(sample_tds, args):
    if bool(args.gpus) and dq.utils.gpus() == 0:
        print("unable to test quik_data portion")
        pytest.skip()
    qt = QuikTrek(args.gpu, args)
    qtr = QuikTraveler(qt, "pytest")
    qtr.add_data(sample_tds)
    assert(isinstance(qtr.data.dataset, td.TensorDataset))
    assert(isinstance(qtr.data.data_loader, data.DataLoader))
    if args.gpu is not None:
        cleanup()


def test_quik_model_ml(args, sample_labels):
    if bool(args.gpus) and dq.utils.gpus() == 0:
        print("unable to test quik_model_ml portion")
        pytest.skip()
    qt = QuikTrek(args.gpu, args)
    qtr = QuikTraveler(qt, "pytest")
    tlog.set_verbosity_error()
    model = get_pretrained_model(
        labels=sample_labels,
        bert_type=args.bert_type
    )
    qtr.add_model(model)
    qtr.set_criterion(nn.CrossEntropyLoss)
    qtr.set_optimizer(AdamW)
    args.sched_kwargs["num_training_steps"] = 100
    qtr.set_scheduler(get_linear_schedule_with_warmup, args.sched_kwargs)
    assert(isinstance(model, BERT_MODELS[args.bert_type]["model"]))
    assert(isinstance(qtr.criterion, nn.CrossEntropyLoss))
    assert(isinstance(qtr.optimizer, AdamW))
    if args.gpu is not None:
        cleanup()


def test_quik_loss(args, batch):
    if bool(args.gpus) and dq.utils.gpus() == 0:
        print("unable to test quik_model_ml portion")
        pytest.skip()
    qt = QuikTrek(args.gpu, args)
    qtr = QuikTraveler(qt, "pytest")
    tlog.set_verbosity_error()
    model = get_pretrained_model(
        labels=batch[2],
        bert_type=args.bert_type
    )
    qtr.add_model(model)
    qtr.set_criterion(nn.CrossEntropyLoss)
    qtr.set_optimizer(AdamW)
    args.sched_kwargs["num_training_steps"] = 100
    qtr.set_scheduler(get_linear_schedule_with_warmup, args.sched_kwargs)
    batch = [tens.to(qtr.world.device) for tens in batch]
    outputs = qtr.model.forward(input_ids=batch[0], attention_mask=batch[1])
    loss = qtr.criterion(outputs[0], batch[2])
    qtr.backward(loss, clip=True)
    qtr.scheduler.step()
    qtr.add_loss(loss, None, 0)
    if args.gpu is not None:
        cleanup()


def test_record_results(args, sample_tds, batch, two_classes):
    if bool(args.gpus) and dq.utils.gpus() == 0:
        print("unable to test quik_model_ml portion")
        pytest.skip()
    qt = QuikTrek(args.gpu, args)
    qtr = QuikTraveler(qt, "pytest")
    tlog.set_verbosity_error()
    qtr.add_data(sample_tds)
    model = get_pretrained_model(
        labels=batch[2],
        bert_type=args.bert_type
    )
    qtr.add_model(model)
    batch = [tens.to(qtr.world.device) for tens in batch]
    outputs = qtr.model.forward(input_ids=batch[0], attention_mask=batch[1])
    # make some preds opposite
    neg = torch.flip(outputs[0][4:6], [1])
    idx = torch.LongTensor([4, 5]).to(qtr.world.device)
    outputs[0].index_copy_(0, idx, neg)
    qtr.data.add_results(outputs[0], batch[2])
    qtr.record_results(two_classes)
    if args.gpu is not None:
        cleanup()


def test_quik_state_dict(args, sample_labels):
    if bool(args.gpus) and dq.utils.gpus() == 0:
        print("unable to test quik_model_ml portion")
        pytest.skip()
    qt = QuikTrek(args.gpu, args)
    qtr = QuikTraveler(qt, "pytest")
    model = get_pretrained_model(
        labels=sample_labels,
        bert_type=args.bert_type
    )
    qtr.add_model(model)
    qtr.save_state_dict("orig")



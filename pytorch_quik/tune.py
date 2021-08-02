# run all:
# chmod 777 install.sh
# ./install.sh
# python mlflow_creds.py
# pip install --user ray[tune] 
# sudo apt install rsync
# pip install --upgrade aioredis==1.3.1
# python /repos/nps-sentiment/main.py data -s nps -bt bert -l 0 1 2
# python /repos/nps-sentiment/main.py data -s nps -bt roberta -l 0 1 2

from ray import tune
from ray.tune.integration.torch import DistributedTrainableCreator
from ray.tune.schedulers import ASHAScheduler
from functools import partial
import nps_sentiment as ns

# ray tune
def run_ddp_tune(args):
    config = {
        "bert_type": tune.choice(['bert', 'roberta']),
        "learning_rate": tune.loguniform(1e-6, 2e-5),
        "batch_size": tune.choice([8, 16, 24]),
        "beta_max": tune.choice([0.99, 0.999]),
        "eps": tune.loguniform(1e-8, 1e-6),
    }
    tune_scheduler = ASHAScheduler(
        max_t=5,
        grace_period=1,
        reduction_factor=2,
        metric="valid_loss",
        mode="min",
    )
    dist_tune = DistributedTrainableCreator(
        partial(ns.model.fine_tune, args=args),
        num_workers=4,
        num_cpus_per_worker=7,
        num_gpus_per_worker=1,
        backend="nccl",
    )

    result = tune.run(
        dist_tune,
        config=config,
        num_samples=args.num_samples,
        scheduler=tune_scheduler,
    )
    best_trial = result.get_best_trial("valid_loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    return best_trial


def main(num_samples=2, max_num_epochs=5):
    args = ns.model.parse_args()
    args.data_date = '20210728'
    args.bs = 16
    # args.num_samples = 4
    # args.max_num_epochs = 2
    args.use_mlflow = True
    args.mixed_precision = True # no descent when False 
    args.use_ray = True
    args.epochs = 2
    args.num_samples = num_samples
    args.max_num_epochs = max_num_epochs
    args.num_workers = 4
    best_trial = run_ddp_tune(args)
    return best_trial   # test_best_model(best_trial)


bt = main()


# testing decrease
import nps_sentiment as ns
import pytorch_quik as pq
from transformers import AdamW, get_linear_schedule_with_warmup, logging
from nps_sentiment.model import train_pass, valid_pass, train_valid
from torch import nn

gpu = 0
args = ns.model.parse_args()
args.gpus = 1
args.data_date = '20210728'
args.use_mlflow = False
args.use_ray = False
# args.bert_type = 'roberta'
train_valid(gpu, args)

# testing valid
import nps_sentiment as ns
import pytorch_quik as pq
from transformers import AdamW, get_linear_schedule_with_warmup, logging
from nps_sentiment.model import train_pass, valid_pass
from torch import nn

gpu = 0
args = ns.model.parse_args()
args.gpus = 1
args.data_date = '20210728'
args.bs = 16
# args.num_samples = 4
# args.max_num_epochs = 2
args.use_mlflow = False
args.mixed_precision = True
args.use_ray = False
# args.bert_type = 'roberta'
args.epochs = 5  
trek = pq.travel.QuikTrek(gpu=gpu, args=args)
tr = pq.travel.QuikTraveler(trek, "train")
logging.set_verbosity_error()
tr.add_data(pq.io.load_torch_object("train", args))
vl = pq.travel.QuikTraveler(trek, "valid")
vldata = pq.io.load_torch_object("valid", args)
vl.add_data(vldata)
model = pq.bert.get_pretrained_model(
    labels=tr.data.labels,
    bert_type=args.bert_type
)
tr.add_model(model)
tr.set_criterion(nn.CrossEntropyLoss)
tr.set_optimizer(AdamW)
args.sched_kwargs["num_training_steps"] = tr.data.total_steps
tr.set_scheduler(get_linear_schedule_with_warmup, args.sched_kwargs)
for epoch in range(args.epochs):
    train_pass(tr, epoch, args)
    valid_pass(vl, epoch, args)

    tr.metrics.write()
    if tr.world.is_logger:
        print(tr.metrics.results)


# optuna - too experimental
import nps_sentiment as ns
import dask_quik as dq
import pytorch_quik as pq
import optuna as ot
from optuna.integration import TorchDistributedTrial as ddpTrial
from optuna.trial import Trial
from transformers import AdamW, get_linear_schedule_with_warmup, logging
from torch import nn


def objective(trial: Trial):
    logging.set_verbosity_error()
    # gpu = trial._trial_id % dq.utils.gpus()
    args = ns.model.parse_args()
    args.data_date = '20210728'
    args.epochs = 1
    args.gpus = 4
    args.test_epoch = 0
    args.use_mlflow = False
    # args.trial_id = trial._trial_id
    # print(args.trial_id)
    # args.bert_type = trial.suggest_categorical('bert_type', ['bert', 'roberta'])
    print("starting trek")
    trek = pq.travel.QuikTrek(gpu, args)
    print("distributing trial")
    distributed_trial = ddpTrial(trial)
    args.lr = distributed_trial.suggest_float('lr', '1e-6', '2e-5')
    print("trek arg:" + str(trek.args.use_mlflow))
    print("starting traveler")
    tr = pq.travel.QuikTraveler(trek, "train")
    print("loading: " + str(pq.io.id_str("train", args)))
    tr.add_data(pq.io.load_torch_object("train", args))
    vl = pq.travel.QuikTraveler(trek, "valid")
    vldata = pq.io.load_torch_object("valid", args)
    vl.add_data(vldata)
    model = pq.bert.get_pretrained_model(
        labels=tr.data.labels,
        bert_type=args.bert_type
    )
    tr.add_model(model)
    tr.set_criterion(nn.CrossEntropyLoss)
    tr.set_optimizer(AdamW)
    args.sched_kwargs["num_training_steps"] = tr.data.total_steps
    tr.set_scheduler(get_linear_schedule_with_warmup, args.sched_kwargs)
    tr.save_state_dict("orig")
    for epoch in range(args.epochs):
        ns.model.train_pass(tr, epoch, args)
        ns.model.valid_pass(vl, epoch, args)
        tr.metrics.write()
        print(tr.metrics.results)
    # test
    tst = pq.travel.QuikTraveler(trek, "test")
    tst.add_data(pq.io.load_torch_object("test", args))
    pred_tens, true_vals = ns.model.test_pass(tst)
    tst.data.add_results(pred_tens, true_vals)
    idx_lbls = pq.utils.indexed_dict(["negative", "neutral", "positive"])
    print("before record")
    print("trek arg:" + str(trek.args.use_mlflow))
    print("test arg:" + str(tst.args.use_mlflow))
    tst.record_results(idx_lbls)
    accuracy = tst.cr["weighted_avg_f1-score"]
    # trek.mlflow.end_run()
    return accuracy


study = ot.create_study()
study.optimize(objective, n_trials=6, n_jobs=4)

# pq.ddp.cleanup()


args = ns.model.parse_args()
args.data_date = '20210728'
args.epochs = 1
args.gpus = 1
args.test_epoch = 0
args.use_mlflow = False
gpu = 1
trek = pq.travel.QuikTrek(gpu, args)
tr = pq.travel.QuikTraveler(trek, "train")
tr.add_data(pq.io.load_torch_object("train", args))
for batch in tr.data.data_loader:
    print(batch.rank)
    print(batch.total_size)
    print(batch.num_replicas)

    self.rank:self.total_size:self.num_replicas

print(trial.rank)
print(trial.total_size)
print()

# regular dist works!
import nps_sentiment as ns
import pytorch_quik as pq
args = ns.model.parse_args()
args.data_date = '20210728'
args.epochs = 1
args.test_epoch = 0
args.use_mlflow = False
pq.ddp.traverse(ns.model.train_valid, args)


# 
import nps_sentiment as ns
import pytorch_quik as pq
import optuna as ot
from optuna.integration import TorchDistributedTrial as ddpTrial
from optuna.trial import Trial
args = ns.model.parse_args()
args.data_date = '20210728'
args.epochs = 1
args.test_epoch = 0
args.use_mlflow = False

def objective(trial: Trial):
    args.trial = trial
    accuracy = pq.ddp.traverse(ns.model.trial_train_valid_test, args)
    return accuracy

study = ot.create_study()
study.optimize(objective, n_trials=4, n_jobs=4)
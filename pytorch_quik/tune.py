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
from ray.tune.integration.torch import DistributedTrainableCreator as DTC
from ray.tune.schedulers import ASHAScheduler, MedianStoppingRule
from functools import partial
import nps_sentiment as ns
from ruamel.yaml import YAML, YAMLError
from typing import Callable, Dict, Any
from argparse import Namespace

RESOURCES = {
    "num_workers": 4,
    "num_cpus_per_worker": 7,
    "num_gpus_per_worker": 1,
    "backend": "nccl",
}

def get_tune_config(filename: str) -> Dict[str, Callable]:
    with open(filename, "r") as stream:
        try:
            yaml_dict = YAML().load(stream)
        except YAMLError as e:
            print(e)
    return {key: getattr(tune, list(dist.keys())[0])(**list(dist.values())[0]) for key, dist in yaml_dict.items()}



def run_ddp_tune(
    train_fcn: Callable, args: Namespace, tune_config: Dict[str, Any]
    ):

    # tune_scheduler = ASHAScheduler(
    #     max_t=5,
    #     grace_period=1,
    #     reduction_factor=2,
    #     metric="valid_loss",
    #     mode="min",
    # )
    tune_scheduler = MedianStoppingRule(
        metric="valid_loss",
        mode="min",
    )
    dist_tune = DTC(partial(train_fcn, args=args), **RESOURCES)

    result = tune.run(
        dist_tune,
        config=tune_config,
        num_samples=args.num_samples,
        scheduler=tune_scheduler,
    )
    # best_trial = result.get_best_trial("valid_loss", "min", "last")
    # print("Best trial config: {}".format(best_trial.config))
    return result

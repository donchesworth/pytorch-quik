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
from ruamel.yaml import YAML, YAMLError
from typing import Callable, Dict, Union
from argparse import Namespace
import sys
import logging

scheds = Union[ASHAScheduler, MedianStoppingRule]

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

RESOURCES = {
    "num_workers": 4,
    "num_cpus_per_worker": 7,
    "num_gpus_per_worker": 1,
    "backend": "nccl",
}

SCHED = {
    "kwargs": {
        "metric": "valid_loss",
        "mode": "min",
    },
    "asha_kwargs": {
        "max_t": 5,
        "grace_period": 1,
        "reduction_factor": 2,
    },
}


def get_tune_config(filename: str) -> Dict[str, Callable]:
    """In order to have Ray Tune come up with permutations of hyperparameters,
    it will need to know the boundaries. get_tune_config will pull a list of
    parameters, and their function for choice. The two common functions are
    sample.Categorical, and sample.Float.

    Args:
        filename (str): The directory + filename for the yaml file

    Returns:
        Dict[str, Callable]: The config, consisting of a parameter name and a
        ray.tune.sample function.
    """
    with open(filename, "r") as stream:
        try:
            yaml_dict = YAML().load(stream)
        except YAMLError as e:
            print(e)
    return {
        key: getattr(tune, list(dist.keys())[0])(**list(dist.values())[0])
        for key, dist in yaml_dict.items()
    }


def run_ddp_tune(
    train_fcn: Callable,
    args: Namespace,
    tune_config: Dict[str, Callable],
    tune_scheduler: scheds,
) -> tune.ExperimentAnalysis:
    """run a Ray Tune hyperparameter search, distributed across
    GPUs.

    Args:
        train_fcn (Callable): The function from your project that will train
        (and validate) a model
        args (Namespace): The list of arguments neeed by the
        DistributedTrainableCreator
        tune_config (Dict[str, Any]): The output of get_tune_config
        tune_scheduler (scheds): The tune scheduler (either ASHAScheduler
        or MedianStoppingRule)

    Returns:
        tune.ExperimentAnalysis: The output of the grid search (also
        found in MLFlow if used).
    """

    if tune_scheduler == ASHAScheduler:
        SCHED["kwargs"].update(SCHED["asha_kwargs"])

    dist_tune = DTC(partial(train_fcn, args=args), **RESOURCES)

    result = tune.run(
        dist_tune,
        config=tune_config,
        num_samples=args.num_samples,
        scheduler=tune_scheduler(SCHED["kwargs"]),
    )
    best_trial = result.get_best_trial("valid_loss", "min", "last")
    logger.info(f"Best trial config: {best_trial.config}")
    return result

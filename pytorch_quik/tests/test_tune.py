from pathlib import Path
from pytorch_quik.tune import get_tune_config


def test_get_tune_config():
    filename = Path(__file__).parent.joinpath("tune_params.yaml")
    tune_params = get_tune_config(filename)
    for sample in tune_params.values():
        assert(sample.__module__ == 'ray.tune.sample')

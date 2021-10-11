from pytorch_quik import arg
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, Namespace
import sys


def test_learn_args():
    sys.argv = [""]
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = arg.add_learn_args(parser)
    args = parser.parse_args()
    assert isinstance(args, Namespace)


def test_mlflow_args():
    sys.argv = [""]
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = arg.add_learn_args(parser)
    args = parser.parse_args()
    assert isinstance(args, Namespace)


def test_ray_tune_args():
    sys.argv = [""]
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser = arg.add_learn_args(parser)
    args = parser.parse_args()
    assert isinstance(args, Namespace)

import pytorch_quik as pq
from pathlib import Path


def test_id_str(args):
    """print an id str"""
    filename = "train_tensor_20210101.pt"
    filename = Path.cwd().joinpath("data", args.bert_type, filename)
    # assert(pq.utils.id_str("train", args) == output)
    output = pq.io.id_str("train", args)
    print(output)
    assert filename == output

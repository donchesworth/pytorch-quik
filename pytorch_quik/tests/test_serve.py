from pytorch_quik import serve, bert
from pathlib import Path
import json
from collections import OrderedDict
from importlib import import_module
import pytest

SERVE_PATH = Path(__file__).parent
SCFILE = 'setup_config.json'
ITNFILE = 'index_to_name.json'
STFILE = 'sample_text.json'
THFILE = 'transformer_handler_pq'


def rejson(filename):
    filename = SERVE_PATH.joinpath(filename)
    with open(filename) as f:
        output = json.load(f)
    filename.unlink()
    return output


def test_save_setup_config(senti_classes, args):
    labels = senti_classes.keys()
    serve.save_setup_config(SERVE_PATH, labels, args)
    output_config = rejson(SCFILE)
    serve_config = bert.get_bert_info(labels, args.bert_type)
    assert serve_config == output_config


def test_save_index_to_name(senti_classes):
    serve.save_index_to_name(SERVE_PATH, senti_classes)
    out = rejson(ITNFILE)
    output_classes = OrderedDict((int(k), v) for k, v in out.items())
    assert senti_classes == output_classes


def test_save_sample():
    serve.save_sample(SERVE_PATH)
    output_txt = rejson(STFILE)
    txt = '{"instances":[{"data": "Great company with fast support"}]}'
    assert txt == output_txt


def test_save_handler():
    serve.save_handler(SERVE_PATH)
    with pytest.raises(ModuleNotFoundError) as e_info:
        import_module(f"pytorch_quik.tests.{THFILE}")
    assert e_info.value.msg == "No module named 'captum'"
    SERVE_PATH.joinpath(f'{THFILE}.py').unlink()


def test_build_extra_files(args, senti_classes):
    serve.build_extra_files(args, senti_classes, SERVE_PATH)
    files = [SCFILE, ITNFILE, STFILE, f'{THFILE}.py']
    for f in files:
        file = SERVE_PATH.joinpath(f)
        assert file.is_file()
        file.unlink()

from pytorch_quik.bert import (
    BERT_MODELS,
    get_encodings,
    get_pretrained_model,
    get_tokenizer,
    get_bert_info,
    save_tokenizer,
)
from pathlib import Path
from transformers import BatchEncoding, logging as tlog


def test_get_bert_info(sample_labels):
    for bert_type in BERT_MODELS.keys():
        serve_config = get_bert_info(bert_type)
        assert isinstance(serve_config, dict)


def test_get_and_save_tokenizer():
    tlog.set_verbosity_error()
    for bert_type, values in BERT_MODELS.items():
        tokenizer = get_tokenizer(bert_type)
        assert isinstance(tokenizer, values["tokenizer"])
        data_path = Path.cwd()
        save_tokenizer(tokenizer, data_path)


def test_get_encodings(sample_data):
    tlog.set_verbosity_error()
    for bert_type, values in BERT_MODELS.items():
        tokenizer = get_tokenizer(bert_type)
        encode_list, tokenizer = get_encodings(
            [sample_data.values], 0, bert_type, tokenizer)
        assert isinstance(tokenizer, values["tokenizer"])
        assert isinstance(encode_list[0], BatchEncoding)


def test_get_bert_model(sample_labels):
    tlog.set_verbosity_error()
    for bert_type, values in BERT_MODELS.items():
        model = get_pretrained_model(labels=sample_labels, bert_type=bert_type)
        assert isinstance(model, values["model"])

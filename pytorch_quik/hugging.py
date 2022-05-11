import numpy as np
from torch import Tensor
from transformers import (
    BertTokenizer,
    RobertaTokenizer,
    MarianTokenizer,
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    MarianMTModel,
    BatchEncoding,
    PreTrainedTokenizer,
    PreTrainedModel,
    logging as tlog,
)
from typing import Optional, List, Union, Dict, OrderedDict
import shutil
from pathlib import Path
import json
from . import io, ddp
from argparse import Namespace
from torch.nn.parallel import DistributedDataParallel as DDP
import sys
import logging
from functools import partial

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

Tokenizers = Union[BertTokenizer, RobertaTokenizer]
Models = Union[BertForSequenceClassification, RobertaForSequenceClassification]

# ESEN = {"src": "es", "tgt": "en"}
HUGGING_MODELS = {
    "bert": {
        "model-name": "bert-base-uncased",
        "tokenizer": BertTokenizer,
        "model-head": BertForSequenceClassification,
        "mode": "sequence_classification",
    },
    "roberta": {
        "model-name": "roberta-base",
        "tokenizer": RobertaTokenizer,
        "model-head": RobertaForSequenceClassification,
        "mode": "sequence_classification",

    },
    "marianmt": {
        "model-name": partial("Helsinki-NLP/opus-mt-{source}-{target}".format),
        "tokenizer": MarianTokenizer,
        "model-head": MarianMTModel,
        "mode": "sequence_to_sequence",
    }
}

BATCH_ENCODE_KWARGS = {
    "add_special_tokens": True,
    "return_attention_mask": True,
    "pad_to_max_length": True,
    "max_length": 256,  # max is 512
    "truncation": True,
    "return_tensors": "pt",
}


def model_info(
    model_type: Optional[str] = "bert",
    labels: Optional[Tensor] = None,
    kwargs: Optional[dict] = None,
) -> Dict[str, Union[str, bool]]:

    """Create a serving config dictionary for model serving.

    Args:
        model_type (str, optional): The type of the model to be used ("bert",
        "roberta", or "marianmt"). Defaults to "bert".
        labels (Tensor, optional): A tensor of the labels in the dataset. This
        will affect the size of the network final layer
        kwargs (dict, optional): If the model-name is a partial (e.g. MarianMT)
        this will provide the missing text. Default is español to english.

    Returns:
        Dict [str, Any]: A dict that could be provided for torch serve
    """
    model_dict = HUGGING_MODELS[model_type]
    model_name = model_dict["model-name"]
    if isinstance(model_name, partial):
        model_name = model_name(**kwargs)
    serve_config = {
        "model_name": model_name,
        "mode": model_dict["mode"],
        "do_lower_case": True,
        "save_mode": "pretrained",
        "max_length": BATCH_ENCODE_KWARGS["max_length"],
        "captum_explanation": False,
        "embedding_name": model_type,
    }
    if labels:
        serve_config["num_labels"] = str(len(np.unique(labels)))
    return serve_config


def get_tokenizer(
    model_type: Optional[str] = "bert",
    cache_dir: Optional[str] = None,
    kwargs: Optional[dict] = {},
) -> Tokenizers:
    """Get a BERT, RoBERTa, or MarianMT tokenizer from transformers to create
    encodings.

    Args:
        model_type (str, optional): The type of tokenizer ("bert", "roberta",
        "marianmt"). Defaults to "bert".
        cache_dir (str, optional): The directory to cache the tokenizer. This
        helps save_tokenizer find it.
        kwargs (dict, optional): If a MarianMT mode, the source and target
        languages

    Returns:
        Tokenizers: The tokenizer to be used by get_encodings.
    """
    model_dict = HUGGING_MODELS[model_type]
    model_name = model_dict["model-name"]
    if isinstance(model_name, partial):
        model_name = model_name(**kwargs)
    for _ in range(5):
        while True:
            try:
                tokenizer = model_dict["tokenizer"].from_pretrained(
                    pretrained_model_name_or_path=model_name,
                    do_lower_case=True,
                    cache_dir=cache_dir,
                )
            except ValueError:
                logger.info("Connection error, trying again")
                continue
            break
    return tokenizer


def update_tokenizer_path(serve_path):
    configpath = Path(serve_path).joinpath("tokenizer_config.json")
    newtokenizer = "./tokenizer.json"
    with open(configpath) as f:
        tconfig = json.load(f)
    srcpath = Path(tconfig["tokenizer_file"])
    dstpath = serve_path.joinpath(newtokenizer)
    shutil.copyfile(srcpath, dstpath)
    tconfig["tokenizer_file"] = newtokenizer
    io.json_write(serve_path, configpath, tconfig)


def save_tokenizer(tokenizer: Tokenizers, serve_path: Path):
    """Serving a torch model requires saving the tokenizer, and
    it needs to be in the path that the model archive is built.

    Args:
        tokenizer (Tokenizers): The BERT or RoBERTa tokenizer used
        serve_path (Path): The location for building the model archive
    """
    tokenizer.save_pretrained(serve_path)
    if not isinstance(tokenizer, MarianTokenizer):
        update_tokenizer_path(serve_path)


def get_encodings(
    array_list: List[np.ndarray],
    idx: int,
    bert_type: Optional[str] = "bert",
    tokenizer: Optional[PreTrainedTokenizer] = None,
    kwargs: Optional[dict] = {},
) -> List[BatchEncoding]:
    """Get BERT or RoBERTa encodings for a list of np.ndarrays for training,
    testing (and validation).

    Args:
        array_list (List[np.ndarray]): The list of training, test, (and
        validation) arrays.
        idx (int): The column index of the text in the original pd.DataFrame
        bert_type (str, optional): The type of bert model to be used. Defaults
        to "bert".
        tokenizer (PreTrainedTokenizer, optional): The tokenizer to use
        kwargs (dict, optional): For overriding BATCH_ENCODE_KWARGS

    Returns:
        List[BatchEncoding]: A list of training, test (and validation)
        encodings of text.
    """
    BATCH_ENCODE_KWARGS.update(kwargs)
    logger.info(f"kwargs are now: {BATCH_ENCODE_KWARGS['return_tensors']}")
    if tokenizer is None:
        tokenizer = get_tokenizer(bert_type)
    encode_list = [
        tokenizer(list(array[:, idx]), **BATCH_ENCODE_KWARGS)
        for array in array_list
    ]
    return encode_list, tokenizer


def get_pretrained_model(
    model_type: Optional[str] = "bert",
    labels: Optional[Tensor] = None,
    kwargs: Optional[dict] = {},
) -> Models:
    """Get a pretrained model from transformers.

    Args:
        labels (Tensor, optional): A tensor of the labels in the dataset. This will
        affect the size of the network final layer
        model_type (str, optional): The type of the model to be used ("bert",
        "roberta", or "marianmt"). Defaults to "bert".

    Returns:
        Models: The pretrained model to be used in training.
    """
    if labels is not None:
        num_lbls = len(np.unique(labels))
    model_dict = HUGGING_MODELS[model_type]
    model_name = model_dict["model-name"]
    if model_type in ["bert", "roberta"]:
        kwargs = {
            "num_labels": num_lbls,
            "output_attentions": False,
            "output_hidden_states": False,
        }
    else:
        if isinstance(model_name, partial):
            model_name = model_name(**kwargs)
            kwargs = {}
    tlog.set_verbosity_error()
    for i in range(0, 3):
        while True:
            try:
                model = model_dict["model-head"].from_pretrained(
                    model_name,
                    **kwargs
                )
            except ValueError:
                logger.info("Connection error, trying again")
                continue
            break
    return model


def save_pretrained_model(
    model: PreTrainedModel,
    args: Optional[Namespace] = None,
    best_epoch: Optional[int] = None,
    serve_path: Optional[Path] = None,
    state_dict: Optional[OrderedDict] = None
):
    """Serving a torch model requires saving the model, and
    it needs to be in the path that the model archive is built.

    Args:
        model (PreTrainedModel): The BERT or RoBERTa model (for this
        purpose post-training)
        args (Namespace): The list of arguments for building paths
        best_epoch (int): The epoch from which to pull the state_dict
        serve_path (Path, optional): Directory to store files for building
        the model archive. Defaults to None.
        state_dict (OrderedDict, optional) Model state dictionary. Could be
        model.state_dict() if using default
    """
    if isinstance(model, DDP):
        model = model.module
    if serve_path is None:
        serve_path = io.id_str("", args).parent
    if state_dict is None:
        state_dict = io.load_torch_object("state_dict", args, best_epoch)
    if "module.classifier.bias" in state_dict:
        logger.info("Removing distribution from model")
        state_dict = ddp.consume_prefix_in_state_dict_if_present(
            state_dict, "module."
        )
    model.save_pretrained(save_directory=serve_path, state_dict=state_dict)

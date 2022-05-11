import numpy as np
from torch import Tensor
from transformers import (
    BertTokenizer,
    RobertaTokenizer,
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    BatchEncoding,
    PreTrainedTokenizer,
    PreTrainedModel,
    logging as tlog,
)
from typing import Optional, List, Union, Dict
import shutil
from pathlib import Path
import json
from . import io, ddp
from argparse import Namespace
from torch.nn.parallel import DistributedDataParallel as DDP
import sys
import logging

logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

Tokenizers = Union[BertTokenizer, RobertaTokenizer]
Models = Union[BertForSequenceClassification, RobertaForSequenceClassification]

BERT_MODELS = {
    "bert": {
        "model-type": "bert-base-uncased",
        "tokenizer": BertTokenizer,
        "model": BertForSequenceClassification,
    },
    "roberta": {
        "model-type": "roberta-base",
        "tokenizer": RobertaTokenizer,
        "model": RobertaForSequenceClassification,
    },
}

BATCH_ENCODE_KWARGS = {
    "add_special_tokens": True,
    "return_attention_mask": True,
    "pad_to_max_length": True,
    "max_length": 256,  # max is 512
    "truncation": True,
    "return_tensors": "pt",
}


def get_bert_info(
    labels: Tensor, bert_type: Optional[str] = "bert"
) -> Dict[str, Union[str, bool]]:

    """Get a BERT or RoBERTa tokenizer from transformers to create
    encodings.
    Args:
        labels (Tensor): A tensor of the labels in the dataset. This will
        affect the size of the network final layer
        bert_type (str, optional): The type of the model to be used ("bert" or
        "roberta"). Defaults to "bert".

    Returns:
        Dict [str, Any]: A dict that could be provided for torch serve
    """
    bert_dict = BERT_MODELS[bert_type]
    num_lbls = len(np.unique(labels))
    serve_config = {
        "model_name": bert_dict["model-type"],
        "mode": "sequence_classification",
        "do_lower_case": True,
        "num_labels": str(num_lbls),
        "save_mode": "pretrained",
        "max_length": BATCH_ENCODE_KWARGS["max_length"],
        "captum_explanation": False,
        "embedding_name": bert_type,
    }
    return serve_config


def get_tokenizer(
    bert_type: Optional[str] = "bert", cache_dir: Optional[str] = None
) -> Tokenizers:
    """Get a BERT or RoBERTa tokenizer from transformers to create
    encodings.

    Args:
        bert_type (str, optional): The type of tokenizer ("bert" or "roberta").
        Defaults to "bert".

    Returns:
        Tokenizers: The tokenizer to be used by get_encodings.
    """
    bert_dict = BERT_MODELS[bert_type]
    for i in range(0, 3):
        while True:
            try:
                tokenizer = bert_dict["tokenizer"].from_pretrained(
                    pretrained_model_name_or_path=bert_dict["model-type"],
                    do_lower_case=True,
                    cache_dir=cache_dir,
                )
            except ValueError:
                logger.info("Connection error, trying again")
                continue
            break
    return tokenizer


def save_tokenizer(tokenizer: Tokenizers, data_path: Path):
    """Serving a torch model requires saving the tokenizer, and
    it needs to be in the path that the model archive is built.

    Args:
        tokenizer (Tokenizers): The BERT or RoBERTa tokenizer used
        data_path (Path): The location for building the model archive
    """
    serve_path = data_path.joinpath("serve")
    tokenizer.save_pretrained(serve_path)
    configpath = Path(serve_path).joinpath("tokenizer_config.json")
    newtokenizer = "./tokenizer.json"
    with open(configpath) as f:
        tconfig = json.load(f)
    srcpath = Path(tconfig["tokenizer_file"])
    dstpath = serve_path.joinpath(newtokenizer)
    shutil.copyfile(srcpath, dstpath)
    tconfig["tokenizer_file"] = newtokenizer
    io.json_write(serve_path, configpath, tconfig)


def save_bert_model(
    model: PreTrainedModel,
    args: Namespace,
    best_epoch: int,
    serve_path: Optional[Path] = None,
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
    """
    if isinstance(model, DDP):
        model = model.module
    if serve_path is None:
        serve_path = io.id_str("", args).parent
    state_dict = io.load_torch_object("state_dict", args, best_epoch)
    if "module.classifier.bias" in state_dict:
        logger.info("Removing distribution from model")
        state_dict = ddp.consume_prefix_in_state_dict_if_present(
            state_dict, "module."
        )
    model.save_pretrained(save_directory=serve_path, state_dict=state_dict)


def get_encodings(
    array_list: List[np.ndarray],
    idx: int,
    bert_type: Optional[str] = "bert",
    tokenizer: Optional[PreTrainedTokenizer] = None,
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

    Returns:
        List[BatchEncoding]: A list of training, test (and validation)
        encodings of text.
    """
    if tokenizer is None:
        tokenizer = get_tokenizer(bert_type)
    encode_list = [
        tokenizer(list(array[:, idx]), **BATCH_ENCODE_KWARGS)
        for array in array_list
    ]
    return encode_list, tokenizer


def get_pretrained_model(
    labels: Tensor, bert_type: Optional[str] = "bert"
) -> Models:
    """Get a pretrained model from transformers.

    Args:
        labels (Tensor): A tensor of the labels in the dataset. This will
        affect the size of the network final layer
        bert_type (str, optional): The type of the model to be used ("bert" or
        "roberta"). Defaults to "bert".

    Returns:
        Models: The pretrained model to be used in training.
    """
    num_lbls = len(np.unique(labels))
    bert_dict = BERT_MODELS[bert_type]
    tlog.set_verbosity_error()
    for i in range(0, 3):
        while True:
            try:
                model = bert_dict["model"].from_pretrained(
                    bert_dict["model-type"],
                    num_labels=num_lbls,
                    output_attentions=False,
                    output_hidden_states=False,
                )
            except ValueError:
                logger.info("Connection error, trying again")
                continue
            break
    return model

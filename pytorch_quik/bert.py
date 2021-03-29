import numpy as np
from torch import Tensor
from transformers import (
    BertTokenizer,
    RobertaTokenizer,
    BertForSequenceClassification,
    RobertaForSequenceClassification,
    BatchEncoding,
)
from typing import Optional, List, Union

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


def get_tokenizer(bert_type: Optional[str] = "bert") -> Tokenizers:
    """Get a BERT or RoBERTa tokenizer from transformers to create
    encodings.

    Args:
        bert_type (str, optional): The type of tokenizer ("bert" or "roberta").
        Defaults to "bert".

    Returns:
        Tokenizers: The tokenizer to be used by get_encodings.
    """
    bert_dict = BERT_MODELS[bert_type]
    tokenizer = bert_dict["tokenizer"].from_pretrained(
        bert_dict["model-type"], do_lower_case=True
    )
    return tokenizer


def get_encodings(
    array_list: List[np.ndarray], idx: int, bert_type: Optional[str] = "bert"
) -> List[BatchEncoding]:
    """Get BERT or RoBERTa encodings for a list of np.ndarrays for training,
    testing (and validation).

    Args:
        array_list (List[np.ndarray]): The list of training, test, (and
        validation) arrays.
        idx (int): The column index of the text in the original pd.DataFrame
        bert_type (str, optional): The type of bert model to be used. Defaults
        to "bert".

    Returns:
        List[BatchEncoding]: A list of training, test (and validation)
        encodings of text.
    """
    tknzr = get_tokenizer(bert_type)
    encode_list = [
        tknzr.batch_encode_plus(array[:, idx], **BATCH_ENCODE_KWARGS)
        for array in array_list
    ]
    return encode_list


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
    model = bert_dict["model"].from_pretrained(
        bert_dict["model-type"],
        num_labels=num_lbls,
        output_attentions=False,
        output_hidden_states=False,
    )
    return model

import json
from pathlib import Path
from pytorch_quik import bert, io
from typing import Optional, List, OrderedDict, KeysView
from argparse import Namespace
import shlex
import subprocess
import requests
from urllib.parse import urlparse

EXTRA_FILES = [
    "config.json",
    "setup_config.json",
    "index_to_name.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "vocab.txt",
]


def save_setup_config(serve_path: str, labels: KeysView, args: Namespace):
    """Create a setup_config json for the huggingface torch serve handler

    Args:
        serve_path (str): The directory to store the sample.
        labels (KeysView): the dictionary keys of index_labels
        args (Namespace): the project argparse namespace.
    """
    serve_config = bert.get_bert_info(labels, args.bert_type)
    io.json_write(serve_path, "setup_config.json", serve_config)


def save_index_to_name(serve_path: str, indexed_labels: OrderedDict[int, str]):
    """Create the required index_to_name file for serving

    Args:
        serve_path (str): The directory to store the sample.
        indexed_labels (OrderedDict[str, int]): the target labels with indexes.
    """
    io.json_write(serve_path, "index_to_name.json", indexed_labels)


def inference_data(data: List[str]) -> str:
    """Format any data into an inference-ready json

    Args:
        data (List[str]): a list of inputs

    Returns:
        str: the final json-ready inference data.
    """
    dlist = [{"data": text} for text in data]
    data = f'{{"instances": {json.dumps(dlist)}}}'
    return data


def save_sample(serve_path):
    """A sample input to test the serving model

    Args:
        serve_path (str): The directory to store the sample.
    """
    sample = ["Great company with fast support"]
    sample = inference_data(sample)
    io.json_write(serve_path, "sample_text.json", sample)


def save_handler(serve_path, url: Optional[str] = None):
    """Download the handler file for serving.

    Args:
        serve_path ([type]): the torch serve directory
        url (str, optional): The url where the handler can be found.
        Defaults to None.
    """
    if url is None:
        url = urlparse("https://raw.githubusercontent.com")
        handler_loc = Path(
            "donchesworth/pytorch-quik/main/",
            "pytorch_quik",
            "handler",
            "transformer_handler_pq.py",
        )
        url = url._replace(path=str(handler_loc))
    filename = Path(serve_path).joinpath(handler_loc.name)
    r = requests.get(url.geturl(), allow_redirects=True)
    open(filename, "wb").write(r.content)


def mar_files(mar_path: Path, sfile: str, hfile: str) -> bool:
    """check of all mar files exist before attempting to
    run torch-model-archiver

    Args:
        mar_path (Path): the directory containing all the mar files.
        sfile (str): the serialized filename
        hfile (str): the handler filename

    Returns:
        bool: Whether all mar files were found.
    """
    all_files = [sfile, hfile]
    all_files.extend(EXTRA_FILES)
    missing = [
        file for file in all_files if not mar_path.joinpath(file).is_file()
    ]
    if len(missing) == 0:
        return True
    else:
        print("These files are missing:")
        print(missing)
        return False


def build_extra_files(
    args: Namespace,
    indexed_labels: OrderedDict[str, int],
    serve_path: Optional[Path] = None,
):
    """build all the boiler plate files to create
    an mar

    Args:
        args (Namespace): the project argparse namespace.
        indexed_labels (OrderedDict[str, int]): the target labels with indexes.
        serve_path (Optional[Path], optional): The diretory to contain
        all serve files. Defaults to None.
    """
    if serve_path is None:
        serve_path = io.id_str("state_dict", args).parent.joinpath("serve")
    serve_path.mkdir(parents=True, exist_ok=True)
    save_setup_config(serve_path, indexed_labels.keys(), args)
    save_index_to_name(serve_path, indexed_labels)
    save_sample(serve_path)
    save_handler(serve_path)


def create_mar(
    args: Namespace,
    model_dir: Optional[Path] = None,
    model_name: Optional[str] = None,
    version: Optional[float] = 1.0,
    serialized_file: Optional[str] = None,
    handler: Optional[str] = "transformer_handler_pq.py",
):
    """build a torch-model-archive file using
    https://github.com/pytorch/serve/tree/master/model-archiver

    Args:
        args (Namespace): the project argparse namespace.
        model_dir (Path, optional): the directory containing
        mar inputs, and the output directory. Defaults to None.
        model_name (str, optional): the name of the model to be served.
        Defaults to None.
        version (float, optional): the model version. Defaults to 1.0.
        serialized_file (str, optional): the model output of save_pretrained().
        Defaults to None.
        handler (str, optional): the serving handler file.
        Defaults to "transformer_handler_pq.py".
    """
    if model_dir is None:
        model_dir = Path(io.id_str("", args)).parent.joinpath("serve")
    export_dir = model_dir.joinpath("mar")
    export_dir.mkdir(parents=True, exist_ok=True)
    xfiles = ",./".join(shlex.quote(x) for x in EXTRA_FILES)
    if model_name is None:
        model_name = args.experiment.replace("-", "_")
    if serialized_file is None:
        serialized_file = "pytorch_model.bin"
    if mar_files(model_dir, serialized_file, handler):
        # --model-file=./model.py
        cmd = f"""torch-model-archiver
            --model-name={model_name}
            --version={version}
            --serialized-file=./{serialized_file}
            --handler=./{handler}
            --extra-files "./{xfiles}"
            --export-path={export_dir}
        """
        sp = subprocess.Popen(shlex.split(cmd), cwd=model_dir)
        sp.communicate()
        print(f"torch archive {model_name}.mar created")

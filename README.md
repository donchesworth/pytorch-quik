![example workflow](https://github.com/donchesworth/pytorch-quik/actions/workflows/github-ci.yml/badge.svg)
[![](https://img.shields.io/pypi/v/pytorch-quik.svg)](https://pypi.org/pypi/name/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Docker Repository on Quay](https://quay.io/repository/donchesworth/rapids-dask-pytorch/status "Docker Repository on Quay")](https://quay.io/repository/donchesworth/rapids-dask-pytorch)
[![code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/donchesworth/pytorch-quik/branch/main/graph/badge.svg?token=U92M8C8AFM)](https://codecov.io/gh/donchesworth/pytorch-quik)

# pytorch-quik   <img src="pytorch-quik.png" align="right" width="120" />

## For the quick-er distribution of neural nets in PyTorch

As I was building out the same set of code for a recommender system, a BERT sentiment model, and a co-worker was about to build a classification model, I decided to standardize the code into this package. It's lightweight, because I didn't want to hide the standard steps to keep the user from learning Neural Networks, but I also didn't want to maintain the code in multiple places.

## Updates: 
### v0.3.0 (10/25/2021)
- Integrated ray tune for hyperparameter optimization using `use_ray_tune = True`
- Added needed unit tests to bring code coverage to 85%
- Updated README
### v0.2.0 (07/22/2021)
- Integrated MLflow into `QuikTrek` using `QuikMlflow` and `use_mlflow = True`
- Added utilities to create all required files for a transformers TorchServe model archive
- Added utilities to build a dataset and query a TorchServe API
### v0.1.0 (06/03/2021)
- Refactored to provide classes `QuikTrek`, `QuikTraveler`, `QuikData`
- Slight change from 0.0.1, where now you need `trek = pq.travel.QuikTrek(args, gpu); tr = pq.travel.QuikTraveler(trek)`
- Added documentation

### Installation

``` bash
pip install pytorch-quik
```

# Usage

## Intro, QuikTrek, QuikTraveler

In its simplest form, you'll want to:

- create a QuikTrek (an object that stores overall attributes and various travelers across your neural network)
- create a QuikTraveler (an object that "travels" forward and backward on your neural network)
- add QuikData
- add a criterion, an optimizer
- train

``` python
import pytorch_quik as pq
from torch import nn, optim

trek = pq.travel.QuikTrek()
tr = pq.travel.QuikTraveler(trek)
tr.add_data(pq.io.load_torch_object("train", tr.args))

# from pytorch.org/tutorials/beginner/pytorch_with_examples.html
model = nn.Sequential(nn.Linear(3, 1), nn.Flatten(0, 1))

tr.add_model(model)
tr.set_criterion(nn.MSELoss)
tr.set_optimizer(optim.Adam)

for epoch in range(tr.epochs):
    tr.model.train()
    for batch in tr.data.data_loader:
        users, items, labels = [tens.to(tr.world.device) for tens in batch]
        outputs = tr.model.forward(users, items)
        loss = tr.criterion(outputs, labels)
        tr.backward(loss)
        tr.criterion.step()
    pq.io.save_state_dict(tr.model, tr.args, epoch)
````

## QuikData

A little more about how the data pull works. I usually run my project from my repo, and from the command line. Either way, `QuikData` expects your tensors, models, and state_dicts to be in your current path, and in a `data` subfolder. It will also expect to use your traveler's arguments to determine the date for the filename. You can set it using argparse like I do, or throw it in to your script like this:
``` python
tr = pq.travel.QuikTraveler()
tr.args.data_date = '20210101'
tr.add_data(pq.io.load_torch_object("train", tr.args))
```
The file will also have the words tensor, model, or state_dict in it, train/valid/test if it's a tensor, and e* if it's a state_dict, where e* stands for which epoch the state_dict was saved in. Therefore, the file structure using your repo as the base directory may look like this (I also save my original state_dict to make sure my model is actually training):
- data/train_tensor_20210101.pt
- data/valid_tensor_20210101.pt
- data/test_tensor_20210101.pt
- data/model_20210101.pt
- data/state_dict_eorig_20210101.pt
- data/state_dict_e0_20210101.pt
- data/state_dict_e1_20210101.pt

## Model and Functions

### Model State

Sometimes your training and validation losses will converge sooner than expected, and you'll want to test an epoch before the final one. this is possible, because the `pq.io.save_state_dict` function will save the weights and biases at the end of the epoch to disk.

### Set_* Functions
Setting the criterion, optimizer, and scheduler just takes a callback, and can use both general defaults and specific ones. For instance, I have an OptKwargs class that can receive parameters via argparse that most optimizers have (lr, weight_decay, eps, and betas), but then you can also feed in specific parameters like `amsgrad=False` if you are using Adam at instantiation like this: `tr.set_optimizer(Adam, amsgrad=False)`. For simplicity I didn't use a scheduler above, but you could include something like `tr.set_scheduler(OneCycleLR)`, and then after your backward, include a `tr.scheduler.step()`.

## Distributed Data Parallel (DDP)

### Intro to DDP

This is really why pytorch-quik is quick-er for me. Adding in DDP can be tough, and I tried to do so and allow you to switch back and forth when necessary. I run my code on an OpenShift cluster, and sometimes can't get on my multi-GPU setup. This allows me to just use a different set of args and just deal with slower code, not broken code!

I would suggest spending time setting up argparse so that you can have your own default arguments for batch size, learning rate, etc, but if you don't want to, you deal with my defaults. These assume you have a GPU on 1 node, which is the simplest benefit from pytorch-quik:

``` python
from argparse import ArgumentParser
parser = pq.args.add_learn_args(ArgumentParser())
args = parser.parse_args()
gpu = 0
tr = pq.travel.QuikTraveler(args, gpu)
tr.run_prep()
```

Notice the addition of providing the QuikTraveler your args, as well as telling it to run on your GPU 0. If you were truly distributing this across GPUs, you'd have to spawn QuikTravelers on each GPU, but more on that later. Also, the tr.run_prep() will start your DDP process_group.

### Automated Mixed Precision (AMP)

Tangentally related is AMP, and if your model.forward() is already set up with mixed precision, this should work for you also. Just add `args.mixed_precision = True` before creating your traveler, and add `with tr.amp.caster` before and within your forward like so (you will have to change your `with autocast():` to be a `with myparam` where myparam is what we're sending in here:
``` python
with tr.amp.caster:
  outputs = tr.model.forward(users, items, tr.amp.caster)
  loss = tr.criterion(outputs, labels)
```
Here is my .forward:
``` python
def forward(self, users, items, caster):
  with caster:
```

### Logging

This isn't true logging (though there is some), but this is a good place to talk about the progress bar and metrics. If your training/validation is distributed, your loss will eventually be aggregated with ring-all reduce on each GPU, so it doesn't matter on which one you calculate loss. So, I make your GPU 0 your "logger" GPU, and it will be the one with `tr.world.is_logger = True`. When you see this used, it's just telling the progress bar to be drawn, or the metrics to be calculated.

#### tqdm Progress Bar

One benefit of this is the [tqdm](https://pypi.org/project/tqdm/) progress bar. I didn't bother with the notebook one so that it can work distributed across GPUs, with IPython, or in a Jupyter Notebook. It should look something like this:

```
epoch: 1/2:  22%|████████████▏                                          | 1020/4591 [00:20<01:13, 48.69it/s]
```

I take care of running it only on your first GPU for you, and calculate your steps based on how your bunches distribute across samples. So, your initialization looks like this: 

``` python
trtq = pq.ddp.tq_bar(tr.data.steps, epoch, tr.epochs)
```

and then your update and close are as they would normally be with bar. I also suggest you `del` your bar because it can have trouble drawing the next one if you don't.

#### Metrics

One cool addition that I pulled from fast.ai (thanks fast.ai!) Is the way that they pull training loss and validation loss as they are training, and show them at the end of each epoch so you can see your progress. It works everywhere because it's just a pandas df, and it passes between training and validation because it's a class attribute. (Here we're only using training, so there's zeros). After the backward, you just need a `tr.metrics.add_loss(loss)` to accumulate it, and then after your loop a `tr.metrics.write()` to build the DataFrame. Then you can access it whenever you want at `tr.metrics.results`:

``` python
print(tr.metrics.results)
   epoch    train_loss  valid_loss   time
0      0        0.6435         0.0  05:03
1      1        0.5281         0.0  06:20
```

## Attributes

Pytorch-quik class instance attributes come in as arguments, and are (mostly) stored in dataclasses.

### Arguments

There are tons of arguments to set when you're training your model. These are split into various types:

- DDP args: (node rank, nodes, gpus, number of workers)
- learning args: (epochs, mixed precision, batch size, learning rate, weight decay, betas, epc, and find_unused_paramters)
- ray tune args: (use_ray_tune, number of samples)
- MLflow args: (use_mlflow, experiment, user, tracking URI, endpoint URL)

There are four ways to set arguments with pytorch-quik:

- Use the defaults: `pq.arg.add_learn_args(parser)`
- Provide them at the command line: `--learning_rate 1.5e-6`
- Defaulted within your script: `pq.arg.add_learn_args(parser, {"learning_rate": 1.5e-6)`
- And of course, fix them afterward (but check their name in the Namespace: `args.lr = 1.5e-6`

Putting all these things together, a simple setup would be:
``` python
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser = pq.arg.add_ddp_args(parser)
parser = pq.arg.add_learn_args(parser)
parser = pq.arg.add_mlflow_args(parser)
parser = pq.arg.add_ray_tune_args(parser)
args = parser.parse_args()
```

### Dataclasses

Similar to arguments, I've organized the attributes within a `QuikTrek` as three dataclasses and `QuikMlflow` as one dataclass containing keyword arguments (kwargs):
- `DlKwargs`: Data Loader kwargs such as batch size, pin memory, and number of workers
- `OptKwargs`: Optimizer kwargs such as learning rate, weight decay, and betas (these attempt to use universal names like `lr`)
- `World`: This is playing off the distributed kwarg of `world_size`, and is really anything about the world, such as nodes, gpus, init group, and ray tune
- `MlfKwargs`: MLflow kwargs such as tracking URI, endpoint URL and is_parent

## Usage Summary

Putting this all together, moving the argument creation into a `main`, and moving the rest into a function (let's call it `train`), which takes just two parameters, `args` and `gpu`. This would be my full example using all the above:

``` python
import pytorch_quik as pq
from torch import nn, optim
from argparse import ArgumentParser

def train(args, gpu):
    trek = pq.travel.QuikTrek(args, gpu)
    tr = pq.travel.QuikTraveler(trek)
    tr.run_prep()

    tr.args.data_date = '20210101'
    tr.add_data(pq.io.load_torch_object("train", tr.args))

    # from pytorch.org/tutorials/beginner/pytorch_with_examples.html
    model = nn.Sequential(nn.Linear(3, 1), nn.Flatten(0, 1))

    tr.add_model(model)
    tr.set_criterion(nn.MSELoss)
    tr.set_optimizer(optim.Adam)
    tr.set_scheduler(OneCycleLR, steps_per_epoch=tr.data.steps)
    if tr.world.is_logger:
        tr.io.save_state_dict(tr.model, tr.args, "orig")

    for epoch in range(tr.epochs):
        if tr.world.is_logger:
            trbar = pq.ddp.tq_bar(tr.data.steps, epoch, tr.epochs)
        tr.model.train()
        for batch in tr.data.data_loader:
            users, items, labels = [tens.to(tr.world.device) for tens in batch]
            with tr.amp.caster:
                outputs = tr.model.forward(users, items, tr.amp.caster)
                loss = tr.criterion(outputs, labels)
            tr.backward(loss)
            tr.scheduler.step()
            tr.metrics.add_loss(loss)
            if tr.world.is_logger:
                trbar.update()
        if tr.world.is_ddp:
            dist.barrier()
        if tr.world.is_logger:
            trbar.close()
            del trbar
            pq.io.save_state_dict(tr.model, tr.args, epoch)
    tr.metrics.write()
    if tr.world.is_logger:
        print(tr.metrics.results)
    if tr.world.is_ddp:
        pq.ddp.cleanup()


def main():
    parser = pq.args.add_learn_args(ArgumentParser())
    args = parser.parse_args()
    gpu = 0
    train(args, gpu)


if __name__ == "__main__":
    main()

````

## Additional Usage Information

### Spawning across GPUs... or not!

This is why I love pytorch-quik.  then you can use my `traverse` function that will make sure your GPUs are available to PyTorch, and if so, send your training across each GPU! If it decides you can't, it send use that same exact code to the CPU run it on there. This is what they would all look like (I'll abbreviate train):

``` python
import pytorch_quik as pq
from torch import nn, optim
from argparse import ArgumentParser

def train(args, gpu):
...

def main():
    parser = pq.args.add_learn_args(ArgumentParser())
    args = parser.parse_args()
    args.gpus = 4
    pq.ddp.traverse(train, args)


if __name__ == "__main__":
    main()
```

### Running from command line

Now, you can package your repo, not even touch the code, and run it on four GPUs like `python main.py --gpus 4`, on one gpu when there are four like `python main.py --gpus 1`, or throw it on a cluster, let the cluster decide where to run it, and when you run `python main.py`, pytorch-quik will _determine if you have GPUs and whether it should distribute or not!_

Cool, huh.

Let me know if you have any questions, and I'll keep adding to this documentation!

# Integrations

## MLflow integration

#### Instantiation

MLflow for model tracking has been integrated with `QuikTrek`. When instantiated, if `use_mlflow=True`, then a `QuikFlow` is instantiated, as well as a run, which involves:
- checking if `args.experiment` exists, and if not, creates it
- creates a run under the experiment
- adds appropriate run tags
- adds data loader, optimizer, and world [dataclasses](#dataclasses) as run parameters

#### Tracking

Next, as training occurs, if the `QuikTraveler` `add_loss` function is used, the "train_loss" metric will be added to MLFlow after _every step_, and the "valid_loss" will be added after every epoch. It's best to use step, since this will align the points on the MLflow graph based on training process, not clock time.

#### Saving artifacts and accuracy metrics

In addition, artifacts are saved when the following `QuikTraveler`:
- `save_state_dict` now saves each epoch's state dict as an artifact in the run
- `record_results`: records the classification report results as metrics, and saves the `png` of the confusion matrix as an artifact

## Ray Tune integration

As it's difficult to manage the execution (and tracking) of hundreds of training runs, pytorch-quik attempted to use both [Optuna](https://optuna.org/) and [Ray Tune](https://docs.ray.io/en/ray-0.4.0/tune.html). Both were successful integrations, but optuna lacked the ability to work with DDP. If there are multiple GPUs on your platform, then ray tune will execute a run on each GPU. When `use_ray_tune=True`, the following options are available:
- Creating a parent (with `is_parent=True`) in MLflow, to contain all runs in a collapsible parent run
- Using `get_tune_config` to pull the hyperparameter options from a yaml file, such as:
    ``` yaml
    bs:
        choice:
            categories: [8, 12, 16, 24]
    weight_decay:
        choice:
            categories: [0, 0.001]
    ```
- using `run_ddp_tune` to have all test permutations be distributed across GPUs

## Huggingface integration

As many of my models are transformers, the following options are available (only yet tested on BERT and RoBERTa models:
- `get_bert_info`: using `arg.bert_type`, will pull model information for a sequence classification model
- `get_pretrained_model`: using `arg.bert_type`, will pull the model itself
- `get_encodings`: using `arg.bert_type`, will pull the model's tokenizer and encode text
- `save_bert_model`: pull a state dict, and add it to the model before using transformer's `save_pretrained`
- `save_tokenizer`: save the tokenizer, and _also update the tokenizer.json_ to include the correct pathing for serving


## Torch Serve integration

### TS Functionality

Finally, in order to productionalize a pytorch model, [TorchServe](https://github.com/pytorch/serve) is an excellent choice. There is quite a bit of overhead to get it up and running, so I've provided some helpful functions. The main ones to note are:
- `build_extra_files`: There are many files needed to build a model archive. This function will build:
    - `setup_config.json`
    - `index_to_name.json`
    - `sample_text.json`
    - A default transformer handler if you don't have one (`transformer_handler_pq`)
- `create_mar`: First checks to see if you have all the required files, then uses the command line function to create your torch model archive.

I should note the benefit of using my customized handler, is that it will provide the predictions like Huggingface's example, but will _also provide the logits_ as required by my stakeholder.

### TS Usage

In order to create your torch model archive (mar) file, there are only a few steps. When I do so, I pull my [state dict from MLflow](#saving-artifacts-and-accuracy-metrics), which makes it even more modular. If you create a function called `parse_my_args()` including all the `pq.args` in the [Arguments snippet](#arguments), and `INDEX_LABELS` is an `OrderedDict` of your labels, here's an example where you (1) pull the state dict, (2) download the model, (3) save the model and files to disk, and (4) create the mar:

``` python
        args = parse_my_args() # this would be all pq.arg functions you'd need
        mlf = pq.mlflow.QuikMlflow(args)
        serve_path = pq.io.id_str("", args).parent.joinpath("serve")
        filter_str = "params.lr = '1.6e-06'" # some filter to differentiate the specific run for your state dict
        args.data_date = mlf.get_state_dict(args.test_epoch, filter_string=filter_str, serve_path=serve_path)
        model = pq.bert.get_pretrained_model(labels=list(INDEX_LABELS.keys()), bert_type=args.bert_type)
        pq.bert.save_bert_model(model, args, args.test_epoch, serve_path)
        pq.serve.build_extra_files(args, INDEX_LABELS, serve_path)
        pq.serve.create_mar(args, serve_path)
```

### Torch Serve API Usage

Now that you have a torch serve mar (let's call it `my_model.mar`, and have started it (hopefully in a container!) by running `torchserve --start --models "my_tc=my_model.mar" --ncs`, then you can use pytorch-quik's `batch_inference` function. The benefits of this function are:
- it takes an array of records, vs. sending one input at a time to the API!
- formats the text as the handler expects (json with an "inference" header and a "data" key for each record)
- batches the API calls so as to not overload the API (I usually use a `batch_size` of 20-35 records)
- opens a session with the API, and retries whenever you receive a 507 error (which is common for me)
- normalizes the response and drops it into a pandas dataframe.

Usage is pretty easy. If your test set is from sklearn train test split, then this should work (`INDEX_LABELS` is an `OrderedDict` of your labels):
``` python
df = pq.api.batch_inference(Xte[:, 0], INDEX_LABELS, batch_size=20)
```

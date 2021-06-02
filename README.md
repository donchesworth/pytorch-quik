# pytorch-quik
![example workflow](https://github.com/donchesworth/pytorch-quik/actions/workflows/github-ci.yml/badge.svg)
[![](https://img.shields.io/pypi/v/pytorch-quik.svg)](https://pypi.org/pypi/name/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Docker Repository on Quay](https://quay.io/repository/donchesworth/rapids-dask-pytorch/status "Docker Repository on Quay")](https://quay.io/repository/donchesworth/rapids-dask-pytorch)
[![code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![codecov](https://codecov.io/gh/donchesworth/pytorch-quik/branch/main/graph/badge.svg?token=U92M8C8AFM)](https://codecov.io/gh/donchesworth/pytorch-quik)

## For distributing (or not) your neural nets in PyTorch quik-er

As I was building out the same set of code for a recommender system, a BERT sentiment model, and a co-worker was about to build a classification model, I decided to standardize the code into this package. It's lightweight, because I didn't want to hide the standard steps to keep the user from learning Neural Networks, but I also didn't want to maintain the code in multiple places.

### Installation

``` bash
pip install pytorch-quik
```

## Usage

### Intro

In its simplest form, you'll want to:

- create a QuikTraveler (an object that "travels" forward and backward on your neural network)
- add a QuikData
- add a criterion, optimizer, etc.
- set up your loop

``` python
import pytorch_quik as pq
from torch import nn
from torch.optim import Adam

qt = pq.travel.QuikTraveler()
qt.add_data(pq.io.load_torch_object("train", qt.args))

# from pytorch.org/tutorials/beginner/pytorch_with_examples.html
model = nn.Sequential(nn.Linear(3, 1), nn.Flatten(0, 1))

qt.add_model(model)
qt.set_criterion(nn.MSELoss)
qt.set_optimizer(Adam)

for epoch in range(qt.epochs):
    trbar = pq.ddp.tq_bar(qt.data.steps, epoch, qt.epochs)
    qt.model.train()
    for batch in qt.data.data_loader:
        users, items, labels = [tens.to(qt.world.device) for tens in batch]
        outputs = qt.model.forward(users, items)
        loss = qt.criterion(outputs, labels)
        qt.backward(loss)
        qt.criterion.step()
        trbar.update()
    trbar.close()
    del trbar
    pq.io.save_state_dict(qt.model, qt.args, epoch)
````

### Progress Bar

One benefit of this is the progress bar using this distributed across GPUs, with IPython, or in a Jupyter Notebook. It should look something like this:

```
epoch: 1/2:  22%|████████████▏                                          | 1020/4591 [00:20<01:13, 48.69it/s]
```

### Model State

Sometimes your training and validation losses will converge sooner than expected, and you'll want to test an epoch before the final one. this is possible, because the `pq.io.save_state_dict` function will save the weights and biases at the end of the epoch to disk.

### Set* Functions
Setting the criterion, optimizer, and scheduler just takes a callback, and can use both general defaults and specific ones. For instance, I have an OptKwargs class that can receive parameters via argparse that most optimizers have (lr, weight_decay, eps, and betas), but then you can also feed in specific parameters like `amsgrad=False` if you are using Adam at instantiation like this: `qt.set_optimizer(Adam, amsgrad=False)`. For simplicity I didn't use set_scheduler above, but you could include something like `qt.set_scheduler(OneCycleLR)`, and then after your backward, include a `tr.scheduler.step()`.

## Distributed Data Parallel (DDP)

This is why pytorch-quik is quick-er for me. Adding in DDP can be tough, and I tried to do so and allow you to switch back and forth when necessary. I run my code on an OpenShift cluster, and sometimes can't get on my multi-GPU setup. This allows me to just use a different set of args and just deal with slower code, not broken code!

I would suggest spending time setting up argparse so that you can have your own default arguments for batch size, learning rate, etc, but if you don't want to, you deal with my defaults. These assume you have a GPUs on 1 node, which is the simplest benefit from pytorch-quik:

``` python
from argparse import ArgumentParser
parser = pq.args.add_learn_args(ArgumentParser())
args = parser.parse_args()
qt = pq.travel.QuikTraveler(args, 0)
qt.run_prep()
```

Notice the addition of providing the QuikTraveler your args, as well as telling it to run on your GPU 0. If you were truly distributing this across GPUs, you'd have to spawn QuikTravelers on each GPU, but more on that later. Also, the qt.run_prep() will start your DDP process_group.

### Automated Mixed Precision (AMP)

Tangentally related is AMP, and if your model.forward() is already set up with mixed precision, this should work for you also. Just add `args.mixed_precision = True` before creating your traveler, and add `with tr.amp.caster` before and within your forward like so (you will have to change your `with autocast():` to be a `with myparam` where myparam is what we're sending in here:
``` python
with qt.amp.caster:
  outputs = tr.model.forward(users, items, qt.amp.caster)
  loss = tr.criterion(outputs, labels)
```
Here is my .forward:
``` python
def forward(self, users: torch.LongTensor, items: torch.LongTensor, caster):
  with caster:
```

Let me know if you have any questions, and I'll keep adding to this documentation!

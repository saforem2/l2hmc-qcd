# l2hmc-qcd

[![CodeFactor](https://www.codefactor.io/repository/github/saforem2/l2hmc-qcd/badge)](https://www.codefactor.io/repository/github/saforem2/l2hmc-qcd)

Application of the L2HMC algorithm to simulations in lattice QCD. A description
of the L2HMC algorithm can be found in the paper:

[*Generalizing Hamiltonian Monte Carlo with Neural Network*](https://arxiv.org/abs/1711.09268)

by [Daniel Levy](http://ai.stanford.edu/~danilevy), [Matt D. Hoffman](http://matthewdhoffman.com/) and [Jascha Sohl-Dickstein](sohldickstein.com)

---

## Overview

**NOTE**: There are compatibility issues with `tensorflow.__version__ > 1.12` To be sure everything runs correctly, make sure `tensorflow==1.12.x` is installed.

Given an analytically described distributions (simple examples can be found in
`l2hmc-qcd/utils/distributions.py`), L2HMC enables training of fast-mixing samplers.

## Modified implementation for Lattice Gauge Theory / Lattice QCD models. 

This work is based on the original implementation which can be found at
[brain-research/l2hmc/](https://github.com/brain-research/l2hmc). 

My current focus is on applying this algorithm to simulations in lattice gauge
theory and lattice QCD, in hopes of obtaining greater efficiency compared to
generic HMC.

This new implementation includes the algorithm as applied to the $2D$ $U{(1)}$ lattice gauge theory model (i.e. compact QED).

Additionally, this implementation includes a convolutional neural network
architecture that is prepended to the network described in the original paper.
The purpose of this additional structure is to better incorporate information
about the geometry of the lattice.

Lattice code can be found in `l2hmc-qcd/lattice/` and the particular code for the
$2D$ $U{(1)}$ lattice gauge model can be found in `l2hmc-qcd/lattice/lattice.py`.

## Features

This model can be trained using distributed training through [`horovod`](https://github.com/horovod/horovod), by passing the `--horovod` flag as a command line argument. 

---

# Organization

## Training

Example command line arguments can be found in `l2hmc-qcd/args`. The module `l2hmc-qcd/main.py` implements wrapper functions that are used to train the model and save the resulting trained graph which can then be loaded and used for inference.

The code responsible for actually training the model can be found in the
`GaugeModelTrainer` object inside the `l2hmc-qcd/trainers/trainer.py` module.

Summary objects for monitoring model performance in TensorBoard are created in the various methods found in `l2hmc-qcd/loggers/summary_utils.py`. These objects are then created inside the `create_summaries(...)` method of the `TrainLogger` class (defined in `l2hmc-qcd/loggers/train_logger.py`).

To train the model, you can either specify command line arguments manually
(descriptions can be found in `utils/parse_args.py`), or use the
`args/args.txt` file, which can be passed directly to `main.py`.

For example, from within the `l2hmc-qcd/args` directory:
```
python3 ../main.py @args.txt
```

All of the relevant command line options are well documented and can be found in `l2hmc-qcd/utils/parse_args.py`. Almost all relevant information about different parameters and run options can be found in this file.

## Inference

Once the training is complete, we can use the trained model to run inference
to gather statistics about relevant lattice observables. This can be done using
the `inference.py` module which implements helper functions for loading and
running the saved model.

Explicitly, assuming we called the `main.py` module from within the 
`l2hmc-qcd/args` directory using the command given above, we can then run
inference via:

```
python ../inference.py --run_steps 5000
```

where `--run_steps` indicates the number of complete MD updates to be performed
(i.e. augmented leapfrog integrator followed by Metroplis-Hastings
accept/reject).

## Model

Model information (including the implementation of the loss function) can be found in `l2hmc-qcd/models/model.py`. This module implements the `GaugeModel` class, which is responsible for building the main tensorflow graph and creating all the relevant tensorflow operations for training and running the L2HMC sampler.

## Dynamics / Network

The augmented L2HMC leapfrog integrator is implemented using the
`GaugeDynamics` object which is located in the `l2hmc-qcd/dynamics/dynamics.py`
module.

The `GaugeDynamics` class has a `build_network` method that builds the neural
network. The network architecture is specified via the `--network_arch` command
line flag, with possible values being: `generic`, `conv2D`, or `conv3D`.

Specific details about the network can be found in
`l2hcm-qcd/network/network.py`. Due to the unconventional architecture
and data-flow of the L2HMC algorithm, the network is implemented by
subclassing the `tf.keras.Model`, which is a sufficiently flexible approach.

## Notebooks
`l2hmc-qcd/notebooks/` contains a random collection of jupyter notebooks that each serve different purposes and should be somewhat self explanatory.

---

## Contact

***Code author:*** Sam Foreman

***Pull requests and issues should be directed to:*** [saforem2](http://github.com/saforem2)

## Citation

If you use this code, please cite the original paper:
```
@article{levy2017generalizing,
  title={Generalizing Hamiltonian Monte Carlo with Neural Networks},
  author={Levy, Daniel and Hoffman, Matthew D. and Sohl-Dickstein, Jascha},
  journal={arXiv preprint arXiv:1711.09268},
  year={2017}
}
```

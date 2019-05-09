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

## Organization

Example command line arguments can be found in `l2hmc-qcd/args`. To run `l2hmc-qcd/gauge_model_main.py` using one of the `.txt` files found in `l2hmc-qcd/args`, simply pass the `*.txt` file as the only command line argument prepended with `@`. 

For example, from within the `l2hmc-qcd/args` directory:
```
python3 ../gauge_model_main.py @gauge_model_args.txt
```

All of the relevant command line options are well documented and can be found in `l2hmc-qcd/utils/parse_args.py`. Almost all relevant information about different parameters and run options can be found in this file.

Model information can be found in `l2hmc-qcd/models/gauge_model.py` which is responsible for building the graph and creating all the relevant tensorflow operations for training and running the L2HMC sampler.

The code responsible for actually implementing the L2HMC algorithm is dividied up between `l2hmc-qcd/dynamics/gauge_dynamics.py` and `l2hmc-qcd/network/`.

The code responsible for performing the augmented leapfrog algorithm is implemented in  the `GaugeDynamics` class defined in `l2hmc-qcd/dynamics/gauge_dynamics.py`.

There are multiple different neural network architectures defined in `l2hmc-qcd/network/` and different architectures can be specified as command line arguments defined in `l2hmc-qcd/utils/parse_args.py`.

`l2hmc-qcd/notebooks/` contains a random collection of jupyter notebooks that each serve different purposes and should be somewhat self explanatory.


## Contact

***Code author:*** Sam Foreman

***Pull requests and issues with forked code:*** @saforem2

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

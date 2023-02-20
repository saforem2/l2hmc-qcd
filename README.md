<div align="center">

![l2hmc-qcd](https://github.com/saforem2/saforem2/blob/main/assets/l2hmc-qcd-small.svg)

<a href="https://hits.seeyoufarm.com"><img alt="hits" src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fsaforem2%2Fl2hmc-qcd&count_bg=%2300CCFF&title_bg=%23555555&icon=&icon_color=%23111111&title=👋&edge_flat=false"></a>    
<a href="https://github.com/saforem2/l2hmc-qcd/"><img alt="l2hmc-qcd" src="https://img.shields.io/badge/-l2hmc--qcd-252525?style=flat&logo=github&labelColor=gray"></a> <a href="https://www.codefactor.io/repository/github/saforem2/l2hmc-qcd"><img alt="codefactor" src="https://www.codefactor.io/repository/github/saforem2/l2hmc-qcd/badge"></a>
<br>
<a href="https://arxiv.org/abs/2112.01582"><img alt="arxiv" src="http://img.shields.io/badge/arXiv-2112.01582-B31B1B.svg"></a> <a href="https://arxiv.org/abs/2105.03418"><img alt="arxiv" src="http://img.shields.io/badge/arXiv-2105.03418-B31B1B.svg"></a> 
<br>
<a href="https://hydra.cc"><img alt="hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a> <a href="https://pytorch.org/get-started/locally/"><img alt="pyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a> <a href="https://www.tensorflow.org"><img alt="tensorflow" src="https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?&logo=TensorFlow&logoColor=white"></a> 
<br>
[<img src="https://raw.githubusercontent.com/wandb/assets/main/wandb-github-badge-28.svg" alt="Weights & Biases monitoring" height=20>](https://wandb.ai/l2hmc-qcd/l2hmc-qcd)

</div>

<details open><summary><b>Contents</b></summary>

- [Overview](#overview)
  * [Papers 📚, Slides 📊, etc.](https://github.com/saforem2/l2hmc-qcd/#training--experimenting)
  * [Background](#background)
- [Installation](#installation)
- [Training](#training)
  - [Configuration Management](#configuration-management)
  - [Running @ ALCF](#running-at-ALCF) 
- [Details](#details)
  * [Organization](#organization)
    + [Dynamics / Network](#dynamics---network)
      - [Network Architecture](#network-architecture)
    + [Lattice](#lattice)

</details>

# Overview

## Papers 📚, Slides 📊 etc.
- [📕 Notebooks](./src/l2hmc/notebooks/):
    - 📙 2D $U(1)$ Model (w/ `fp16` or `fp32` for training)
    	- [`src/l2hmc/notebooks/l2hmc-2dU1.ipynb`](./src/l2hmc/notebooks/l2hmc-2dU1.ipynb)
    	- [alt link (if Github won't load)](https://nbviewer.org/github/saforem2/l2hmc-qcd/blob/dev/src/l2hmc/notebooks/l2hmc-2dU1.ipynb)
    - 📒 4D $SU(3)$ Model (w/ `complex128` + `fp64` for training)
        - PyTorch:
            - [`src/l2hmc/notebooks/pytorch-SU3d4.ipynb`](./src/l2hmc/notebooks/l2hmc-2dU1.ipynb)
    	    - [alt link (if github won't load)](https://nbviewer.org/github/saforem2/l2hmc-qcd/blob/dev/src/l2hmc/notebooks/pytorch-SU3d4.ipynb)

- 📝 Papers:
    - [Accelerated Sampling Techniques for Lattice Gauge Theory](https://saforem2.github.io/l2hmc-dwq25/#/) @ [BNL & RBRC: DWQ @ 25](https://indico.bnl.gov/event/13576/) (12/2021)
    - [Training Topological Samplers for Lattice Gauge Theory](https://bit.ly/l2hmc-ect2021) from the [*ML for HEP, on and off the Lattice*](https://indico.ectstar.eu/event/77/) @ $\mathrm{ECT}^{*}$ Trento (09/2021) (+ 📊 [slides](https://www.bit.ly/l2hmc-ect2021))
    - [Deep Learning Hamiltonian Monte Carlo](https://arxiv.org/abs/2105.03418) @ [Deep Learning for Simulation (SimDL) Workshop](https://simdl.github.io/overview/) **ICLR 2021**
        - 📚 : [arXiv:2105.03418](https://arxiv.org/abs/2105.03418)  
        - 📊 : [poster](https://www.bit.ly/l2hmc_poster)


## Background
The L2HMC algorithm aims to improve upon
[HMC](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo) by optimizing a
carefully chosen loss function which is designed to minimize autocorrelations
within the Markov Chain, thereby improving the efficiency of the sampler.

A detailed description of the original L2HMC algorithm can be found in the paper:

[*Generalizing Hamiltonian Monte Carlo with Neural Network*](https://arxiv.org/abs/1711.09268)

with implementation available at
[brain-research/l2hmc/](https://github.com/brain-research/l2hmc) by [Daniel
Levy](http://ai.stanford.edu/~danilevy), [Matt D.
Hoffman](http://matthewdhoffman.com/) and [Jascha
Sohl-Dickstein](sohldickstein.com).

Broadly, given an *analytically* described target distribution, π(x), L2HMC provides a *statistically exact* sampler that:

- Quickly converges to the target distribution (fast ***burn-in***).
- Quickly produces uncorrelated samples (fast ***mixing***).
- Is able to efficiently mix between energy levels.
- Is capable of traversing low-density zones to mix between modes (often difficult for generic HMC).

# Installation

## From Source (recommended)


> **Warning**<br>
> It is recommended to install _inside_ an existing virtual environment<br>
> (ideally one with `tensorflow, pytorch [horovod,deepspeed]` already installed)


1. Clone + navigate into repo

    ```Shell
    git clone https://github.com/saforem2/l2hmc-qcd
    cd l2hmc-qcd
    ```

3. Test install

    ```Shell
    python3 -c 'import l2hmc ; print(l2hmc.__file__)'
    ```

## From PyPi

- [`l2hmc`](https://pypi.org/project/l2hmc/) on PyPi:

    ```Shell
    python3 -m pip install l2hmc
    ```

# Training

## Configuration Management

This project uses [`hydra`](https://hydra.cc) for configuration management and
supports distributed training for both PyTorch and TensorFlow.


The main entry point is [`src/l2hmc/main.py`](./src/l2hmc/main.py),
which contains  the logic for running an end-to-end `Experiment`.

An `Experiment` consists of the following sub-tasks:

1. Training
2. Evaluation
3. HMC (for comparison and to measure model improvement)


In particular, we support the following combinations of `framework` + `backend` for distributed training:

- TensorFlow (+ Horovod for distributed training)
- PyTorch +
    - DDP
    - Horovod
    - DeepSpeed

**All** configuration options can be dynamically overridden via the CLI at runtime, 
and we can specify our desired `framework` and `backend` combination via:

```Shell
python3 main.py mode=debug framework=pytorch backend=deepspeed precision=fp16
```

to run a (non-distributed) Experiment with `pytorch + deepspeed` with `fp16` precision.

The [`l2hmc/conf/config.yaml`](./src/l2hmc/conf/config.yaml) contains a brief
explanation of each of the various parameter options, and values can be
overriden either by modifying the `config.yaml` file, or directly through the
command line, e.g.

```Shell
cd src/l2hmc
./train.sh mode=debug framework=pytorch > train.log 2>&1 &
tail -f train.log $(tail -1 logs/latest)
```

Additional information about various configuration options can be found in:

- [`src/l2hmc/configs.py`](./src/l2hmc/configs.py):
  Contains implementations of the (concrete python objects) that are adjustable for our experiment.
- [`src/l2hmc/conf/config.yaml`](./src/l2hmc/conf/config.yaml):
  Starting point with default configuration options for a generic `Experiment`.


for more information on how this works I encourage you to read [Hydra's
Documentation Page](https://hydra.cc).


## Running at ALCF

For running with distributed training on ALCF systems, we provide a complete
[`src/l2hmc/train.sh`](./src/l2hmc/train.sh) 
script which should run without issues on either Polaris or ThetaGPU @ ALCF.


# Details

**Goal:** Use L2HMC to **efficiently** generate _gauge configurations_ for
calculating observables in lattice QCD.

A detailed description of the (ongoing) work to apply this algorithm to
simulations in lattice QCD (specifically, a 2D U(1) lattice gauge theory model)
can be found in [arXiv:2105.03418](https://arxiv.org/abs/2105.03418).

<div align="center">
 <img src="assets/l2hmc_poster.jpeg" alt="l2hmc-qcd poster" width="90%" />
</div>

## Organization

### Dynamics / Network

For a given target distribution, π(x), the `Dynamics` object
([`src/l2hmc/dynamics/`](src/l2hmc/dynamics)) implements methods for generating
proposal configurations (x' ~ π) using the generalized leapfrog update.


This generalized leapfrog update takes as input a buffer of lattice
configurations `x` and generates a proposal configuration `x' = Dynamics(x)` by
evolving generalized L2HMC dynamics.


#### Network Architecture

An illustration of the `leapfrog layer` updating `(x, v) --> (x', v')` can be seen below.

<div align="center">
 <img src="assets/lflayer.png" alt="leapfrog layer" width=800/>
</div>


## Contact

***Code author:*** Sam Foreman

***Pull requests and issues should be directed to:*** [saforem2](http://github.com/saforem2)

## Citation

If you use this code or found this work interesting, please cite our work along with the original paper:

```bibtex
@misc{foreman2021deep,
      title={Deep Learning Hamiltonian Monte Carlo}, 
      author={Sam Foreman and Xiao-Yong Jin and James C. Osborn},
      year={2021},
      eprint={2105.03418},
      archivePrefix={arXiv},
      primaryClass={hep-lat}
}
```

```bibtex
@article{levy2017generalizing,
  title={Generalizing Hamiltonian Monte Carlo with Neural Networks},
  author={Levy, Daniel and Hoffman, Matthew D. and Sohl-Dickstein, Jascha},
  journal={arXiv preprint arXiv:1711.09268},
  year={2017}
}
```

## Acknowledgement


> **Note**<br>
> This research used resources of the Argonne Leadership Computing Facility, which is a DOE Office of Science User Facility supported under contract DE_AC02-06CH11357.<br>
> This work describes objective technical results and analysis.<br>
> Any subjective views or opinions that might be expressed in the work do not necessarily represent the views of the U.S. DOE or the United States Government.

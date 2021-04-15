# l2hmc-qcd  [![CodeFactor](https://www.codefactor.io/repository/github/saforem2/l2hmc-qcd/badge)](https://www.codefactor.io/repository/github/saforem2/l2hmc-qcd) [![HitCount](http://hits.dwyl.com/saforem2/l2hmc-qcd.svg)](http://hits.dwyl.com/saforem2/l2hmc-qcd)

---

**Note:** An end-to-end training + inference example can be found in
[this notebook](l2hmc-qcd/notebooks/complete_example_2021_01_26.ipynb).

## Overview

The L2HMC algorithm aims to improve upon [HMC](https://en.wikipedia.org/wiki/Hamiltonian_Monte_Carlo)
by optimizing a carefully chosen loss function which is designed to minimize autocorrelations
within the Markov Chain, thereby improving the efficiency of the sampler.

This work is based on the original implementation: [brain-research/l2hmc/](https://github.com/brain-research/l2hmc).

A detailed description of the L2HMC algorithm can be found in the paper:

[*Generalizing Hamiltonian Monte Carlo with Neural Network*](https://arxiv.org/abs/1711.09268)

by [Daniel Levy](http://ai.stanford.edu/~danilevy), [Matt D. Hoffman](http://matthewdhoffman.com/) and [Jascha Sohl-Dickstein](sohldickstein.com).

Broadly, given an *analytically* described target distribution, π(x), L2HMC provides a *statistically exact* sampler that:

- Quickly converges to the target distribution (fast ***burn-in***).
- Quickly produces uncorrelated samples (fast ***mixing***).
- Is able to efficiently mix between energy levels.
- Is capable of traversing low-density zones to mix between modes (often difficult for generic HMC).


## L2HMC for LatticeQCD

**Goal:** Use L2HMC to **efficiently** generate _gauge configurations_ for calculating observables in lattice QCD.

A detailed description of the (ongoing) work to apply this algorithm to simulations in 
lattice QCD (specifically, a 2D U(1) lattice gauge theory model) can be found in [`doc/main.pdf`](doc/main.pdf).

<div align="center">
 <img src="assets/l2hmc_poster.jpeg" alt="l2hmc-qcd poster"/>
</div>

## Organization

### Dynamics / Network

The base class for the augmented L2HMC leapfrog integrator is implemented in the [`BaseDynamics`](l2hmc-qcd/dynamics/base_dynamics.py) (a `tf.keras.Model` object).

The [`GaugeDynamics`](l2hmc-qcd/dynamics/gauge_dynamics.py) is a subclass of `BaseDynamics` containing modifications for the 2D U(1) pure gauge theory.

The network is defined in [` l2hmc-qcd/network/functional_net.py`](l2hmc-qcd/network/functional_net.py).


#### Network Architecture

An illustration of the `leapfrog layer` updating `(x, v) --> (x', v')` can be seen below.

<div align="center">
 <img src="assets/lflayer.png" alt="leapfrog layer" width=800/>
</div>

<!---The network takes as input the position `x`, momentum `v` and and outputs the quantities `sx, tx, qx`, which are then used in the augmented Hamiltonian dynamics to update `x`.--->

<!---Similarly, the network used for updating the momentum variable `v` has an identical architecture, taking as inputs the position `x`, the gradient of the potential, `dUdX`, and the same fictitious time `t`, and outputs the quantities `sv, tv, qv` which are then used to update `v`.--->

<!---**Note:** In the image above, the quantities `x', v''` represent the outputs of a Dense layer followed by a `ReLu` nonlinearity.--->

### Lattice

Lattice code can be found in [`lattice.py`](l2hmc-qcd/lattice/lattice.py), specifically the `GaugeLattice` object that provides the base structure on which our target distribution exists.

Additionally, the `GaugeLattice` object implements a variety of methods for calculating physical observables such as the average plaquette, ɸₚ, and the topological charge Q,

### Training

The training loop is implemented in [`l2hmc-qcd/utils/training_utils.py `](l2hmc-qcd/utils/training_utils.py).

To train the sampler on a 2D U(1) gauge model using the parameters specified in [` bin/train_configs.json`](bin/train_configs.json):

```bash
$ python3 /path/to/l2hmc-qcd/l2hmc-qcd/train.py --json_file=/path/to/l2hmc-qcd/bin/train_configs.json
```

Or via the [` bin/train.sh `](bin/train.sh) script provided in [` bin/ `](bin/).

## Features

- **Distributed training**
  (via [`horovod`](https://github.com/horovod/horovod)): If `horovod` is installed, the model can be trained across multiple GPUs (or CPUs) by:

  ```bash
  #!/bin/bash
  
  TRAINER=/path/to/l2hmc-qcd/l2hmc-qcd/train.py
  JSON_FILE=/path/to/l2hmc-qcd/bin/train_configs.json
  
  horovodrun -np ${PROCS} python3 ${TRAINER} --json_file=${JSON_FILE}
  ```

## Contact
---
***Code author:*** Sam Foreman

***Pull requests and issues should be directed to:*** [saforem2](http://github.com/saforem2)

## Citation

If you use this code, please cite the original paper:
```bibtex
@article{levy2017generalizing,
  title={Generalizing Hamiltonian Monte Carlo with Neural Networks},
  author={Levy, Daniel and Hoffman, Matthew D. and Sohl-Dickstein, Jascha},
  journal={arXiv preprint arXiv:1711.09268},
  year={2017}
}
```

## Acknowledgement

<!---<div align="center">
 <img src="assets/anl.png" alt="Argonne National Laboratory Icon" width=500/>
</div>!--->
This research used resources of the Argonne Leadership Computing Facility, which is a DOE Office of Science User Facility supported under contract DE_AC02-06CH11357.  This work describes objective technical results and analysis. Any subjective views or opinions that might be expressed in the work do not necessarily represent the views of the U.S. DOE or the United States
Government. Declaration of Interests - None.


## Stargazers over time

[![Stargazers over time](https://starchart.cc/saforem2/l2hmc-qcd.svg)](https://starchart.cc/saforem2/l2hmc-qcd)

# l2hmc-qcd  [![CodeFactor](https://www.codefactor.io/repository/github/saforem2/l2hmc-qcd/badge)](https://www.codefactor.io/repository/github/saforem2/l2hmc-qcd) ![HitCount](http://hits.dwyl.io/saforem2/l2hmc-qcd.svg)

**Update (06/17/2020):** Major rewrite of previous code, new implementations are compatible with `tensorflow >= 2.2` and are capable of being run both imperatively ('eager execution', the default in `tensorflow 2.x`), and through graph execution, obtained by compiling the necessary functions with `tf.function`. 


A description of the L2HMC algorithm can be found in the paper:

[*Generalizing Hamiltonian Monte Carlo with Neural Network*](https://arxiv.org/abs/1711.09268)

by [Daniel Levy](http://ai.stanford.edu/~danilevy), [Matt D. Hoffman](http://matthewdhoffman.com/) and [Jascha Sohl-Dickstein](sohldickstein.com)


# Overview

We are interested in applying the L2HMC algorithm to generate *gauge configurations* for LatticeQCD.    

This work is based on the original implementation which can be found at [brain-research/l2hmc/](https://github.com/brain-research/l2hmc). 



Given an *analytically* described target distribution, $\pi(x)$, L2HMC provides a *statistically exact* sampler that:

- Quickly converges to the target distribution (fast ***burn-in***).
- Quickly produces uncorrelated samples (fast ***mixing***).
- Is able to efficiently mix between energy levels.
- Is capable of traversing low-density zones to mix between modes (often difficult for generic HMC).


# L2HMC for LatticeQCD

**Goal:** Use L2HMC to generate _gauge configurations_ for lattice QCD. 

All lattice QCD simulations are performed at finit lattice spacing $a$, and need an extrapolation to the continuum in order to be used for computing values of physical quantities.
More reliable extrapolations can be done by simulating the theory at increasingly smaller lattice spacings.

The picture that results when the lattice spacing is reduced and the physics kept constant is that all finite physical quantities of negative mass dimension diverge if measured in lattice units.

In statistical mechanics language, this states that the continuum limit is a critical point of the theory since the correlation lengths diverge.
MCMC algorithms are known to encounter difficulties when used for simulating theories close to a critical point, an issue known as the _critical slowing down_ of the algorithm.

**The L2HMC algorithm aims to improve upon HMC by optimizing a carefully chosen loss function which is designed to minimize autocorrelations within the Markov Chain.**

In doing so, the overall efficiency of the simulation is subsequently improved.

## $U(1)$ Lattice Gauge Theory

We start by considering the simpler $(1+1)$-dimensional $U(1)$ lattice gauge
theory, defined on an $N_{x} \times N_{t}$ lattice with periodic boundary
conditions.

The action of this gauge theory is defined in terms of the *link variables*

<div align="center">
 <img src="assets/link_var.svg" alt="U_{\mu}(i) = e^{i\phi_{\mu}(i)}, \quad \phi_{\mu}(i) \in [0, 2\pi)"/>
</div>

and can be written as

<div align="center">
 <img src="assets/action1.svg" alt="S = \sum_{P}\, 1 - \cos(\phi_{P})"/>
</div>

&nbsp;

where $\phi_{P}$ is the sum of the link variables around an elementary plaquette:

&nbsp;

<div align="center">
<img src="assets/plaquette_eq.svg" alt="\phi_{P} \equiv \phi_{\mu\nu}(i) = \phi_{\mu}(i) + \phi_{\nu}(i+\hat{\mu}) - \phi_{\mu}(i+\hat\nu) - \phi_{\nu}(i)"/>
</div>

&nbsp;
 
<div align="center">
  <img src="assets/nerds2.svg" alt="image-20200220120110456" style="width:85vw; min-width:330px; height=auto"/>
</div>

&nbsp;

### Target distribution:

- Our target distribution is then given by:

  <div align="center">
  <img src="assets/target_distribution.svg" alt="\pi(\phi) = \frac{e^{-\beta S[\phi]}}{\mathcal{Z}}"/>
  </div>

  where $\mathcal{Z}$ is the partition function (normalizing factor), $\beta$ is the inverse coupling constant, and $S[\phi]$ is the Wilson gauge action for the 2D $U(1)$ theory.

Lattice methods for the 2D $U(1)$ gauge model are implemented using the `GaugeLattice` object, which can be found at [`l2hmc-qcd/lattice/lattice.py`](l2hmc-qcd/lattice/lattice.py)


# Organization

## Dynamics / Network
The augmented L2HMC leapfrog integrator is implemented in the [`Dynamics`](l2hmc-qcd/dynamics/dynamics.py) object.

The `Dynamics` object is built by subclassing `tf.keras.Model`, and consists of a [`GaugeNetwork`](l2hmc-qcd/network/gauge_network.py) which, in turn, is a collection of `tf.keras.layers`, each of which can be found in [`network/layers.py`](l2hmc-qcd/network/layers.py).

Specific details about the network can be found in
[`l2hmc-qcd/network/gauge_network.py`](l2hmc-qcd/network/gauge_network.py).

## Lattice

Lattice code can be found in [`lattice.py`](l2hmc-qcd/lattice/lattice.py),
specifically the `GaugeLattice` object that provides the base structure on
which our target distribution exists.

Additionally, the `GaugeLattice` object implements a variety of methods for
calculating physical observables such as the average plaquette, $\phi_{P}$, and
the topological charge $\mathcal{Q}$,

## Model

[`models/gauge_model.py`](l2hmc-qcd/models/gauge_model.py) is a wrapper object around the `Dynamics` model and implements methods for calculating the loss as well as running individual training and inference steps. 


## Training / Inference

To train the model, either run the training script from [`bin/train.sh`](bin/train.sh), or modify the command line arguments found in [`bin/train_args.txt`](bin/train_args.txt) and run:
```
python3 train.py @/path/to/train_args.txt
```

Inference is automatically ran after training the model, but can also be ran by loading in a trained model from a checkpoint.

This can be done by running the inference script in [`bin/run.sh`](bin/run.sh), or by simply calling (with any value of `beta (float)` or `run_steps (int)`:
```
python3 run.py --run_steps 1000 --log_dir=/path/to/log_dir --beta 5.
```
where `log_dir` is the directory (automatically created during training) containing the `checkpoints` subdirectory, where the training checkpoints can be found.

All of the relevant command line options are well documented and can be found
in [`l2hmc-qcd/utils/parse_args.py`](l2hmc-qcd/utils/parse_args.py) (training) or [`l2hmc-qcd/utils/parse_inference_args.py`](l2hmc-qcd/utils/parse_inference_args.py) (inference).

 Almost all relevant information about different parameters and run options
 can be found in this file.


# Features

- **Distributed training**
(via [`horovod`](https://github.com/horovod/horovod)): The ability to train the
sampler across multiple nodes (using data-parallelism) can be enabled simply by
passing the `--horovod` command line argument to the training script `train.py`.

# Contact

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

# l2hmc-qcd

---

Application of the L2HMC algorithm to simulations in lattice QCD.

A description of the L2HMC algorithm can be found in the paper:

[*Generalizing Hamiltonian Monte Carlo with Neural Network*](https://arxiv.org/abs/1711.09268)

by [Daniel Levy](http://ai.stanford.edu/~danilevy), [Matt D. Hoffman](http://matthewdhoffman.com/) and [Jascha Sohl-Dickstein](sohldickstein.com)

---

## Overview

Given an analytically described distributions (simple examples can be found in
`utils/distributions.py`), L2HMC enables training of fast-mixing samplers.

---
## Forked implementation for Lattice Gauge Theory models. 

This work is based on the original implementation which can be found at
[brain-research/l2hmc/](https://github.com/brain-research/l2hmc). 

My current focus is on applying this algorithm to simulations in lattice gauge
theory and lattice QCD, in hopes of obtaining greater efficiency compared to
generic HMC.

This new implementation includes the algorithm as applied to the $2D U{(1)}$ lattice gauge theory model (i.e. compact QED).

Additionally, this implementation includes a convolutional neural network
architecture that is prepended to the network described in the original paper.
The purpose of this additional structure is to better incorporate information
about the geometry of the lattice.

Lattice code can be found in `l2hmc/lattice/` and the particular code for the
$2D U{(1)}$ lattice gauge model can be found in `l2hmc/lattice/lattice.py`.


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

# L2HMC: Automatic Training of MCMC Samplers

TensorFlow open source implementation for training MCMC samplers from the paper:

[*Generalizing Hamiltonian Monte Carlo with Neural Network*](https://arxiv.org/abs/1711.09268)

by [Daniel Levy](http://ai.stanford.edu/~danilevy), [Matt D. Hoffman](http://matthewdhoffman.com/) and [Jascha Sohl-Dickstein](sohldickstein.com)

---

Given an analytically described distributions (implemented as in `utils/distributions.py`), L2HMC enables training of fast-mixing samplers. We provide an example, in the case of the Strongly-Correlated Gaussian, in the notebook `SCGExperiment.ipynb` --other details are included in the paper.

---
## Forked implementation for Lattice Gauge Theory models. 

Forked from original version at
[brain-research/l2hmc/](https://github.com/brain-research/l2hmc). The focus of
this implementation is on applying the L2HMC algorithm to lattice gauge theory
models. Current implementation includes U(1) model. 

Additionally, this implementation includes a convolutional neural network
architecture that is prepended to the network described in the original paper.
The purpose of this additional structure is to better incorporate information
about the geometry of the lattice.

Lattice code can be found in `l2hmc/lattice/` with the implementation of gauge
models in `l2hmc/lattice/lattice.py`.



## Contact

***(Original) Code author:*** Daniel Levy

***(Modified) Code author:*** Sam Foreman

***Pull requests and issues for original code:*** @daniellevy

***Pull requests and issues with forked code:*** @saforem2

## Citation

If you use this code, please cite our paper:
```
@article{levy2017generalizing,
  title={Generalizing Hamiltonian Monte Carlo with Neural Networks},
  author={Levy, Daniel and Hoffman, Matthew D. and Sohl-Dickstein, Jascha},
  journal={arXiv preprint arXiv:1711.09268},
  year={2017}
}
```

## Note

This is not an official Google product.

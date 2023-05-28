# Dynamics

<!-- <details open><summary><b>Organization: </b></summary> -->

<details closed> <summary> <b> 📂 <a
href="https://github.com/saforem2/l2hmc-qcd/tree/main/src/l2hmc/dynamics">
dynamics/ </a></b></summary>

 - 🐍 [\_\_init\_\_.py](__init__.py)
 - 🐍 [dynamics.py](./dynamics.py)
 - 📂 [__pytorch__/](./pytorch)
   - 🐍 [\_\_init\_\_.py](pytorch/__init__.py)
   - 🐍 [dynamics.py](pytorch/dynamics.py)
 - 📂 [__tensorflow__/](./tensorflow)
   - 🐍 [\_\_init\_\_.py](tensorflow/__init__.py)
   - 🐍 [dynamics.py](tensorflow/dynamics.py)
</details>


## Overview

Here we describe the `Dynamics` object, the main work-horse of our
application.

We provide both TensorFlow and PyTorch implementations[^1] in
[`pytorch/dynamics.py`](./pytorch/dynamics.py) and
[`tensorflow/dynamics.py`](./tensorflow/dynamics.py)

## Implementation

We describe below the PyTorch implementation.

```python
from l2hmc.configs import DynamicsConfig
from l2hmc.network.pytorch.network import NetworkFactory

class Dynamics(nn.Module):
    def __init__(
            self,
            potential_fn: Callable,
            config: DynamicsConfig,
            network_factory: Optional[NetworkFactory] = None,
    ) -> None:
```

Explicitly, our `Dynamics` object is a subclass of `torch.nn.Module` that
takes as input:

1. `potential_fn: Callable`:  
  The potential ( / _action_ / _negative log likelihood_) of our theory.  

   Should have signature:
   ```python
   def potential_fn(x: torch.Tensor, beta: float) -> torch.Tensor:
   ```

   where `beta ~ 1 / Temperature` is the _inverse coupling constant_ of
   our theory.

2. `config: DynamicsConfig`:  
  A `@dataclass` object defining the configuration used to build
  a `Dynamics` object. Looks like:  
  ```python
  @dataclass
  class DynamicsConfig:
      nchains: int          # num of parallel chains
      group: str            # 'U1' or 'SU3'
      latvolume: List[int]  # lattice volume
      nleapfrog: int        # num leapfrog steps / trajectory
      eps: float            # (initial) step size in leapfrog update
      eps_hmc: Optional[float] = None  # step size for HMC updates
      use_ncp: bool = True  # use Non-Compact Projection for 2D U(1)
      verbose: bool = True
      eps_fixed: bool = False  # use a FIXED step size (non-trainable)
      use_split_xnets: bool = True  # use diff networks for each `x` update
      use_separate_networks: bool = True  # use diff nets for each LF step
      # Switch update style
      # - merge_directions = True:
      #   - N * [forward_update] + N * [backward_update]
      # - merge_directions = False:
      #   - [forward / backward] d ~ U(+,-) ; N * [d update]
      merge_directions: bool = True
  ```

  





[^1]: We strive to keep the implementations as close as possible.

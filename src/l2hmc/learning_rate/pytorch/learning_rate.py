"""
We use the class introduced by

https://nlp.seas.harvard.edu/2018/04/03/attention.html#optimizer
"""
from __future__ import absolute_import, division, print_function, annotations

from torch.optim.lr_scheduler import LambdaLR


def rate(step, model_size, factor, warmup):
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(
            step ** (-0.5),
            step * warmup ** (-1.5)
        )
    )


def lr_schedule(model_size, factor, warmup, optimizer) -> LambdaLR:
    return LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(
            step=step,
            model_size=model_size,
            factor=factor,
            warmup=warmup
        )
    )


class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.model_size = model_size
        self._rate = 0

    def state_dict(self):
        """Returns the state of the warmup scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items()
            if key != 'optimizer'
        }

    def load_state_dict(self, state_dict):
        """Loads the warmup scheduler's state.
        Arguments:
            state_dict (dict): warmup scheduler state. Should be an object
            returned from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return (
            self.model_size ** (-0.5) * min(
                (step ** (-0.5)),
                (step * self.warmup ** (-1.5))
            )
        )

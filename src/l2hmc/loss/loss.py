"""
loss.py

Contains base Loss class.
"""
from __future__ import absolute_import, annotations, division, print_function

from typing import Any, Callable, Optional
from l2hmc.configs import LossConfig


def mixed_loss(loss: float, weight: float) -> float:
    return (weight / loss) - (loss / weight)


class BaseLoss:
    def __init__(
            self,
            config: LossConfig,
            metrics_fn: Callable,
            loss_fns: dict[str, Callable],
            loss_weights: Optional[dict[str, float]] = None,
    ) -> None:
        self.config = config
        self.loss_fns = loss_fns
        self.metrics_fn = metrics_fn
        assert callable(self.metrics_fn)
        if loss_weights is None:
            loss_weights = {key: 1.0 for key in self.loss_fns.keys()}

        self.loss_weights = loss_weights

    def __call__(self, xin, xprop, acc) -> float:
        return self.calc_loss(xin, xprop, acc)

    def metrics(
            self,
            xin: Any,
            xout: Optional[Any] = None,
    ) -> dict:
        """Calculate metrics in addition to the Loss."""
        metrics = {}
        metrics.update(self.metrics_fn(xin))

        if xout is not None:
            metrics_out = self.metrics_fn(xout)
            for key, val in metrics.items():
                metrics[f'd{key}'] = metrics_out[key] - val

        return metrics

    def calc_losses(
            self,
            xin: Any,
            xprop: Any,
            acc: Any
    ) -> tuple[float, dict[str, float]]:
        """Aggregate and calculate all losses."""
        losses = {}
        total = 0.0
        for key, loss_fn in self.loss_fns.items():
            loss = loss_fn(xin, xprop, acc)
            weight = self.loss_weights[key]
            if self.config.use_mixed_loss:
                total += mixed_loss(loss, weight)
            else:
                total += loss / weight

            losses[key] = loss

        return total, losses

    def calc_loss(self, xin: Any, xprop: Any, acc: Any) -> float:
        """Aggregate and calculate all losses."""
        total, _ = self.calc_losses(xin, xprop, acc)
        return total

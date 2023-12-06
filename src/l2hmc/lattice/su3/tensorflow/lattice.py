"""
lattice.py

Contains implementation of generic GaugeLattice object.
"""
from __future__ import absolute_import, annotations, division, print_function
import logging
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

from l2hmc.configs import Charges
import l2hmc.group.su3.tensorflow.group as g
from l2hmc.lattice.lattice import Lattice

log = logging.getLogger(__name__)
# from l2hmc import get_logger
# log = get_logger(__name__)
# from l2hmc.lattice.su3.lattice.

# from typing import Generator
# from l2hmc.lattice.generators import generate_SU3

Array = np.ndarray
PI = np.pi
TWO_PI = 2. * np.pi

Tensor = tf.Tensor
TensorLike = tf.types.experimental.TensorLike
TF_FLOAT = tf.keras.backend.floatx()


def pbc(tup: tuple[int], shape: tuple[int]) -> list:
    return np.mod(tup, shape).tolist()


def mat_adj(mat: Array) -> Array:
    return mat.conj().T


Site: Tuple[int, int, int, int]                # t, x, y, z
Link: Tuple[int, int, int, int, int]           # t, x, y, z, dim
Buffer: Tuple[int, int, int, int, int, int]    # b, t, x, y, z, dim


# ---------------------------------------------------------------
# TODO:
# - Finish implementation of BaseLatticeSU3
# - Write tensorflow and torch implementations as subclasses
#
# class LatticeSU3(BaseLatticeSU3):
# ---------------------------------------------------------------

C1Symanzik = -1.0/12.0  # tree-level
C1Iwasaki = -0.331
C1DBW2 = -1.4088


class LatticeSU3(Lattice):
    """4D Lattice with SU(3) links."""
    dim = 4

    def __init__(
            self,
            nchains: int,
            shape: list[int],
            c1: float = 0.0,
    ) -> None:
        """4D SU(3) Lattice object for dealing with lattice quantities.

        NOTE:
            `x.shape = [nb, 4, nt, nx, ny, nz, 3, 3]`.

        Args:
         - nb (int): Batch dimension, i.e. number of chains to run in parallel
         - shape: Lattice shape, list or tuple of 4 ints
         - c1: Constant indicating whether or not to use rectangle terms ?
        """
        assert len(shape) == 4  # (nb, nt, nx, dim)
        self.g = g.SU3()
        self.nt, self.nx, self.ny, self.nz = shape
        self.volume = self.nt * self.nx * self.ny * self.nz
        self.c1 = tf.constant(c1, TF_FLOAT)
        super().__init__(group=self.g, nchains=nchains, shape=shape)

    def random(self) -> Tensor:
        return self.g.random(list(self._shape))

    def random_momentum(self) -> Tensor:
        return self.g.random_momentum(list(self._shape))

    def update_link(
            self,
            x: Tensor,
            p: Tensor,
    ) -> Tensor:
        return self.g.mul(self.g.exp(p), x)

    def coeffs(self, beta: Tensor | float) -> dict:
        """Coefficients for the plaquette and rectangle terms."""
        beta = tf.cast(beta, TF_FLOAT)
        rect_coeff = tf.scalar_mul(self.c1, beta)
        plaq_coeff = beta * (
            tf.constant(1.0, TF_FLOAT) - tf.constant(8.0, TF_FLOAT) * self.c1
        )

        return {'plaq': plaq_coeff, 'rect': rect_coeff}

    def _link_staple_op(self, link: Tensor, staple: Tensor) -> Tensor:
        return self.g.mul(link, staple)

    def _plaquette(self, x: TensorLike, u: int, v: int):
        """U[μ](x) * U[ν](x+μ) * U†[μ](x+ν) * U†[ν](x)"""
        assert isinstance(x, Tensor)  # and len(x.shape.as_list > 1)
        xuv = self.g.mul(
            x[:, u], tf.roll(x[:, v], shift=-1, axis=u + 1)  # type:ignore
        )
        xvu = self.g.mul(
            x[:, v], tf.roll(x[:, u], shift=-1, axis=v + 1)  # type:ignore
        )
        return self.g.trace(self.g.mul(xuv, xvu, adjoint_b=True))

    def deriv_action_plaq(self, x):
        stf = [[None] * 4 for _ in range(4)]
        stu = [[None] * 4 for _ in range(4)]
        for u in tf.range(1, 4):
            for v in tf.range(0, u):
                xu = tf.roll(x[:, u], shift=-1, axis=v+1)
                xv = tf.roll(x[:, v], shift=-1, axis=u+1)
                xuv = self.g.mul(xu, xv, adjoint_b=True)
                stf[u][v] = self.g.mul(x[:, v], xuv)
                stf[v][u] = self.g.mul(x[:, u], xuv, adjoint_b=True)
                xvu = self.g.mul(x[:, v], x[:, u], adjoint_a=True)
                stu[u][v] = self.g.mul(xvu, xv)
                stu[v][u] = self.g.mul(xvu, xu, adjoint_a=True)

    def _calc_plaq(
            self,
            x: Tensor,
            u: int,
            v: int
    ) -> Tensor:
        xu = x[:, u]  # type:ignore
        xv = x[:, v]  # type:ignore
        xuv = self.g.mul(xu, tf.roll(xv, shift=-1, axis=u+1))
        xvu = self.g.mul(xv, tf.roll(xu, shift=-1, axis=v+1))
        return self.g.trace(self.g.mul(xuv, xvu, adjoint_b=True))

    def _wilson_loops(
            self,
            x: Tensor,
            needs_rect: bool = False
    ) -> tuple[Tensor, Tensor]:
        # y.shape = [nb, d, nt, nx, nx, nx, 3, 3]
        x = tf.reshape(x, self._shape)
        assert isinstance(x, Tensor)
        assert len(x.shape) == 8
        # assert isinstance(x, Tensor)
        pcount = 0
        rcount = 0
        plaqs = tf.TensorArray(x.dtype, size=0, dynamic_size=True)
        rects = tf.TensorArray(x.dtype, size=0, dynamic_size=True)
        for u in range(1, 4):
            for v in range(0, u):
                xu = x[:, u]  # type: ignore
                xv = x[:, v]  # type: ignore
                yuv = self.g.mul(xu, tf.roll(xv, shift=-1, axis=u+1))
                yvu = self.g.mul(xv, tf.roll(xu, shift=-1, axis=v+1))
                plaq = self.g.trace(self.g.mul(yuv, yvu, adjoint_b=True))
                plaqs = plaqs.write(pcount, plaq)
                pcount += 1

                # plaqs.append(plaq)
                if needs_rect:
                    xu = x[:, u]  # type: ignore
                    xv = x[:, v]  # type: ignore
                    yu = tf.roll(xu, shift=-1, axis=v+1)
                    yv = tf.roll(xv, shift=-1, axis=u+1)
                    uu = self.g.mul(xv, yuv, adjoint_a=True)
                    ur = self.g.mul(xu, yvu, adjoint_a=True)
                    ul = self.g.mul(yuv, yu, adjoint_b=True)
                    ud = self.g.mul(yvu, yv, adjoint_b=True)
                    ul_ = tf.roll(ul, shift=-1, axis=u+1)
                    ud_ = tf.roll(ud, shift=-1, axis=v+1)
                    tr_urul_ = (
                        self.g.trace(self.g.mul(ur, ul_, adjoint_b=True))
                    )
                    tr_uuud_ = (
                        self.g.trace(self.g.mul(uu, ud_, adjoint_b=True))
                    )
                    rects.write(rcount, tr_urul_)
                    rects.write(rcount + 1, tr_uuud_)
                    rcount += 1

        return plaqs.stack(), rects.stack()

    def _plaquettes(self, x: Tensor) -> Tensor:
        ps, rs = self._wilson_loops(x)
        plaqs = tf.reduce_sum(tf.math.real(ps), axis=range(2, len(ps.shape)))
        psum = tf.reduce_sum(plaqs, axis=0)

        # NOTE: return psum / (len(ps) * dim(link) * volume)
        return psum / (6 * 3 * self.volume)

    def _plaqs(self, wloops: Tensor) -> Tensor:
        plaqs = tf.reduce_sum(
            tf.math.real(wloops),
            axis=range(2, len(wloops.shape))
        )
        psum = tf.reduce_sum(plaqs, axis=0)

        return psum / (6 * 3 * self.volume)

    def charges(self, x: Tensor) -> Charges:
        ps, _ = self._wilson_loops(x)
        return Charges(intQ=self._int_charges(wloops=ps),
                       sinQ=self._sin_charges(wloops=ps))

    def int_charges(self, x: Tensor) -> Tensor:
        ps, _ = self._wilson_loops(x)
        return self._int_charges(wloops=ps)

    def sin_charges(self, x: Tensor) -> Tensor:
        ps, _ = self._wilson_loops(x)
        return self._sin_charges(wloops=ps)

    def _charges(self, wloops: Tensor) -> Charges:
        wloops_imag = tf.math.reduce_sum(
            tf.math.imag(wloops),
            axis=range(2, len(wloops.shape))
        )
        qsum = tf.math.reduce_sum(wloops_imag, axis=0)
        qint = qsum / (32 * (np.pi ** 2))
        qsin = qsum / (6 * 3 * self.volume)
        return Charges(intQ=qint, sinQ=qsin)

    def _int_charges(self, wloops: Tensor) -> Tensor:
        qint = tf.reduce_sum(
            tf.math.imag(wloops),
            axis=range(2, len(wloops.shape))
        )
        qsum = tf.reduce_sum(qint, axis=0)

        return qsum / (32 * (np.pi ** 2))

    def _sin_charges(self, wloops: Tensor) -> Tensor:
        qsin = tf.reduce_sum(
            tf.math.imag(wloops),
            axis=range(2, len(wloops.shape))
        )
        qsum = tf.reduce_sum(qsin, axis=0)

        return qsum / (6 * 3 * self.volume)

    def wilson_loops(self, x: Tensor) -> Tensor:
        ps, _ = self._wilson_loops(x=x, needs_rect=False)
        return ps

    def kinetic_energy(self, v: Tensor) -> Tensor:
        return self.g.kinetic_energy(
            tf.reshape(v, self._shape)
        )

    def potential_energy(self, x: Tensor, beta: Tensor) -> Tensor:
        return self.action(x, beta)

    def action(
            self,
            x: Tensor,
            beta: Tensor,
    ) -> Tensor:
        """Returns the action"""
        coeffs = self.coeffs(beta)
        ps, rs = self._wilson_loops(x, needs_rect=self.c1 != 0)
        assert isinstance(x, Tensor)
        plaqs = tf.reduce_sum(
            tf.math.real(ps),
            axis=range(2, len(ps.shape))
        )
        psum = tf.cast(tf.reduce_sum(plaqs, axis=0), TF_FLOAT)
        pcoeff = tf.cast(coeffs['plaq'], TF_FLOAT)

        action = tf.math.multiply(pcoeff, psum)

        if self.c1 != 0:
            rects = tf.reduce_sum(
                tf.math.real(rs),
                axis=range(2, len(rs.shape))
            )
            rsum = tf.reduce_sum(rects, axis=0)
            rcoeff = tf.cast(coeffs['rect'], TF_FLOAT)
            action += tf.math.multiply(rcoeff, rsum)

        return action * tf.constant(-1.0 / 3.0, TF_FLOAT)

    def _action(
            self,
            wloops: tuple[Tensor, Tensor],
            beta: Tensor,
    ) -> Tensor:
        coeffs = self.coeffs(beta)
        ps, rs = wloops
        psum = tf.math.reduce_sum(
            tf.math.reduce_sum(tf.math.real(ps), range(2, len(ps.shape)))
        )
        pcoeff = tf.cast(coeffs['plaq'], TF_FLOAT)
        action = tf.scalar_mul(pcoeff, psum)
        if self.c1 != 0:
            rsum = tf.math.reduce_sum(
                tf.math.reduce_sum(
                    tf.math.real(rs),
                    axis=range(2, len(rs.shape))
                ),
                axis=0,
            )
            rcoeff = tf.cast(coeffs['rect'], TF_FLOAT)
            action += tf.math.multiply(rcoeff, rsum)
            # action = action + coeffs['rect'] * rsum

        return tf.divide(action, 3.0)

    def action_with_grad(
            self,
            x: Tensor,
            beta: Tensor
    ) -> tuple[Tensor, Tensor]:
        if tf.executing_eagerly():
            with tf.GradientTape() as tape:
                tape.watch(x)
                s = self.action(x, beta)

            dsdx = tape.gradient(s, x)
        else:
            s = self.action(x, beta)
            dsdx = tf.gradients(s, [x])[0]

        dsdx = self.g.projectTAH(self.g.mul(dsdx, x, adjoint_b=True))

        return s, dsdx

    def grad_action(self, x: Tensor, beta: Tensor) -> Tensor:
        """Returns the derivative of the action"""
        if tf.executing_eagerly():
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(x)
                action = self.action(x, beta)
            g = tape.gradient(action, x)
        else:
            g = tf.gradients(self.action(x, beta), [x])[0]

        return self.g.projectTAH(self.g.mul(g, x, adjoint_b=True))

    def calc_metrics(
            self,
            x: Tensor,
            beta: Optional[Tensor] = None,
            xinit: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        # wloops = self.wilson_loops(x)
        # wloops = tf.reduce_sum(wloops, 0)
        plaqs = self._plaquettes(x)
        q = self.charges(x)
        # qsin = self._sin_charges(wloops)
        # qint = self._int_charges(wloops)
        # TODO: FIX ME
        metrics = {'plaqs': plaqs, 'sinQ': q.sinQ, 'intQ': q.intQ}
        if beta is not None:
            # s = self.action(x, beta)
            s, dsdx = self.action_with_grad(x, beta)
            metrics.update({
                'action': s,
                'dsdx': dsdx,
            })
            if xinit is not None:
                # action_ = self.action(xinit, beta)
                s_, dsdx_ = self.action_with_grad(xinit, beta)
                metrics.update({
                    'daction': tf.abs(tf.subtract(s, s_)),
                    'dsdx': tf.abs(tf.subtract(dsdx, dsdx_))
                })

        if xinit is not None:
            wloops_ = self.wilson_loops(xinit)
            plaqs_ = self.plaqs(wloops=wloops_)
            q_ = self._charges(wloops=wloops_)
            metrics.update({
                'dplaqs': tf.abs(plaqs, plaqs_),
                'dQint': tf.abs(q.intQ - q_.intQ),
                'dQsin': tf.abs(q.sinQ - q_.sinQ),
            })

        if beta is not None:
            s = self.action(x, beta)
            metrics['action'] = s
        return metrics

    @staticmethod
    def mixed_loss(loss: Tensor, weight: float) -> Tensor:
        w = tf.constant(weight, dtype=TF_FLOAT)
        return (w / loss) - (loss / w)

    def plaq_loss(
            self,
            acc: Tensor,
            x1: Tensor,
            x2: Tensor,
            use_mixed_loss: Optional[bool] = None,
            weight: Optional[float] = 0.0,
    ) -> Tensor:
        wloops1 = self.wilson_loops(x1)
        wloops2 = self.wilson_loops(x2)
        return self._plaq_loss(
            acc=acc,
            wloops1=wloops1,
            wloops2=wloops2,
            weight=weight,
            use_mixed_loss=use_mixed_loss,
        )

    def charge_loss(
            self,
            acc: Tensor,
            x1: Tensor,
            x2: Tensor,
            use_mixed_loss: Optional[bool] = None,
            weight: Optional[float] = 0.0,
    ) -> Tensor:
        wloops1 = self.wilson_loops(x1)
        wloops2 = self.wilson_loops(x2)
        return self._charge_loss(
            acc=acc,
            wloops1=wloops1,
            wloops2=wloops2,
            weight=weight,
            use_mixed_loss=use_mixed_loss,
        )

    def _plaq_loss(
            self,
            acc: Tensor,
            wloops1: Tensor,
            wloops2: Tensor,
            use_mixed_loss: Optional[bool] = None,
            weight: Optional[float] = None,
    ) -> Tensor:
        weight = 1.0 if weight is None else weight
        dw = tf.reduce_sum(tf.subtract(wloops2, wloops1), axis=0)
        # calculate squared plaquette diff. as
        #   dwilson_loops = 2. * (1. - cos(w2 - w1))
        ploss = acc * tf.reduce_sum(
            2. * (tf.ones_like(wloops1) - tf.math.cos(dw)),
            axis=tuple(range(2, len(wloops1.shape)))
        )
        if use_mixed_loss:
            ploss += 1e-4  # to prevent division by zero in mixed_loss
            return tf.reduce_mean(self.mixed_loss(ploss, weight))

        return tf.reduce_mean(-ploss / weight)

    def _charge_loss(
            self,
            acc: Tensor,
            wloops1: Tensor,
            wloops2: Tensor,
            use_mixed_loss: Optional[bool] = None,
            weight: Optional[float] = None,
    ) -> Tensor:
        """Calculate the charge loss from initial and proposed Wilson loops."""
        weight = 1.0 if weight is None else weight
        qloss = acc * tf.math.square(
            tf.subtract(
                self._sin_charges(wloops=wloops2),
                self._sin_charges(wloops=wloops1),
            )
        )
        if use_mixed_loss:
            qloss += 1e-4
            return tf.reduce_mean(self.mixed_loss(qloss, weight))
        return tf.reduce_mean(-qloss / weight)

    def calc_loss(
            self,
            xinit: Tensor,
            xprop: Tensor,
            acc: Tensor,
            use_mixed_loss: Optional[bool] = True,
            charge_weight: Optional[float] = None,
            plaq_weight: Optional[float] = None,
    ) -> Tensor:
        plaq_weight = 1.0 if plaq_weight is None else plaq_weight
        charge_weight = 1.0 if charge_weight is None else charge_weight

        w1 = self.wilson_loops(x=xinit)
        w2 = self.wilson_loops(x=xprop)

        loss = tf.constant(0.0, dtype=TF_FLOAT)
        if plaq_weight > 0.0:
            loss += self._plaq_loss(
                acc=acc,
                wloops1=w1,
                wloops2=w2,
                weight=plaq_weight,
                use_mixed_loss=use_mixed_loss,
            )
        if charge_weight > 0.0:
            loss += self._charge_loss(
                acc=acc,
                wloops1=w1,
                wloops2=w2,
                weight=charge_weight,
                use_mixed_loss=use_mixed_loss,
            )

        return loss


if __name__ == '__main__':
    lattice = LatticeSU3(3, [4, 4, 4, 8])
    beta = tf.constant(1.0)
    x = lattice.random()
    v = lattice.random_momentum()
    action = lattice.action(x, beta)
    kinetic = lattice.kinetic_energy(v)

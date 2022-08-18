import jax
from jax import grad
import jax.numpy as jnp

import numpy as np
from jax import jit
from jax.lax import lgamma
from tqdm import tqdm

from typing import Optional, List

from jax.example_libraries import optimizers
import repair.utils as ut
import repair.funcs as F


class Model:
    def __init__(
        self,
        A: np.array,
        B: np.array,
        A_names: Optional[List[str]] = None,
        B_names: Optional[List[str]] = None,
        l1: float = 1e-8,
        n_bins: int = 5,
        base: Optional[float] = None,
    ):

        self.A = jnp.asarray(A.astype(np.float32))
        self.B = jnp.asarray(B.astype(np.float32))

        self.A_names = np.array(A_names,dtype=object)
        self.B_names = np.array(B_names,dtype=object)

        self.A = F.standard_normalize(self.A)
        # TODO: test if this is important...
        # self.A = self.A / jnp.linalg.norm(self.A, axis=1, keepdims=True)
        self.B = F.standard_normalize(self.B)

        self.n_bins = n_bins
        self.bins = self.binarize(A)

        self.S = self.A.shape[0]
        self.Ca = self.A.shape[1]
        self.Cb = self.B.shape[1]

        self.A = self.A.T
        self.B = self.B.T

        self.l1 = l1

        self.Q = None
        self.p = None

        if base is None:
            self.base = np.e
        else:
            self.base = float(base)

        self._initialize()
        self._update_params()

    def _initialize(
        self,
    ):
        if self.A_names is None:
            self.A_names = [str(x) for x in range(self.Ca)]
        if self.B_names is None:
            self.B_names = [str(x) for x in range(self.Cb)]

        self._build_loss()
        self.Q = jnp.asarray(np.random.normal(0, 1, size=(self.Ca, self.Cb)))
        self.p = jnp.asarray(np.random.normal(0, 1, size=(self.n_bins, 1)))

    def binarize(self, X) -> np.ndarray:
        sums = np.sum(X, axis=0)
        qs = np.quantile(sums, np.linspace(0, 1, self.n_bins + 1))
        bins = np.digitize(sums, qs) - 1
        return jnp.asarray(bins)

    def _update_params(
        self,
    ):
        self.params = {
            "Q": self.Q,
            "p": self.p,
        }

    @property
    def M(
        self,
    ):
        return np.asarray(ut.custom_softmax(self.Q, base=self.base, axis=1))

    def _build_loss(
        self,
    ):
        if self.l1 is None or self.l1 == 0:
            self.loss = self._mse_loss
        else:
            self.loss = self._mse_entropy_loss

    def _mse_loss(self, Q, p, **kwargs):
        M = jax.nn.softmax(Q, axis=1)
        g = jnp.exp(p)[self.bins, :]
        lossValue = F.mse_loss(self.A, self.B, M, g)

        return lossValue

    def _mse_entropy_loss(self, Q, p, **kwargs):
        M = jax.nn.softmax(Q, axis=1)
        g = jnp.exp(p)[self.bins]
        lossValue = F.mse_loss(self.A, self.B, M, g)
        lossValue += self.l1 * F.entropy_loss(M)

        return lossValue

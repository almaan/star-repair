import jax
from jax import grad
import jax.numpy as jnp

import numpy as np
from jax import jit
from jax.lax import lgamma
from tqdm import tqdm

from typing import Optional, Union

from jax.example_libraries import optimizers
import repair.models as m
import anndata as ad


@jit
def mse_loss(A,B,M,g):
    kgmb = g * jnp.dot(M,B)
    mse = jnp.mean((A - kgmb) **2)
    return mse

@jit
def entropy_loss(X:jnp.array):
    return -jnp.sum(X * jnp.log(X))

@jit
def standard_normalize(X:jnp.array,axis:int=0):
    mu = jnp.mean(X,axis=axis,keepdims = True)
    std = jnp.std(X,axis=axis,keepdims = True)
    return (X-mu) / std

def fit(model: "m.Models",
        n_steps:int,
        learning_rate:int = 1e-2,
        verbose:bool=False)->np.array:
    opt_init, opt_update, get_params = optimizers.adam(step_size=learning_rate)
    opt_state = opt_init((model.Q,model.p))

    lossHistory = np.zeros(n_steps)

    def step(s,opt_state):
        value, grads = jax.value_and_grad(model.loss,argnums = (0,1))(*get_params(opt_state))
        opt_state = opt_update(s,
                               grads,
                               opt_state)

        return value, opt_state

    if verbose:
        iterator = tqdm(range(n_steps))
    else:
        iterator = range(n_steps)

    for i in iterator:
        value, opt_state = step(i, opt_state)
        lossHistory[i] = float(value)
        model.Q,model.p = get_params(opt_state)

    return lossHistory


def filter(adata: ad.AnnData, min_counts: Optional[int]= None, min_x_obs:
           Optional[Union[int,float]] = None,x: int = 1)->ad.AnnData:

    keep = np.ones(adata.shape[1]).astype(bool)
    if min_counts is not None:
        n_total = np.asarray(adata.X.sum(axis=0))
        keep *= n_total > min_counts

    if min_x_obs is not None:
        n_above_x = np.sum(adata.X >= x,axis=0)
        keep *= n_above_x >= min_x_obs

    adata = adata[:,keep]
    return adata




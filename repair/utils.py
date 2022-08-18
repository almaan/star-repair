import os.path as osp
import numpy as np
from typing import Tuple, List,Any, Union, Optional
import pandas as pd
import anndata as ad
import re
import jax.numpy as jnp
from jax import lax
import repair.constants as C
import datetime



Array = Any


def _read_data(pth:str)->Any:
    if pth.endswith(("tsv", "tsv.gz")):
        data = pd.read_csv(pth, header=0, index_col=0, sep="\t")
    elif pth.endswith(("csv", "csv.gz")):
        data = pd.read_csv(pth, header=0, index_col=0, sep=",")
    elif pth.endswith("h5ad"):
        data = ad.read_h5ad(pth)
    else:
        filetype = ".".join(osp.basename(pth).split(".")[1::])
        raise NotImplementedError(
            "{} does not currently support files like {}".format(C.NAME, filetype)
        )
    return data



def get_string_num(string):
    return list(map(lambda x:re.findall("[0-9]+",x)[0],string))

def custom_softmax(x: Array,
                   base: Array = jnp.e,
                   axis: Optional[Union[int, Tuple[int, ...]]] = -1,
                   where: Optional[Array] = None,
                   initial: Optional[Array] = None) -> Array:

  r"""Softmax function.

  Computes the function which rescales elements to the range :math:`[0, 1]`
  such that the elements along :code:`axis` sum to :math:`1`.

  .. math ::
    \mathrm{softmax}(x) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}

  Args:
    x : input array
    axis: the axis or axes along which the softmax should be computed. The
      softmax output summed across these dimensions should sum to :math:`1`.
      Either an integer or a tuple of integers.
    where: Elements to include in the :code:`softmax`.
    initial: The minimum value used to shift the input array. Must be present
      when :code:`where` is not None.
  """
  x_max = jnp.max(x, axis, where=where, initial=initial, keepdims=True)
  unnormalized = jnp.power(float(base),x - lax.stop_gradient(x_max))
  return unnormalized / jnp.sum(unnormalized, axis, where=where, keepdims=True)


# def tuple_list_to_dict(tuple_list: List[Tuple[S,T]])->Dict[S,T]:
#     return {a:b for a,b in tuple_list}


def timestamp() -> str:
    return re.sub(':|-|\.| |','',
                  str(datetime.datetime.now()))

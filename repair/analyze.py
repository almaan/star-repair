import repair.models as m
import repair.utils as ut
import anndata as ad
import os.path as osp
import repair.constants as C
from typing import Tuple, List, Optional, Union
import numpy as np
import re
import pandas as pd
import lap
import repair.funcs as F
import anndata as ad


def read_data(
    pth: str,
) -> ad.AnnData:

    cnt = ut._read_data(pth)

    if not isinstance(cnt, ad.AnnData):
        cnt = ad.AnnData(cnt)

    return cnt


def match_name(s: List[str], p: str):
    if isinstance(s, str):
        s = [s]

    return np.array(list(map(lambda x: re.search(p, x) is not None, s)),dtype = bool)


def split(
    adata: ad.AnnData, chA_pattern: str, chB_pattern: str
) -> Tuple[ad.AnnData, ad.AnnData]:

    is_a = match_name(adata.var.index.values, chA_pattern)
    is_b = match_name(adata.var.index.values, chB_pattern)

    adata_A = adata[:, is_a]
    adata_B = adata[:, is_b]

    adata_A.var["clone_num"] = ut.get_string_num(adata_A.var.index)
    adata_A.index = adata_A.var.clone_num
    adata_B.var["clone_num"] = ut.get_string_num(adata_B.var.index)
    adata_B.index = adata_B.var.clone_num

    return adata_A, adata_B


def assign(model: "m.Model") -> pd.DataFrame:
    _, row, col = lap.lapjv(-np.asarray(model.M), extend_cost=True)
    ab_map = row
    # ab_map = np.argmax(model.M, axis=1)

    score = model.M[(np.arange(len(model.A_names)),ab_map)]
    res = pd.DataFrame(
        dict(chainA=model.A_names, chainB=model.B_names[ab_map], score = score),
        index=["Pair_{}".format(x) for x in range(len(ab_map))],
    )
    return res


def run(
    pth: str,
    A_pattern: str,
    B_pattern: str,
    n_bins: int,
    base: Optional[float],
    n_steps: int,
    learning_rate: float,
    l1: float,
    global_dict: Optional[dict],
    min_counts: Optional[int] = None,
    min_x_obs: Optional[Union[int, float]] = None,
    x: Optional[int] = 1,
    save_output: bool = True,
    save_adata: bool = False,
    save_loss_history: bool = False,
) -> pd.DataFrame:

    if global_dict is None:
        global_dict = dict()

    if "outdir" not in global_dict:
        global_dict["outdir"] = osp.dirname(pth)

    adata = read_data(pth)
    if save_adata and not pth.endswith(".h5ad"):
        adata.write_h5ad(
            osp.join(
                global_dict["outdir"],
                global_dict.get("tag", C.NAME) + "_anndata.h5ad",
            )
        )

    adata = F.filter(adata, min_counts, min_x_obs)
    adata_A, adata_B = split(adata, A_pattern, B_pattern)
    del adata

    model = m.Model(
        adata_A.X,
        adata_B.X,
        A_names=list(adata_A.var.index),
        B_names=list(adata_B.var.index),
        n_bins=n_bins,
    )

    lossHistory = F.fit(model, n_steps=n_steps, verbose=True, learning_rate=1e-2)
    if save_loss_history:
        loss_path = osp.join(
            global_dict["outdir"],
            global_dict.get("tag", C.NAME) + "_loss_history.dat",
        )
        with open(loss_path, "w+") as f:
            f.writelines("\n".join([str(x) for x in lossHistory]))

    results = assign(model)

    if save_output:
        results.to_csv(osp.join(global_dict["outdir"],global_dict["tag"] + "_analysis_result.tsv"),sep = "\t")

    return results

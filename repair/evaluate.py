import repair.models as m
import repair.utils as ut
import anndata as ad
import os.path as osp
import repair.constants as C
from typing import Tuple, List, Optional, Union, Any, Dict
import numpy as np
import re
import pandas as pd
import lap
import repair.funcs as F
import anndata as ad


def make_pair_list(df: Union[pd.DataFrame,ad.AnnData],
                   col_A: Optional[str] = None,
                   col_B: Optional[str] = None,
                   )->List[Tuple[Any,Any]]:

    if col_A is None and col_B is None:
        col_A,col_B = df.columns[0:2]
    elif col_A is None:
        col_A = df.columns[np.argmax(df.columns != col_B)]
    elif col_B is None:
        col_B = df.columns[np.argmax(df.columns != col_A)]


    pair_list = list(zip(df[col_A].values,df[col_B].values))
    return pair_list


def match_lists(l1:List[Tuple[str,str]],l2: List[Tuple[str,str]]):

    l1_1,_ = zip(*l1)
    l2_1,l2_2 = zip(*l2)

    inter_l1_1_l2_1 = len(set(l1_1).intersection(set(l2_1)))
    inter_l1_1_l2_2 = len(set(l1_1).intersection(set(l2_2)))

    if inter_l1_1_l2_1 < inter_l1_1_l2_2:
        l2 = list(zip(l2_2,l2_1)) 

    return l1,l2

def evaluate_pairing(inferred: List[Tuple[str,str]],
                     true: List[Tuple[str,str]],
                     scores: Optional[List[float]] = None,
                     )->pd.DataFrame:
    res = []

    li,lt = match_lists(inferred,true)

    for k,(a,b) in enumerate(li):
        for gtp in lt:
            if a in gtp:
                if b in gtp:
                    status = "C"
                else:
                    status = "F"
                res.append([a,b,status])
                if scores is not None:
                    res[-1].append(scores[k])

            if b in gtp:
                if a in gtp:
                    status = "C"
                else:
                    status = "F"
                res.append([a,b,status])
                if scores is not None:
                    res[-1].append(scores[k])


    columns = ["inf_chainA","inf_chainB","status"]
    if scores is not None:
        columns.append("score")

    res = pd.DataFrame(res,columns = columns)

    summary  = dict()

    nCorrect = sum(res.status.values == "C")
    nFalse = len(res) - nCorrect

    summary["all"] = pd.DataFrame(dict(correct = nCorrect, incorrect = nFalse, accuracy = round(nCorrect / len(res),2)),index = ["values"])

    if scores is not None:
        _res = res.copy()
        res.sort_values("score",inplace = True,ascending = True)
        res.drop_duplicates("inf_chainA",inplace = True,keep = "first")
        nCorrect = sum(res.status.values == "C")
        nFalse = len(res) - nCorrect
        summary["high"] = pd.DataFrame(dict(correct = nCorrect, incorrect = nFalse, accuracy = round(nCorrect / len(res),2)), index = ["values"])
        res = _res

    return res,summary



def run(inferred_pth: str,
        ground_truth_pth: str,
        cols_i: Optional[Tuple[str,str]] = None,
        cols_t: Optional[Tuple[str,str]] = None,
        save_output: bool = True,
        global_dict: Optional[dict] = None,
        )->Tuple[pd.DataFrame,Dict[str,pd.DataFrame]]:

    if global_dict is None:
        global_dict = dict()

    if "outdir" not in global_dict:
        global_dict["outdir"] = osp.basename(inferred_pth)

    if "tag" not in global_dict:
        global_dict["tag"] = C.NAME


    inf = ut._read_data(inferred_pth)
    gt = ut._read_data(ground_truth_pth)

    if cols_i is not None:
        col_A_i,col_B_i = cols_i
    else:
        col_A_i,col_B_i = None,None

    li = make_pair_list(inf,col_A = col_A_i,col_B = col_B_i)

    if cols_t is not None:
        col_A_t,col_B_t = cols_t
    else:
        col_A_t,col_B_t = None,None

    lt = make_pair_list(gt,col_A = col_A_t,col_B = col_B_t)

    res,summary = evaluate_pairing(li,lt,scores = inf["score"].values)

    if save_output:
        res.to_csv(osp.join(global_dict["outdir"],global_dict["tag"] + "_evaluation_result.tsv"),sep = "\t")

        for k,v in summary.items():
            v.to_csv(osp.join(global_dict["outdir"],global_dict["tag"] + "_evaluation_summary_{}.tsv".format(k)),sep = "\t")

    return res,summary











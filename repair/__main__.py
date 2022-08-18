#!/usr/bin/python3
import argparse as arp
import repair.evaluate as ev
import repair.analyze as az
import repair.utils as ut
from os import getcwd
import numpy as np
import repair.constants as C
import os.path as osp


def parser()->arp.ArgumentParser:

    parent_parser = arp.ArgumentParser(add_help = False)

    subparsers = parent_parser.add_subparsers(dest = "command")
    analyze_parser = subparsers.add_parser("analyze",formatter_class = arp.ArgumentDefaultsHelpFormatter)
    apaa = analyze_parser.add_argument

    apaa("-i","--input",type = str,required = True)
    apaa("-o","--outdir",type = str, required =False,default = None)
    apaa("-ap","--a_pattern",type = str, required = False,default = "^IGH")
    apaa("-bp","--b_pattern",type = str, required = False,default = "^IGL")
    apaa("-lr","--learning_rate",type = float, default = 1e-2)
    apaa("-e","--epochs",type = int, default = 2000)
    apaa("-l1","--l1_regularization",type = float, default = 0.0)
    apaa("-mxo","--min_x_obs", default = None,type = float)
    apaa("-x","--x",default = 1,type = float)
    apaa("-mc","--min_counts", default = None,type = float)
    apaa("-b","--base",type = float, default = np.e)
    apaa("-nb","--n_bins",type = int, default = 1)
    apaa("-sa","--save_adata",action = "store_true",default = False)

    eval_parser = subparsers.add_parser("evaluate",formatter_class = arp.ArgumentDefaultsHelpFormatter)
    evaa = eval_parser.add_argument

    evaa("-i","--inferred_pairs",type = str, required = True)
    evaa("-gt","--ground_truth",type = str, required = True)
    evaa("-o","--outdir",type = str, required =False)
    evaa("-ci","--colnames_inferred",nargs = 2, default = None)
    evaa("-ct","--colnames_ground_truth",nargs = 2, default = None)


    return parent_parser


def run(args):

    if args.outdir is None:
        args.outdir = getcwd()

    global_dict = dict(outdir = args.outdir)

    if args.command == "analyze":
        global_dict["tag"] = ut.timestamp()

        results = az.run(pth = args.input,
                         A_pattern = args.a_pattern,
                         B_pattern = args.b_pattern,
                         n_steps = args.epochs,
                         n_bins = args.n_bins,
                         base = args.base,
                         learning_rate = args.learning_rate,
                         l1 = args.learning_rate,
                         min_counts = args.min_counts,
                         min_x_obs = args.min_x_obs,
                         x = args.x,
                         save_output = True,
                         save_adata  = args.save_adata,
                         save_loss_history = True,
                         global_dict = global_dict,
        )

    elif args.command == "evaluate":
        import re
        tag_pattern = "[0-9]{20}"
        match = re.match(tag_pattern,osp.basename(args.inferred_pairs))
        if match is not None:
            global_dict["tag"] = match.group()
        else:
            global_dict["tag"] = C.NAME

        results = ev.run(inferred_pth = args.inferred_pairs,
                         ground_truth_pth = args.ground_truth,
                         cols_i = args.colnames_inferred,
                         cols_t = args.colnames_ground_truth,
                         save_output = True,
                         global_dict = global_dict,
                         )
def main()->None:
    prs = parser()
    args = prs.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        from sys import exit
        print("Terminated by user")
        exit(-1)


if __name__ == "__main__":
    main()


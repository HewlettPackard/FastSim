"""
Explore feature importance for application power usage via decision tress
"""

import argparse, os

import pandas as pd
import numpy as np

from funcs import parse_cache


POWER_COLS = ["JobID", "Start", "End", "ConsumedEnergyRaw"]
SUBMIT_COLS = ["ReqCPUS", "ReqNodes", "Group", "QOS", "ReqMem",
               "Submit", "JobName", "Timelimit", "SubmitLine"]
FINISH_COLS = [] # TODO

def clean_df(df, queue_only=False):
    df = df.loc[
        (
            (df.Group.notna()) & (df.ReqMem.notna()) & (~df.ReqMem.str.contains("\?")) &
            (df.QOS.notna()) & (df.SubmitLine.notna()) & (df.Timelimit.notna()) &
            (df.Timelimit != "Partition_Limit") & (df.Timelimit != "UNLIMITED")
        )
    ]

    cols=["ReqCPUS", "ReqNodes", "ReqMem"]
    df[cols] = df[cols].replace(
        { "K" : "e+03", "M" : "e+06", "G" : "e+09", "T" : "e+12" }, regex=True
    ).astype(float).astype(int)

    if not queue_only:
        pass # TODO loc + replace for new cols

    return df

def main(args):
    df_power = parse_cache(
        args.data,
        args.cache,
        ".".join(os.path.basename(args.data).split(".")[:-1]),
        "decision_tree_queue_df",
        cols=POWER_COLS+SUBMIT_COLS if args.queue_data else POWER_COLS+SUBMIT_COLS+FINISH_COLS
    )

    df_power = clean_df(df_power, queue_only=args.queue_data)

    print(np.sort(df_power.ReqCPUS.unique()))
    print(np.sort(df_power.ReqNodes.unique()))
    print(np.sort(df_power.Group.unique()))
    print(np.sort(df_power.ReqMem.unique()))
    print(np.sort(df_power.QOS.unique()))
    print(np.sort(df_power.JobName.unique()))
    print(np.sort(df_power.Submit.unique()))
    print(np.sort(df_power.SubmitLine.unique()))
    print(np.sort(df_power.Timelimit.unique()))



def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("data", type=str)

    datacols_group = parser.add_mutually_exclusive_group()
    datacols_group.add_argument(
        "--queue_data", action="store_true",
        help="Use only job submission data"
    )
    datacols_group.add_argument(
        "--queue_and_run_data", action="store_true",
        help="Use job submission and completion data"
    )

    parser.add_argument(
        "--cache", type=str, default="",
        help="How to use the cache (save|load)"
    )

    return parser.parse_args()

if __name__ == '__main__':
    pd.options.display.float_format = '{:.0f}'.format

    main(parse_arguments())


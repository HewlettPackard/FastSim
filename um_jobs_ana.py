"""
"""

import argparse

import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from funcs import timelimit_str_to_timedelta

pd.options.mode.chained_assignment = None


def clean(df, no_nodelist=False):
    df.JobID = df.JobID.astype(str)
    df_jobs = df.loc[(~df.JobID.str.contains("\."))]
    df_steps = df.loc[(df.JobID.str.contains("\."))]

    len_original = len(df_jobs)
    df_jobs = df_jobs.loc[(df_jobs.State.isin(["COMPLETED", "TIMEOUT"]))]
    print("{} failed jobs removed".format(len_original - len(df_jobs)))

    df_steps["ParentJobID"] = df_steps.JobID.apply(lambda row: row.split(".")[0])

    cols_to_convert = [
        "MaxRSS", "MaxVMSize", "MaxDiskWrite", "MaxDiskRead", "MaxPages", "AvePages", "NTasks",
        "AveDiskWrite", "AveDiskRead", "AveVMSize", "AveRSS"
    ]
    df_steps = df_steps.loc[(df_steps[cols_to_convert].notna().all(axis=1))]
    df_steps[cols_to_convert] = df_steps[cols_to_convert].replace(
        { "K" : "e+03", "M" : "e+06", "G" : "e+09", "T" : "e+12" }, regex=True
    ).astype(float).astype(int)

    len_original = len(df_jobs)
    df_jobs = df_jobs.loc[(
        df_jobs.apply(
            lambda row: (
                row.JobID in df_steps.ParentJobID.unique() and
                df_steps.ParentJobID.value_counts()[row.JobID] == 3
            ),
            axis=1
        )
    )]
    print("{} jobs without steps removed".format(len_original - len(df_jobs)))

    cols_to_convert = ["ReqCPUS", "ReqNodes", "ReqMem", "AllocNodes"]
    df_jobs[cols_to_convert] = df_jobs[cols_to_convert].replace(
        { "K" : "e+03", "M" : "e+06", "G" : "e+09", "T" : "e+12" }, regex=True
    ).astype(float).astype(int)

    df_jobs.Elapsed = df_jobs.Elapsed.apply(
        lambda row: round(timelimit_str_to_timedelta(row).total_seconds())
    )
    df_jobs.TotalCPU = df_jobs.TotalCPU.apply(
        lambda row: round(timelimit_str_to_timedelta(row).total_seconds())
    )
    df_jobs.CPUTime = df_jobs.CPUTime.apply(
        lambda row: round(timelimit_str_to_timedelta(row).total_seconds())
    )
    # 144 have ConsumedEnergy raw empty or 0 and some have very large, not using this for now
    # print(len(df_jobs.loc[(df_jobs.ConsumedEnergyRaw == '') | (df_jobs.ConsumedEnergyRaw == '0')]))
    # df_jobs["PowerPerNode"] = df_jobs.apply(
    #     lambda row: (
    #         float(row.ConsumedEnergyRaw) / float(row.Elapsed * row.AllocNodes)
    #     ),
    #     axis=1
    # )

    df_jobs["MaxRSSJob"] = np.nan
    df_jobs["AveRSSJob"] = np.nan
    df_jobs["MaxVMSizeJob"] = np.nan
    df_jobs["AveVMSizeJob"] = np.nan
    df_jobs["MaxDiskReadJob"] = np.nan
    df_jobs["AveDiskReadJob"] = np.nan
    df_jobs["MaxDiskWriteJob"] = np.nan
    df_jobs["AveDiskWriteJob"] = np.nan
    df_jobs["MaxPagesJob"] = np.nan
    df_jobs["AvePagesJob"] = np.nan
    df_jobs["AveCPUJob"] = np.nan

    for jobid in tqdm(df_jobs.JobID.unique(), desc="Aggregating data from job steps..."):
        df_steps_slice = df_steps.loc[(df_steps.ParentJobID == jobid)]

        max_rss = df_steps_slice.MaxRSS.max()
        max_vmsize = df_steps_slice.MaxVMSize.max()
        max_diskread = df_steps_slice.MaxDiskRead.max()
        max_diskwrite = df_steps_slice.MaxDiskWrite.max()
        max_pages = df_steps_slice.MaxPages.max()

        ave_rss = df_steps_slice.apply(lambda row: row.AveRSS * row.NTasks, axis=1).sum()
        ave_vmsize = df_steps_slice.apply(lambda row: row.AveVMSize * row.NTasks, axis=1).sum()
        ave_diskread = df_steps_slice.apply(lambda row: row.AveDiskRead * row.NTasks, axis=1).sum()
        ave_diskwrite = df_steps_slice.apply(
            lambda row: row.AveDiskWrite * row.NTasks, axis=1
        ).sum()
        ave_pages = df_steps_slice.apply(lambda row: row.AvePages * row.NTasks, axis=1).sum()
        ave_cpu = df_steps_slice.apply(lambda row: row.AveCPU * row.NTasks, axis=1).sum()

        df_jobs.loc[(df_jobs.JobID == jobid), "MaxRSSJob"] = max_rss
        df_jobs.loc[(df_jobs.JobID == jobid), "AveRSSJob"] = ave_rss
        df_jobs.loc[(df_jobs.JobID == jobid), "MaxVMSizeJob"] = max_vmsize
        df_jobs.loc[(df_jobs.JobID == jobid), "AveVMSizeJob"] = ave_vmsize
        df_jobs.loc[(df_jobs.JobID == jobid), "MaxDiskReadJob"] = max_diskread
        df_jobs.loc[(df_jobs.JobID == jobid), "AveDiskReadJob"] = ave_diskread
        df_jobs.loc[(df_jobs.JobID == jobid), "MaxDiskWriteJob"] = max_diskwrite
        df_jobs.loc[(df_jobs.JobID == jobid), "AveDiskWriteJob"] = ave_diskwrite
        df_jobs.loc[(df_jobs.JobID == jobid), "MaxPagesJob"] = max_pages
        df_jobs.loc[(df_jobs.JobID == jobid), "AvePagesJob"] = ave_pages
        df_jobs.loc[(df_jobs.JobID == jobid), "AveCPUJob"] = ave_cpu

    df_jobs = df_jobs.drop(
        [
            "MaxRSS", "MaxVMSize", "MaxDiskWrite", "MaxDiskRead", "MaxPages", "AvePages", "NTasks",
            "AveDiskWrite", "AveDiskRead", "AveVMSize", "AveRSS", "AveCPU", "ConsumedEnergyRaw"
        ],
        axis=1
    )
    print(df_jobs.columns)
    pd.set_option('display.max_columns', 500)
    print(df_jobs)

    if no_nodelist:
        df_jobs = df_jobs.drop(["NodeList"], axis=1)
    else:
        # Convert node list to some metric related to network groups
        pass

    # TODO: remove any columns that are the same for all entries

    return df_jobs


def main(args):
    df = pd.read_csv(
        args.um_jobs, delimiter='|', lineterminator='\n', header=0
    ).dropna(how='all', axis=1) # ignore last empty column caused by trainling '|'

    df_jobs = clean(df)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("um_jobs", type=str)

    # parser.add_argument("--optional", type=, default=, help=)

    return parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments())


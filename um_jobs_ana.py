"""
"""

import argparse, subprocess, os

import pandas as pd
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

from funcs import timelimit_str_to_timedelta

pd.options.mode.chained_assignment = None

ARCHER_HSN = "/work/y02/shared/archer2_hsn/archer_hsn"
CACHE_DIR = "/work/y02/y02/awilkins/pandas_cache"
PLOT_DIR = "/work/y02/y02/awilkins/archer2_jobdata/plots"


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

    # Some ReqMem are 2**64, I am assuming this is an accounting error and they should just be the
    # same as the others
    df_jobs_reqmem_unique = df_jobs.ReqMem.unique()
    if len(df_jobs_reqmem_unique) == 2 and "18446744073709551616.00M" in df_jobs_reqmem_unique:
        if df_jobs_reqmem_unique[0] != "18446744073709551616.00M":
            valid_reqmem = df_jobs_reqmem_unique[0]
        else:
            valid_reqmem = df_jobs_reqmem_unique[1]
    df_jobs.ReqMem = df_jobs.ReqMem.apply(
        lambda row: valid_reqmem if row == "18446744073709551616.00M" else row
    )

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
    df_jobs["CPUUtil"] = df_jobs.apply(lambda row: row.TotalCPU / row.CPUTime, axis=1)

    df_jobs["MaxTaskRSS"] = np.nan
    df_jobs["TotJobRSS"] = np.nan
    df_jobs["MaxTaskVMSize"] = np.nan
    df_jobs["TotJobVMSize"] = np.nan
    df_jobs["MaxTaskDiskRead"] = np.nan
    df_jobs["TotJobDiskRead"] = np.nan
    df_jobs["MaxTaskDiskWrite"] = np.nan
    df_jobs["TotJobDiskWrite"] = np.nan
    df_jobs["MaxTaskPages"] = np.nan
    df_jobs["TotJobPages"] = np.nan

    for jobid in df_jobs.JobID.unique():
        df_steps_slice = df_steps.loc[(df_steps.ParentJobID == jobid)]

        max_rss = df_steps_slice.MaxRSS.max()
        max_vmsize = df_steps_slice.MaxVMSize.max()
        max_diskread = df_steps_slice.MaxDiskRead.max()
        max_diskwrite = df_steps_slice.MaxDiskWrite.max()
        max_pages = df_steps_slice.MaxPages.max()

        tot_rss = df_steps_slice.apply(lambda row: row.AveRSS * row.NTasks, axis=1).sum()
        tot_vmsize = df_steps_slice.apply(lambda row: row.AveVMSize * row.NTasks, axis=1).sum()
        tot_diskread = df_steps_slice.apply(lambda row: row.AveDiskRead * row.NTasks, axis=1).sum()
        tot_diskwrite = df_steps_slice.apply(
            lambda row: row.AveDiskWrite * row.NTasks, axis=1
        ).sum()
        tot_pages = df_steps_slice.apply(lambda row: row.AvePages * row.NTasks, axis=1).sum()

        df_jobs.loc[(df_jobs.JobID == jobid), "MaxTaskRSS"] = max_rss
        df_jobs.loc[(df_jobs.JobID == jobid), "TotJobRSS"] = tot_rss
        df_jobs.loc[(df_jobs.JobID == jobid), "MaxTaskVMSize"] = max_vmsize
        df_jobs.loc[(df_jobs.JobID == jobid), "TotJobVMSize"] = tot_vmsize
        df_jobs.loc[(df_jobs.JobID == jobid), "MaxTaskDiskRead"] = max_diskread
        df_jobs.loc[(df_jobs.JobID == jobid), "TotJobDiskRead"] = tot_diskread
        df_jobs.loc[(df_jobs.JobID == jobid), "MaxTaskDiskWrite"] = max_diskwrite
        df_jobs.loc[(df_jobs.JobID == jobid), "TotJobDiskWrite"] = tot_diskwrite
        df_jobs.loc[(df_jobs.JobID == jobid), "MaxTaskPages"] = max_pages
        df_jobs.loc[(df_jobs.JobID == jobid), "TotJobPages"] = tot_pages

    df_jobs["AveTaskDiskReadRate"] = df_jobs.apply(
        lambda row: row.TotJobDiskRead / row.Elapsed, axis=1
    )
    df_jobs["AveTaskDiskWriteRate"] = df_jobs.apply(
        lambda row: row.TotJobDiskWrite/ row.Elapsed, axis=1
    )
    df_jobs["AveTaskPageFaultRate"] = df_jobs.apply(
        lambda row: row.TotJobPages / row.Elapsed, axis=1
    )

    df_jobs = df_jobs.drop(
        [
            "MaxRSS", "MaxVMSize", "MaxDiskWrite", "MaxDiskRead", "MaxPages", "AvePages", "NTasks",
            "AveDiskWrite", "AveDiskRead", "AveVMSize", "AveRSS", "AveCPU", "ConsumedEnergyRaw",
            "CPUTime", "ExitCode", "NodeList", "Flags"
        ],
        axis=1
    )

    if not no_nodelist:
        df_jobs["SumSquaresNetworkGroup"] = np.nan
        df_jobs["NumNetworkGroups"] = np.nan

        grps = set()
        job_grp_cnts = {}

        input("About to run {}, press Enter to confirm".format(ARCHER_HSN))

        for jobid in tqdm(df_jobs.JobID.unique(), desc="Getting job network group counts..."):
            cmd_output = subprocess.check_output([ARCHER_HSN, "-g", str(jobid)]).decode('utf-8')
            cmd_output = cmd_output.split("\n")[1:-1]
            grp_cnts = [ " ".join(line.split()).split() for line in cmd_output ]

            df_jobs.loc[(df_jobs.JobID == jobid), "SumSquaresNetworkGroup"] = sum(
                [ int(grp_cnt[1])**2 for grp_cnt in grp_cnts ]
            )
            df_jobs.loc[(df_jobs.JobID == jobid), "NumNetworkGroups"] = len(grp_cnts)

            job_grp_cnts[jobid] = grp_cnts
            for grp_cnt in grp_cnts:
                grps.add(grp_cnt[0])

        for grp in grps:
            df_jobs[grp] = 0

        for jobid in df_jobs.JobID.unique():
            for grp_cnts in job_grp_cnts[jobid]:
                df_jobs.loc[(df_jobs.JobID == jobid), grp_cnts[0] ] = int(grp_cnts[1])

    df_jobs_nunique = df_jobs.nunique()
    print("Dropping constant {} columns: {}".format(
        len(df_jobs_nunique == 1), df_jobs_nunique[df_jobs_nunique == 1].index
    ))
    df_jobs = df_jobs.drop(df_jobs_nunique[df_jobs_nunique == 1].index, axis=1)

    print(df_jobs)

    return df_jobs


def main(args):
    if not args.cache or args.cache == "save":
        df = pd.read_csv(
            args.um_jobs, delimiter='|', lineterminator='\n', header=0
        ).dropna(how='all', axis=1) # ignore last empty column caused by trailing '|'
        df_jobs = clean(df)
    elif args.cache == "load":
        df_jobs = pd.read_pickle(os.path.join(CACHE_DIR, "um_jobs.pkl"))

    if args.cache == "save":
        df_jobs.to_pickle(os.path.join(CACHE_DIR, "um_jobs.pkl"))

    grps = [ str(grp) for grp in range(2, 48) ] 
    print("group\t samples")
    for grp in grps:
        print("{}\t{}".format(grp, len(df_jobs.loc[(df_jobs[grp] != 0)])))

    df_jobs_slice = df_jobs.loc[(df_jobs.State != "TIMEOUT")]
    for x in ["SumSquaresNetworkGroup", "TotJobDiskWrite", "TotJobDiskRead", "TotJobPages", "TotJobRSS", "NumNetworkGroups"]:
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.scatter(df_jobs_slice[x], df_jobs_slice.Elapsed, s=6)
        ax.set_ylabel("Elapsed")
        ax.set_xlabel(x)
        fig.tight_layout()
        plt.show()

    corr = df_jobs.drop(["JobID", "JobName", "State"] + grps, axis=1).corr(
        method="pearson", numeric_only=True
    )
    print(corr)

    bar_data = corr.Elapsed.drop("Elapsed").sort_values(ascending=False)
    x = np.array(range(len(bar_data)))
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    plt.bar(x, bar_data)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_data.index, fontsize=8, rotation=45)
    ax.set_ylabel("Correlation Coeff", fontsize=14)
    plt.title("Correlation of job data with Elapsed", fontsize=14)
    fig.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "um_jobs_ana_corr.pdf"))
    plt.show()

    corr_grps = df_jobs[grps + ["Elapsed"]].corr(
        method="pearson", numeric_only=True
    )
    print(corr_grps)

    bar_data = corr_grps.Elapsed.drop("Elapsed").sort_values(ascending=False)
    x = np.array(range(len(bar_data)))
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    plt.bar(x, bar_data)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_data.index, fontsize=8, rotation=45)
    ax.set_ylabel("Correlation Coeff", fontsize=14)
    plt.title("Correlation of job data with Elapsed", fontsize=14)
    fig.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "um_jobs_ana_corr_grps.pdf"))
    plt.show()

    print("\nRemoved TIMEOUTS\n")
    df_jobs_slice = df_jobs.loc[(df_jobs.State != "TIMEOUT")]
    corr_notimeout = df_jobs_slice.drop(["JobID", "JobName", "State"] + grps, axis=1).corr(
        method="pearson", numeric_only=True
    )
    print(corr_notimeout)

    bar_data = corr_notimeout.Elapsed.drop("Elapsed").sort_values(ascending=False)
    x = np.array(list(range(len(bar_data))))
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    plt.bar(x, bar_data)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_data.index, fontsize=8, rotation=45)
    ax.set_ylabel("Correlation Coeff", fontsize=14)
    plt.title("Correlation of job data with Elapsed (excluding TIMEOUT jobs)", fontsize=14)
    fig.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "um_jobs_ana_corr_notimeout.pdf"))
    plt.show()

    corr_notimeout_grps = df_jobs_slice[grps + ["Elapsed"]].corr(
        method="pearson", numeric_only=True
    )
    print(corr_notimeout_grps)

    bar_data = corr_notimeout_grps.Elapsed.drop("Elapsed").sort_values(ascending=False)
    x = np.array(range(len(bar_data)))
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    plt.bar(x, bar_data)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_data.index, fontsize=8, rotation=45)
    ax.set_ylabel("Correlation Coeff", fontsize=14)
    plt.title("Correlation of job data with Elapsed (excluding TIMEOUT)", fontsize=14)
    fig.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "um_jobs_ana_corr_notimeout_grps.pdf"))
    plt.show()

    df_jobs.State = df_jobs.State.apply(lambda row: 1 if row == "TIMEOUT" else 0)
    corr_with_timeout = df_jobs.drop(["JobID", "JobName", "Elapsed"] + grps, axis=1).corr(
        method="pearson", numeric_only=True
    )
    print(corr_with_timeout)

    bar_data = corr_with_timeout.State.drop("State").sort_values(ascending=False)
    x = np.array(range(len(bar_data)))
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    plt.bar(x, bar_data)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_data.index, fontsize=8, rotation=45)
    ax.set_ylabel("Correlation Coeff", fontsize=14)
    plt.title("Correlation of job data with TIMEOUT", fontsize=14)
    fig.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "um_jobs_ana_corr_with_timeout.pdf"))
    plt.show()

    corr_with_timeout_grps = df_jobs[grps + ["State"]].corr(
        method="pearson", numeric_only=True
    )
    print(corr_with_timeout_grps)

    bar_data = corr_with_timeout_grps.State.drop("State").sort_values(ascending=False)
    x = np.array(range(len(bar_data)))
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    plt.bar(x, bar_data)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_data.index, fontsize=8, rotation=45)
    ax.set_ylabel("Correlation Coeff", fontsize=14)
    plt.title("Correlation of job data with TIMEOUT", fontsize=14)
    fig.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "um_jobs_ana_corr_with_timeout_grps.pdf"))
    plt.show()

    # for name in ["ch330", "ck777", "ck778"]:
    #     print("\nOnly {} jobs\n".format(name))
    #     df_jobs_slice = df_jobs.loc[(df_jobs.JobName.str.contains(name))]
    #     print(df_jobs_slice.drop(["JobID", "JobName", "State"], axis=1).corr(method="pearson", numeric_only=True))


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("um_jobs", type=str)

    parser.add_argument("--cache", type=str, default="", help="(load|save)")

    return parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments())


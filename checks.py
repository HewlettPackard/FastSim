"""
Code to check some things from the sacct dump
"""

import os

import pandas as pd
import numpy as np

from funcs import process_power, timelimit_str_to_timedelta
from decision_trees import clean_df


SACCT_FILE="/work/y02/y02/awilkins/sacct_2022-10-01_2022-11-07.txt"

CPU_UTIL = False
CPU_STEP_CHECKS = False
STEP_AVE_CHECKS = True

def step_ave_checks(df):
    df = process_power(df, remove_steps=False)

    df_2390128 = df.loc[(df.JobID.str.contains("2390128"))]

    cols = [
    "ReqMem", "AveRSS", "AveVMSize", "AvePages", "NTasks", "MaxRSS", "MaxVMSize", "MaxPages"
    ]
    df_2390128[cols] = df_2390128[cols].replace(
            { "K" : "e+03", "M" : "e+06", "G" : "e+09", "T" : "e+12" }, regex=True
    )

    print(df_2390128[[
        "JobID", "AllocCPUS", "Elapsed", "ReqMem", "TotalCPU", "CPUTime", "NTasks", "AveCPU",
        "AvePages", "AveVMSize", "AveRSS", "MaxRSS", "MaxVMSize", "MaxPages"
    ]])

    df_2390128_steps = df_2390128.loc[(df_2390128.JobID.str.contains("\."))]

    df_2390128_steps.AveCPU = df_2390128_steps.AveCPU.apply(
        lambda row: timelimit_str_to_timedelta(row)
    )
    df_2390128_steps.TotalCPU = df_2390128_steps.TotalCPU.apply(
        lambda row: timelimit_str_to_timedelta(row)
    )
    df_2390128_steps.CPUTime = df_2390128_steps.CPUTime.apply(
        lambda row: timelimit_str_to_timedelta(row)
    )

    print("ReqMem: {} G".format(
        df_2390128.loc[(df_2390128.JobID == "2390128")].ReqMem.astype(float).astype(int).sum() /
        1e+9
    ))
    print("Sum of AveRSS * NTasks for steps: {} G".format(
        df_2390128_steps.loc[(df_2390128_steps.AveRSS.notna())].apply(
            lambda row: int(float(row.AveRSS) * float(row.NTasks)), axis=1
        ).sum() / 1e+9
    ))
    print("Sum of MaxRSS * NTasks for steps: {} G".format(
        df_2390128_steps.loc[(df_2390128_steps.MaxRSS.notna())].apply(
            lambda row: int(float(row.MaxRSS) * float(row.NTasks)), axis=1
        ).sum() / 1e+9
    ))
    print("Sum of AveVMSize * NTasks for steps: {} G".format(
        df_2390128_steps.loc[(df_2390128_steps.AveVMSize.notna())].apply(
            lambda row: int(float(row.AveVMSize) * float(row.NTasks)), axis=1
        ).sum() / 1e+9
    ))
    print("Sum of MaxVMSize * NTasks for steps: {} G".format(
        df_2390128_steps.loc[(df_2390128_steps.MaxVMSize.notna())].apply(
            lambda row: int(float(row.MaxVMSize) * float(row.NTasks)), axis=1
        ).sum() / 1e+9
    ))
    print("Sum of AvePages * NTasks for steps: {}".format(
        df_2390128_steps.loc[(df_2390128_steps.AvePages.notna())].apply(
            lambda row: int(float(row.AvePages) * float(row.NTasks)), axis=1
        ).sum()
    ))
    print("Sum of MaxPages * NTasks for steps: {}".format(
        df_2390128_steps.loc[(df_2390128_steps.MaxPages.notna())].apply(
            lambda row: int(float(row.MaxPages) * float(row.NTasks)), axis=1
        ).sum()
    ))

    print("TotalCPU {} CPUTime {}".format(
        df_2390128.loc[(df_2390128.JobID == "2390128")].TotalCPU.sum(),
        df_2390128.loc[(df_2390128.JobID == "2390128")].CPUTime.sum()
    ))
    print("Sum of AveCPU * NTasks for steps {}".format(
        df_2390128_steps.loc[(df_2390128_steps.AveCPU.notna())].apply(
            lambda row: row.AveCPU * int(float(row.NTasks)), axis=1
        ).sum()
    ))

    print("\n")

    df_2390372 = df.loc[(df.JobID.str.contains("2390372"))]

    cols = [
    "ReqMem", "AveRSS", "AveVMSize", "AvePages", "NTasks", "MaxRSS", "MaxVMSize", "MaxPages"
    ]
    df_2390372[cols] = df_2390372[cols].replace(
            { "K" : "e+03", "M" : "e+06", "G" : "e+09", "T" : "e+12" }, regex=True
    )

    print(df_2390372[[
        "JobID", "AllocCPUS", "Elapsed", "ReqMem", "TotalCPU", "CPUTime", "NTasks", "AveCPU",
        "AvePages", "AveVMSize", "AveRSS", "MaxRSS", "MaxVMSize", "MaxPages"
    ]])

    df_2390372_steps = df_2390372.loc[(df_2390372.JobID.str.contains("\."))]

    df_2390372_steps.AveCPU = df_2390372_steps.AveCPU.apply(
        lambda row: timelimit_str_to_timedelta(row)
    )
    df_2390372_steps.TotalCPU = df_2390372_steps.TotalCPU.apply(
        lambda row: timelimit_str_to_timedelta(row)
    )
    df_2390372_steps.CPUTime = df_2390372_steps.CPUTime.apply(
        lambda row: timelimit_str_to_timedelta(row)
    )

    print("ReqMem: {} G".format(
        df_2390372.loc[(df_2390372.JobID == "2390372")].ReqMem.astype(float).astype(int).sum() /
        1e+9
    ))
    print("Sum of AveRSS * NTasks for steps: {} G".format(
        df_2390372_steps.loc[(df_2390372_steps.AveRSS.notna())].apply(
            lambda row: int(float(row.AveRSS) * float(row.NTasks)), axis=1
        ).sum() / 1e+9
    ))
    print("Sum of MaxRSS * NTasks for steps: {} G".format(
        df_2390372_steps.loc[(df_2390372_steps.MaxRSS.notna())].apply(
            lambda row: int(float(row.MaxRSS) * float(row.NTasks)), axis=1
        ).sum() / 1e+9
    ))
    print("Sum of AveVMSize * NTasks for steps: {} G".format(
        df_2390372_steps.loc[(df_2390372_steps.AveVMSize.notna())].apply(
            lambda row: int(float(row.AveVMSize) * float(row.NTasks)), axis=1
        ).sum() / 1e+9
    ))
    print("Sum of MaxVMSize * NTasks for steps: {} G".format(
        df_2390372_steps.loc[(df_2390372_steps.MaxVMSize.notna())].apply(
            lambda row: int(float(row.MaxVMSize) * float(row.NTasks)), axis=1
        ).sum() / 1e+9
    ))
    print("Sum of AvePages * NTasks for steps: {}".format(
        df_2390372_steps.loc[(df_2390372_steps.AvePages.notna())].apply(
            lambda row: int(float(row.AvePages) * float(row.NTasks)), axis=1
        ).sum()
    ))
    print("Sum of MaxPages * NTasks for steps: {}".format(
        df_2390372_steps.loc[(df_2390372_steps.MaxPages.notna())].apply(
            lambda row: int(float(row.MaxPages) * float(row.NTasks)), axis=1
        ).sum()
    ))

    print("TotalCPU {} CPUTime {}".format(
        df_2390372.loc[(df_2390372.JobID == "2390372")].TotalCPU.sum(),
        df_2390372.loc[(df_2390372.JobID == "2390372")].CPUTime.sum()
    ))
    print("Sum of AveCPU * NTasks for steps {}".format(
        df_2390372_steps.loc[(df_2390372_steps.AveCPU.notna())].apply(
            lambda row: row.AveCPU * int(float(row.NTasks)), axis=1
        ).sum()
    ))


def cpu_step_checks(df):
    df = process_power(df, remove_steps=False)

    # print(df[["JobID", "AllocCPUS", "Elapsed", "TotalCPU"]][7000:7050])

    df_2390128 = df.loc[(df.JobID.str.contains("2390128"))]

    print(df_2390128[["JobID", "AllocCPUS", "Elapsed", "TotalCPU"]])

    df_2390128.TotalCPU = df_2390128.TotalCPU.apply(lambda row : timelimit_str_to_timedelta(row))

    df_2390128_steps = df_2390128.loc[(df_2390128.JobID.str.contains("\."))]

    print(df_2390128_steps.TotalCPU.sum())
    print(df_2390128.loc[(df_2390128.JobID == "2390128")].TotalCPU.sum())

    print("\n")

    df_2390372 = df.loc[(df.JobID.str.contains("2390372"))]

    print(df_2390372[["JobID", "AllocCPUS", "Elapsed", "TotalCPU"]])

    df_2390372.TotalCPU = df_2390372.TotalCPU.apply(lambda row : timelimit_str_to_timedelta(row))

    df_2390372_steps = df_2390372.loc[(df_2390372.JobID.str.contains("\."))]

    print(df_2390372_steps.TotalCPU.sum())
    print(df_2390372.loc[(df_2390372.JobID == "2390372")].TotalCPU.sum())


def cpu_util(df):
    df_power_nosteps = process_power(df, remove_steps=True)

    df_power_nosteps = clean_df(df_power_nosteps, queue_only=True)

    df_power_nosteps = df_power_nosteps.loc[(df_power_nosteps.TotalCPU.notna())]

    df_power_nosteps.TotalCPU = df_power_nosteps.TotalCPU.apply(
        lambda row: round(timelimit_str_to_timedelta(row).total_seconds())
    )
    df_power_nosteps.CPUTime = df_power_nosteps.CPUTime.apply(
        lambda row: round(timelimit_str_to_timedelta(row).total_seconds())
    )
    df_power_nosteps["CPUUtil"] = df_power_nosteps.apply(
        lambda row: row.TotalCPU/row.CPUTime, axis=1
    )

    print(df_power_nosteps[
        ["JobID", "Elapsed", "AllocCPUS", "TotalCPU", "CPUTime", "CPUUtil", "PowerPerNode"]
    ][2000:2050])
    print(df_power_nosteps.CPUUtil.sort_values().unique())

if __name__ == "__main__":
    df = pd.read_csv(SACCT_FILE, delimiter="|", header=0, usecols=range(70), nrows=50000)

    if CPU_UTIL:
        cpu_util(df)

    if CPU_STEP_CHECKS:
        cpu_step_checks(df)

    if STEP_AVE_CHECKS:
        step_ave_checks(df)


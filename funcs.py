"""
Helper functions
"""

import os, datetime

import pandas as pd


def mkdir_p(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        return dir
    return False


def power_print_dump(df_power):
    print("\nFor jobs started after {start} and ending before {end} ({duration}):\n".format(
        start=min(df_power.Start), end=max(df_power.End),
        duration=max(df_power.End) - min(df_power.Start)
    ))
    print("Total consumed energy {} J".format(df_power.ConsumedEnergyRaw.sum()))
    print("Mean power usage for system {} W".format(
        (
            df_power.ConsumedEnergyRaw.sum() /
            (max(df_power.End) - min(df_power.Start)).total_seconds()
        )
    ))
    print("Max|Mean|Min job power = {} | {} | {} W\n".format(
        max(df_power.Power), df_power.Power.mean(), min(df_power.Power)
    ))


def parse_cache(df, cache, data_name, df_name, cols, remove_steps=True):
    if isinstance(df, str) and cache != "load": # filepath
        df = pd.read_csv(
            df,
            delimiter='|',
            header=0,
            usecols=cols,
        )

    mkdir_p("/work/y02/y02/awilkins/pandas_cache/{}".format(data_name))
    if cache == "save":
        df_power = process_power(df, cols=cols, remove_steps=remove_steps)
        df_power.to_pickle(
            "/work/y02/y02/awilkins/pandas_cache/{}/{}.pkl".format(data_name, df_name)
        )
    elif cache == "load":
        df_power = pd.read_pickle(
            "/work/y02/y02/awilkins/pandas_cache/{}/{}.pkl".format(data_name, df_name)
        )
    else:
        df_power = process_power(df, cols=cols, remove_steps=remove_steps)

    return df_power


def process_power(df, cols=None, remove_steps=True):
    if cols:
        df_power = df.loc[
            (
                (df.Start != "Unknown") & (df.Start.notna() &
                (df.End != "Unknown") & (df.End.notna())) &
                (df.ConsumedEnergyRaw != "") & (df.ConsumedEnergyRaw.notna())
            ),
            cols
        ]
    else:
        df_power = df.loc[
            (
                (df.Start != "Unknown") & (df.Start.notna() &
                (df.End != "Unknown") & (df.End.notna())) &
                (df.ConsumedEnergyRaw != "") & (df.ConsumedEnergyRaw.notna())
            )
        ]

    if remove_steps:
        df_power = df_power[
            df_power.apply(lambda row: len(str(row.JobID).split(".")) == 1, axis=1)
        ]

    df_power.ConsumedEnergyRaw = pd.to_numeric(df_power.ConsumedEnergyRaw)
    df_power = df_power.loc[df_power.ConsumedEnergyRaw > 0]

    df_power.Start = pd.to_datetime(df_power.Start, format="%Y-%m-%dT%H:%M:%S")
    df_power.End = pd.to_datetime(df_power.End, format="%Y-%m-%dT%H:%M:%S")

    df_power["DeltaT"] = df_power.apply(lambda row: (row.End - row.Start).total_seconds(), axis=1)
    df_power = df_power.loc[df_power.DeltaT > 0]

    df_power["Power"] = df_power.apply(
        lambda row: float(row.ConsumedEnergyRaw)/float(row.DeltaT), axis=1
    )
    # Sometimes ConsumedEnergyRaw (very rare) is obviously wrong causing non-physical powers
    print("Anomalous rows removed:")
    print(df_power.loc[df_power.Power >= 10000000])
    df_power = df_power.loc[df_power.Power < 10000000]

    # There are some short jobs that get duplicated a few hundred times in the slurm data
    df_power = df_power[~df_power.duplicated(subset="JobID", keep="first")]

    return df_power


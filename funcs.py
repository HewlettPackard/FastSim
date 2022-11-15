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


def timelimit_str_to_timedelta(t_str):
    days, hrs = 0, 0
    try:
        if "-" in t_str:
            days = int(t_str.split("-")[0])
            t_str = t_str.split("-")[1]
    except:
        print(t_str)

    if t_str.count(":") == 1 and t_str.count("."): # MM:SS.SS
        mins, secs = t_str.split(":")
        mins = int(mins)
        secs = float(secs)
    elif t_str.count(":") == 2: ## HH:MM:SS (SS has no decimal place for these ones)
        hrs, mins, secs = map(int, t_str.split(":"))
    else:
        raise NotImplementedError("Bruh")

    return datetime.timedelta(days=days, hours=hrs, minutes=mins, seconds=secs)


def hour_to_timeofday(hr):
    if hr > 0 and hr <= 4:
        return "late night"
    elif hr > 4 and hr <= 8:
        return "early morning"
    elif hr > 8 and hr <= 12:
        return "morning"
    elif hr > 12 and hr <= 16:
        return "afternoon"
    elif hr > 16 and hr <= 20:
        return "evening"
    else: # > 20 and before or equal to 12am
        return "night"

    print(hr)

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


def parse_cache(df, cache, data_name, df_name, cols, remove_steps=True, nrows=None):
    # Ew
    has_partition = "Partition" in cols
    cols += ["Partition"] if not has_partition else []

    if isinstance(df, str) and cache != "load": # filepath
        df = pd.read_csv(
            df,
            delimiter='|',
            header=0,
            usecols=cols,
            nrows=nrows
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
        return df_power
    else:
        df_power = process_power(df, cols=cols, remove_steps=remove_steps)

    return df_power.drop(["Partition"], axis=1) if not has_partition else df_power


def process_power(df, cols=None, remove_steps=True):
    if cols:
        # Partition slice is removing data analysis nodes
        # (note this will leave job steps without a parent)
        df_power = df.loc[
            (
                (df.Start != "Unknown") & (df.Start.notna() &
                (df.End != "Unknown") & (df.End.notna())) &
                (df.ConsumedEnergyRaw != "") & (df.ConsumedEnergyRaw.notna()) &
                (df.Partition != "serial")
            ),
            cols
        ]
    else:
        df_power = df.loc[
            (
                (df.Start != "Unknown") & (df.Start.notna() &
                (df.End != "Unknown") & (df.End.notna())) &
                (df.ConsumedEnergyRaw != "") & (df.ConsumedEnergyRaw.notna()) &
                (df.Partition != "serial")
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


import os, sys
from typing import Union
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from classes import Queue

sys.path.append("/work/y02/y02/awilkins/archer2_jobdata")
from funcs import parse_cache, timelimit_str_to_timedelta, hour_to_timeofday


""" Helpers """

def convert_to_raw(df, cols):
    df[cols] = df[cols].astype(str)
    df[cols] = df[cols].replace(
        { "K" : "e+03", "M" : "e+06", "G" : "e+09", "T" : "e+12" }, regex=True
    ).astype(float).astype(int)
    return df


def model_power_predictions(model, df):
    df = df.copy()
    convert_to_raw(df, ["ReqCPUS", "ReqNodes", "ReqMem"])
    df.Submit = pd.to_datetime(df.Submit, format="%Y-%m-%dT%H:%M:%S")
    df.Submit = df.Submit.apply(lambda row: hour_to_timeofday(row.hour))
    df.Timelimit = df.Timelimit.apply(
        lambda row: round(timelimit_str_to_timedelta(row).total_seconds() / 60)
    )
    return model.predict(df) # NOTE: ColumnTransformer ignores columns it wasn't fit with

""" End Helpers """


def prep_job_data(data, cache, df_name, model, rows=None):
    df_jobs = parse_cache(
        data, cache, ".".join(os.path.basename(data).split(".")[:-1]), df_name, 
        [
            "JobID", "Start", "End", "Submit", "Elapsed", "ConsumedEnergyRaw", "AllocNodes",
            "Timelimit", "ReqCPUS", "ReqNodes", "Group", "QOS", "ReqMem"
        ],
        nrows=rows
    )

    convert_to_raw(df, "AllocNodes")

    if model:
        df_jobs["PowerPerNode"] = model_power_predictions(model, df)
    else:
        df_jobs["PowerPerNode"] = df_jobs.apply(
            lambda row: float(row.Power) / float(row.AllocNodes), axis=1
        )
    df_jobs["TruePowerPerNode"] = df_jobs.apply(
        lambda row: float(row.Power) / float(row.AllocNodes), axis=1
    )


    df_jobs.Elapsed = df_jobs.Elapsed.apply(lambda row: timelimit_str_to_timedelta(row))
    df_jobs.Timelimit = df_jobs.Timelimit.apply(lambda row: timelimit_str_to_timedelta(row))
    df_jobs.Submit = pd.to_datetime(df_jobs.Submit, format="%Y-%m-%dT%H:%M:%S")

    df_jobs = df_jobs.drop(["ReqCPUS", "ReqNodes", "Group", "QOS", "ReqMem"], axis=1)

    return df_jobs


def run_sim(
    df_jobs, system, scheduler, t0, seed=None, verbose=False, min_step=timedelta(seconds=10),
    custom_low_or_high=None
):
    queue = Queue(df_jobs, scheduler, t0, custom_low_or_high=custom_low_or_high)

    np.random.seed(seed)

    cnt = 0
    time = t0

    while queue.all_jobs or queue.queue or system.running_jobs:
        # Not enough precision to compute timedeltas with datetime.max
        try:
            t_step = max(min(queue.next_newjob() - time, system.next_event() - time), min_step)
        except pd._libs.tslibs.np_datetime.OutOfBoundsTimedelta:
            if queue.next_newjob() == datetime.max:
                t_step = system.next_event() - time
            else:
                t_step = queue.next_newjob() - time
        # Enforce minimum step to reduce simulation time
        if t_step < min_step:
            t_step = min_step
        time += t_step

        system.step(t_step)
        queue.step(t_step, 1 if queue.queue else None)

        system.submit_jobs(queue)

        # Print every 3 hours at most
        if verbose and (time.hour != (time - t_step).hour and not time.hour % 3):
            print(
                "{} (step {}):\n".format(time, cnt) +
                "Utilisation = {:.2f}%\tNodesDrained = {}({})\tPower = {:.4f} MW\tQueueSize = {}\t\
                 RunningJobs = {}".format(
                    system.occupancy_history[-1] * 100, system.nodes_drained,
                    system.nodes_drained_carryover, system.power_usage, len(queue.queue),
                    len(system.running_jobs)
                )
            )

        cnt += 1

    return system

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
            "Timelimit", "ReqCPUS", "ReqNodes", "Group", "QOS", "ReqMem", "User", "Account",
            "Partition"
        ],
        nrows=rows
    )

    convert_to_raw(df_jobs, "AllocNodes")

    if model:
        df_jobs["PowerPerNode"] = model_power_predictions(model, df_jobs)
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

    df_jobs = df_jobs.drop(["ReqCPUS", "ReqNodes", "Group", "ReqMem"], axis=1)

    # Some error in slurm accounting, can correct for case of one other user in account
    for i, anomalous_row in df_jobs.loc[(df_jobs.User == "00:00:00")].iterrows():
        acc_users = df_jobs.loc[(df_jobs.Account == anomalous_row.Account)].User.unique()
        if len(acc_users) == 2:
            df_jobs.at[i, "User"] = acc_users[1] if acc_users[0] == "00:00:00" else acc_users[0]

    return df_jobs


def run_sim(
    df_jobs, system, t0, priority_sorter, seed=None, verbose=False, min_step=timedelta(seconds=10),
    no_retained=False, mf_priority_calc_step=False
):
    queue = Queue(df_jobs, t0, priority_sorter)

    np.random.seed(seed)

    cnt = 0
    time = t0

    if not no_retained:
        num_retained = lambda queue: 1 if queue.queue else None
    else:
        num_retained = lambda queue: None

    while queue.all_jobs or queue.queue or system.running_jobs:
        # Not enough precision to compute timedeltas with datetime.max
        try:
            t_step = max(min(queue.next_newjob() - time, system.next_event() - time), min_step)
        except pd._libs.tslibs.np_datetime.OutOfBoundsTimedelta:
            if queue.next_newjob() == datetime.max:
                t_step = max(system.next_event() - time, min_step)
            else:
                t_step = max(queue.next_newjob() - time, min_step)

        # Fair tree can ignore min step
        if mf_priority_calc_step:
            next_calc = priority_sorter.fairtree.next_calc() - time
            if next_calc <= t_step:
                t_step = next_calc
                priority_sorter.fairtree.fairshare_calc(system.running_jobs, time + t_step)

        time += t_step

        finished_jobs = system.step(t_step)
        if mf_priority_calc_step:
            for job in finished_jobs:
                priority_sorter.fairtree.job_finish_usage_update(job)

        queue.step(t_step, num_retained(queue))

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

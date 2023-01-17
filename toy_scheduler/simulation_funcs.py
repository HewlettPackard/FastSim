import os, sys
from collections import defaultdict
from typing import Union
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from classes import Queue, Dependency, Partition, Node
from globals import *

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

def get_dependency_arg(submit_line):
    words = submit_line.split(" ")
    dep_arg = None
    for i_last_word, word in enumerate(words[1:]):
        # Batch script or executable marks end of options
        if word[0] != "-" and (words[i_last_word][0] != "-" or "=" in words[i_last_word]):
            break
        if "--dependency=" in word:
            dep_arg = word.split("--dependency=")[1]
            break
        if word == "-d":
            dep_arg = words[i_last_word + 2]
            break

    return dep_arg

def get_nodes_and_partitions(node_events_dump):
    df_events = pd.read_csv(
        node_events_dump, delimiter='|', lineterminator='\n', header=0,
        usecols=["NodeName", "TimeStart", "TimeEnd", "State"]
    )
    df_events = df_events.loc[
        (
            (df_events.NodeName.notna()) & (df_events.NodeName.str.contains("nid")) &
            (df_events.TimeEnd != "Unknown") & (df_events.TimeStart != "Unknown")
        )
    ]

    df_events.TimeStart = pd.to_datetime(df_events.TimeStart, format="%Y-%m-%dT%H:%M:%S")
    df_events.TimeEnd = pd.to_datetime(df_events.TimeEnd, format="%Y-%m-%dT%H:%M:%S")
    df_events["Duration"] = df_events.apply(lambda row: (row.TimeEnd - row.TimeStart), axis=1)
    df_events.State = df_events.State.apply(lambda row: "DRAIN" if "DRAIN" in row else "DOWN")
    df_events["Id"] = df_events.NodeName.apply(lambda row: int(row.split("nid")[1]))

    partitions = {
        "standard" : Partition("standard", 1, 1.0), "highmem" : Partition("highmem", 1, 1.0)
    }
    nodes = []
    for nid in range(1000, 6860):
        node_down_schedule = []
        for _, row in df_events.loc[(df_events.Id == nid)].iterrows():
            node_down_schedule.append([row.TimeStart, row.Duration, row.State])
        node_down_schedule.sort(key=lambda schedule: schedule[0])

        # Merge any adjacent events
        i_event = 0
        while i_event < len(node_down_schedule) - 1:
            event, next_event = node_down_schedule[i_event], node_down_schedule[i_event + 1]
            if event[0] + event[1] == next_event[0] and event[2] == next_event[2]:
                node_down_schedule[i_event][1] += next_event[1]
                node_down_schedule.pop(i_event + 1)
                continue
            node_down_schedule[i_event] = tuple(event)
            i_event += 1
        for i_event in range(len(node_down_schedule)):
            node_down_schedule[-1] = tuple(node_down_schedule[-1])

        if (2756 <= nid <= 3047) or (6376 <= nid <= 6667):
            nodes.append(Node(nid, 1000, node_down_schedule=node_down_schedule))
            partitions["highmem"].add_node(nodes[-1])
        else:
            nodes.append(Node(nid, 0, node_down_schedule=node_down_schedule))
        partitions["standard"].add_node(nodes[-1])

    return nodes, partitions

""" End Helpers """


def prep_job_data(data, cache, df_name, model, rows=None):
    df_jobs = parse_cache(
        data, cache, ".".join(os.path.basename(data).split(".")[:-1]), df_name,
        [
            "JobID", "Start", "End", "Submit", "Elapsed", "ConsumedEnergyRaw", "AllocNodes",
            "Timelimit", "ReqCPUS", "ReqNodes", "Group", "QOS", "ReqMem", "User", "Account",
            "Partition", "SubmitLine", "JobName", "Reason", "State"
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

    df_jobs["DependencyArg"] = df_jobs.SubmitLine.apply(lambda row: get_dependency_arg(row))

    df_jobs = df_jobs.drop(["ReqCPUS", "ReqNodes", "Group", "ReqMem", "SubmitLine"], axis=1)

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
    ns, ps = get_nodes_and_partitions(NODE_EVENTS_FILE)
    system.set_nodes_partitions(ns, ps)
    print(len(system.node_down_order))
    print(len(system.down_nodes))

    queue = Queue(df_jobs, t0, priority_sorter)
    queue.set_system(system)

    np.random.seed(seed)

    cnt = 0
    time = t0

    if not no_retained:
        num_retained = lambda queue: 1 if queue.queue else None
    else:
        num_retained = lambda queue: None

    while queue.all_jobs or queue.queue or system.running_jobs or queue.waiting_dependency:
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

        # Print every third hour
        if verbose and (time.hour != (time - t_step).hour and not time.hour % 3):
            print(
                "{} (step {}):\n".format(time, cnt) +
                "Utilisation = {:.2f}% (highmem {:.2f}%)\tNodesDown = {}\t\
                 Power = {:.4f} MW\n".format(
                    system.occupancy_history[-1] * 100,
                    sum(
                        1 / 584 for job in system.running_jobs for node in job.assigned_nodes if (
                            "highmem" in [ partition.name for partition in node.partitions ]
                        )
                    ) * 100,
                    sum(1 for node in system.down_nodes if not node.free), system.power_usage
                ) +
                "QueueSize = {} (held by priority {} (highmem {}) ".format(
                    (
                        len(queue.queue) +
                        len(queue.waiting_dependency) +
                        sum(len(jobs) for jobs in queue.qos_held.values())
                    ),
                    len(queue.queue), sum(1 for job in queue.queue if job.partition == "highmem")
                ) +
                "dependency {} qos {} (".format(
                    len(queue.waiting_dependency),
                    sum(len(jobs) for jobs in queue.qos_held.values())
                ) +
                " ".join(
                    "{}={}".format(
                        qos.name, len(jobs)
                    ) for qos, jobs in queue.qos_held.items() if len(jobs)
                ) +
                "))\tRunningJobs = {}\n".format(len(system.running_jobs))
            )

        cnt += 1

    return system

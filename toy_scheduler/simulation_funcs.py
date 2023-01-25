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

def get_reservation_arg(submit_line):
    words = submit_line.split(" ")
    res_arg = ""
    for i_last_word, word in enumerate(words[1:]):
        # Batch script or executable marks end of options
        if word[0] != "-" and (words[i_last_word][0] != "-" or "=" in words[i_last_word]):
            break
        if "--reservation=" in word:
            res_arg = word.split("--reservation=")[1]
            break

    return res_arg

def get_begin_arg(submit_line):
    words = submit_line.split(" ")
    begin_arg = None
    for i_last_word, word in enumerate(words[1:]):
        # Batch script or executable marks end of options
        if word[0] != "-" and (words[i_last_word][0] != "-" or "=" in words[i_last_word]):
            break
        if "--begin=" in word:
            begin_arg = word.split("--begin=")[1]
            break
        if word == "-b":
            begin_arg = words[i_last_word + 2]
            break

    return begin_arg

def convert_nodelist_to_node_nums(nid_str):
    node_nums = []

    nid_str = nid_str.strip("nid").strip("[").strip("]")
    for nid_str_entry in nid_str.split(","):
        if "-" not in nid_str_entry:
            node_nums.append(int(nid_str_entry))
            continue

        nid_str_range = nid_str_entry.split("-")
        for node_num in range(int(nid_str_range[0]), int(nid_str_range[1]) + 1):
            node_nums.append(node_num)

    return node_nums

def get_nodes_and_partitions(node_events_dump, reservations_dump):
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

    # Basic reservations implementation, only consider reservations that are still in the database
    # (this looks sufficient for ARCHER2) and ignore any flags such as REPLACE_DOWN
    df_reservations = pd.read_csv(
        reservations_dump, delimiter='|', lineterminator='\n', header=0,
        usecols=["RESV_NAME", "STATE", "START_TIME", "END_TIME", "NODELIST"]
    )
    df_reservations = df_reservations.loc[(df_reservations.STATE == "ACTIVE")]

    df_reservations.START_TIME = pd.to_datetime(
        df_reservations.START_TIME, format="%Y-%m-%dT%H:%M:%S"
    )
    df_reservations.END_TIME = pd.to_datetime(df_reservations.END_TIME, format="%Y-%m-%dT%H:%M:%S")
    df_reservations.NODELIST = df_reservations.NODELIST.apply(convert_nodelist_to_node_nums)

    valid_reservations = [ row.RESV_NAME for _, row in df_reservations.iterrows() ]

    df_reservations = df_reservations.explode("NODELIST")

    partitions = {
        "standard" : Partition("standard", 1, 1.0), "highmem" : Partition("highmem", 1, 1.0)
    }
    nodes = []
    for nid in range(1000, 6860):
        down_schedule = []
        for _, row in df_events.loc[(df_events.Id == nid)].iterrows():
            down_schedule.append([row.TimeStart, row.Duration, row.State])
        down_schedule.sort(key=lambda schedule: schedule[0])

        reservation_schedule = []
        for _, row in df_reservations.loc[(df_reservations.NODELIST == nid)].iterrows():
            reservation_schedule.append((row.START_TIME, row.END_TIME, row.RESV_NAME))
        reservation_schedule.sort(key=lambda schedule: schedule[0])

        # Merge any adjacent events
        i_event = 0
        while i_event < len(down_schedule) - 1:
            event, next_event = down_schedule[i_event], down_schedule[i_event + 1]
            if event[0] + event[1] == next_event[0] and event[2] == next_event[2]:
                down_schedule[i_event][1] += next_event[1]
                down_schedule.pop(i_event + 1)
                continue
            i_event += 1

        if (2756 <= nid <= 3047) or (6376 <= nid <= 6667):
            nodes.append(
                Node(
                    nid, 1000, down_schedule=down_schedule,
                    reservation_schedule=reservation_schedule
                )
            )
            partitions["highmem"].add_node(nodes[-1])
        else:
            nodes.append(
                Node(
                    nid, 0, down_schedule=down_schedule, reservation_schedule=reservation_schedule
                )
            )
        partitions["standard"].add_node(nodes[-1])

    return nodes, partitions, valid_reservations

""" End Helpers """


def prep_job_data(data, cache, df_name, model, rows=None):
    df_jobs = parse_cache(
        data, cache, ".".join(os.path.basename(data).split(".")[:-1]), df_name,
        [
            "JobID", "Start", "End", "Submit", "Elapsed", "ConsumedEnergyRaw", "AllocNodes",
            "Timelimit", "ReqCPUS", "ReqNodes", "Group", "QOS", "ReqMem", "User", "Account",
            "Partition", "SubmitLine", "JobName", "Reason", "State"
        ],
        nrows=rows, fix_anomalous_powers=True
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

    # df_jobs.Elapsed = df_jobs.Elapsed.apply(lambda row: timelimit_str_to_timedelta(row))
    # A small number of jobs have elapsed > end - start, think end - start is more reliable
    df_jobs.Elapsed = df_jobs.End - df_jobs.Start
    df_jobs.Timelimit = df_jobs.Timelimit.apply(lambda row: timelimit_str_to_timedelta(row))
    df_jobs.Submit = pd.to_datetime(df_jobs.Submit, format="%Y-%m-%dT%H:%M:%S")

    df_jobs["DependencyArg"] = df_jobs.SubmitLine.apply(lambda row: get_dependency_arg(row))
    df_jobs["ReservationArg"] = df_jobs.SubmitLine.apply(lambda row: get_reservation_arg(row))
    df_jobs["BeginArg"] = df_jobs.SubmitLine.apply(lambda row: get_begin_arg(row))

    # Think I need to do this for dependencies
    print("{} hetrogeneous JobIDs converted to regular JobIDs".format(
        len(df_jobs.loc[(df_jobs.JobID.str.contains("+", regex=False))])
    ))
    df_jobs.JobID = df_jobs.JobID.apply(
        lambda row: str(int(row.split("+")[0]) + int(row.split("+")[1])) if "+" in row else row
    )

    df_jobs = df_jobs.drop(["ReqCPUS", "ReqNodes", "Group", "ReqMem", "SubmitLine"], axis=1)

    # Some error in slurm accounting, can correct for case of one other user in account
    for i, anomalous_row in df_jobs.loc[(df_jobs.User == "00:00:00")].iterrows():
        acc_users = df_jobs.loc[(df_jobs.Account == anomalous_row.Account)].User.unique()
        if len(acc_users) == 2:
            df_jobs.at[i, "User"] = acc_users[1] if acc_users[0] == "00:00:00" else acc_users[0]

    return df_jobs


def run_sim(
    df_jobs, system, t0, priority_sorter, seed=None, verbose=False, no_retained=False,
    fairtree_interval=None, sched_interval=None, bf_interval=None
):
    nodes, partitions, valid_reservations = get_nodes_and_partitions(
        NODE_EVENTS_FILE, RESERVATIONS_FILE
    )
    system.set_nodes_partitions(nodes, partitions)

    queue = Queue(df_jobs, t0, priority_sorter, valid_reservations=valid_reservations)
    queue.set_system(system)

    np.random.seed(seed)

    cnt = 0
    time = t0

    if not no_retained:
        num_retained = lambda queue: 1 if queue.queue else None
    else:
        num_retained = lambda queue: None

    previous_hour = t0.hour
    previous_small_sched = t0
    next_bf_time = t0 + bf_interval
    next_sched_time = t0 + sched_interval
    next_fairtree_time = t0 + fairtree_interval
    while queue.all_jobs or queue.queue or system.running_jobs or queue.waiting_dependency:
        # No event by event scheduling
        if DEFER:
            time = min(
                next_bf_time, next_sched_time, next_fairtree_time, system.next_event(),
                queue.next_newjob()
            )

            small_sched_possible = (
                True if not SMALL_SCHED_OFF and time == system.next_event() else False
            )

            # NOTE There should be a notion of a main scheduler every sched_interval time that
            # checks all jobs and then a quick one that checks the first 100 (configurable) but for
            # ARCHER with two partitions I think they are equivalent

            # Still need to end jobs and collect new jobs at the actual time even if scheduling is
            # no longer event based
            finished_jobs = system.step(time)
            for job in finished_jobs:
                priority_sorter.fairtree.job_finish_usage_update(job)
            # Need to update dependencies even if there are no new submissions
            # NOTE submitted_jobs_step and finished_jobs_step are being cleared after check
            # dependencies since this is the only place they are being used and I dont want
            # to keep checking the dependencies with the same jobs
            queue.step(time, num_retained(queue))

            if time == next_fairtree_time:
                priority_sorter.fairtree.fairshare_calc(system.running_jobs, time)
                next_fairtree_time += fairtree_interval

            if time == next_sched_time:
                sched = True
                next_sched_time += sched_interval
            elif small_sched_possible and time > previous_small_sched + SCHED_MIN_INTERVAL:
                sched = True
                previous_small_sched = time
            else:
                sched = False

            if time == next_bf_time:
                backfill = True
                next_bf_time += bf_interval
                # sched is a subset of what backfill does so may as well do a pass here to see if
                # we can save some computation
                sched = True
            else:
                backfill = False

            system.submit_jobs(queue, sched=sched, backfill=backfill)

        else:
            # Not enough precision to compute timedeltas with datetime.max
            time = min(queue.next_newjob(), system.next_event(), next_fairtree_time)

            if time == next_fairtree_time:
                priority_sorter.fairtree.fairshare_calc(system.running_jobs, time)
                next_fairtree_time += fairtree_interval
                continue

            finished_jobs = system.step(time)
            for job in finished_jobs:
                priority_sorter.fairtree.job_finish_usage_update(job)

            queue.step(time, num_retained(queue))

            system.submit_jobs(queue)

        # Print every third hour
        if verbose and (time.hour != previous_hour and not time.hour % 3):
            previous_hour = time.hour
            print(
                "{} (step {}):\n".format(time, cnt) +
                "Idle Nodes = {} (highmem {})\tNodesReserved = {} " \
                "(Idle = {})\tNodesDown = {}\tPower = {:.4f} MW\n".format(
                    system.idle_history[-1], system.partitions["highmem"].available_nodes(),
                    # sum(
                    #     100 / 584 for job in system.running_jobs for node in job.assigned_nodes if (
                    #         "highmem" in [ partition.name for partition in node.partitions ]
                    #     )
                    # ),
                    sum(1 for node in system.nodes if node.reservation),
                    sum(1 for node in system.nodes if node.reservation and not node.running_job),
                    sum(1 for node in system.down_nodes if not node.running_job),
                    system.power_usage
                ) +
                "QueueSize = {} (held by priority {} (partition highmem {} qos lowpriority {}) " \
                "dependency {} qos holds {} (".format(
                    (
                        len(queue.queue) +
                        len(queue.waiting_dependency) +
                        sum(len(jobs) for jobs in queue.qos_held.values())
                    ),
                    len(queue.queue),
                    sum(1 for job in queue.queue if job.partition == "highmem"),
                    sum(1 for job in queue.queue if job.qos.name == "lowpriority"),
                    len(queue.waiting_dependency),
                    sum(len(jobs) for jobs in queue.qos_held.values())
                ) +
                " ".join(
                    "{}={}".format(
                        qos.name, len(jobs)
                    ) for qos, jobs in queue.qos_held.items() if len(jobs)
                ) +
                ") qos submit holds {} (".format(
                    sum(len(jobs) for jobs in queue.qos_submit_held.values())
                ) +
                " ".join(
                    "{}={}".format(
                        qos.name, len(jobs)
                    ) for qos, jobs in queue.qos_submit_held.items() if len(jobs)
                ) +
                "))\tRunningJobs = {}\n".format(len(system.running_jobs))
            )

        cnt += 1

    return system

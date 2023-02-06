import re
from enum import Enum
import datetime; from datetime import timedelta
from collections import defaultdict

import pandas as pd

from helpers import get_sbatch_cli_arg, timelimit_str_to_timedelta, convert_to_raw


class Queue:
    def __init__(self, job_data, partitions, priority_sorter=None):
        self.priority_sorter = priority_sorter

        # hardcoded for now
        # Don't have a proper way to simulate reservations as I can't see a history of deleted ones
        self.qoss = {
            "normal" : QOS("normal", 0, -1, -1, -1, -1, -1),
            "reservation" : QOS("reservation", 0.1, -1, -1, -1, -1, -1),
            "taskfarm" : QOS("taskfarm", 0.1, -1, 128, 512, 32, 128),
            "standard" : QOS("standard", 0.1, -1, -1, 1024, 16, 64),
            "short" : QOS("short", 0.1, -1, -1, 32, 4, 16),
            "long" : QOS("long", 0.1, 2048, -1, 512, 16, 16),
            "largescale" : QOS("largescale", 0.1, -1, -1, -1, 1, 8),
            "highmem" : QOS("highmem", 0.1, -1, -1, 256, 16, 16),
            "lowpriority" : QOS("lowpriority", 0.0002, -1, -1, 2048, 16, 16)
        }

        df_jobs = self._prep_job_data(job_data)
        self.time = df_jobs.Start.min()
        self.all_jobs = [
            Job(
                job_row.JobID, job_row.Submit, job_row.AllocNodes, job_row.Elapsed,
                job_row.Timelimit, job_row.TruePowerPerNode, job_row.TruePowerPerNode,
                job_row.Start, job_row.User, job_row.Account, self.qoss[job_row.QOS],
                partitions.get_partition_by_name(job_row.Partition), job_row.DependencyArg,
                job_row.JobName, job_row.Reason, job_row.ReservationArg, job_row.BeginArg
            ) for _, job_row in df_jobs.iterrows()
        ]
        self.all_jobs.sort(key=lambda job: (job.submit, job.id))
        # NOTE verify with jid first to ensure all jids have a Job in the data
        self._verify_dependencies()
        jid_to_job = { job.id : job for job in self.all_jobs }
        for job in self.all_jobs:
            job.init_dependency(jid_to_job)
        self.queue = []

        self.waiting_dependency = []

        self.qos_held = { qos : [] for qos in self.qoss.values() }
        self.qos_submit_held = { qos : [] for qos in self.qoss.values() }

        self._verify_reservations(partitions.valid_reservations)
        self.reservations = defaultdict(list)

    def set_priority_sorter(self, priority_sorter):
        self.priority_sorter = priority_sorter

    def next_newjob(self):
        try:
            return self.all_jobs[0].submit
        except IndexError:
            return datetime.datetime.max

    def step(self, time, running_jobs):
        self.time = time

        pre_step_res_priority_len = {
            res : len(res_queue) for res, res_queue in self.reservations.items()
        }
        pre_step_priority_len = len(self.queue)

        self._check_dependencies(running_jobs)
        self._check_qos_holds(running_jobs)

        if self.time < self.next_newjob():
            if len(self.queue) != pre_step_priority_len:
                self.queue = self.priority_sorter.sort(self.queue, self.time)
            return

        try:
            while self.all_jobs[0].submit <= self.time:
                new_job = self.all_jobs.pop(0)

                if new_job.qos.hold_job_submit_usr(new_job):
                    self.qos_submit_held[new_job.qos].append(new_job.qos_submit_hold())
                    continue

                new_job.submit_job()

                if new_job.dependency:
                    if not new_job.dependency.can_release(self.queue, running_jobs):
                        self.waiting_dependency.append(new_job.dependency_hold())
                        continue

                if new_job.qos.hold_job(new_job):
                    self.qos_held[new_job.qos].append(new_job.qos_resource_hold(self.time))
                    continue

                if new_job.reservation:
                    self.reservations[new_job.reservation].append(new_job.priority(self.time))
                    continue

                self.queue.append(new_job.priority(self.time))

        except IndexError: # No more new jobs
            pass

        if len(self.queue) != pre_step_priority_len:
            self.queue = self.priority_sorter.sort(self.queue, self.time)

        for res, res_queue in self.reservations.items():
            if len(res_queue) != pre_step_res_priority_len[res]:
                self.reservations[res] = self.priority_sorter.sort(res_queue, self.time)

    def _check_dependencies(self, running_jobs):
        released = []
        for i_job, job in enumerate(self.waiting_dependency):
            if not job.dependency.can_release(self.queue, running_jobs):
                continue

            released_job = self.waiting_dependency[i_job]
            if released_job.qos.hold_job(released_job):
                self.qos_held[released_job.qos].append(released_job.qos_resource_hold(self.time))
            elif released_job.reservation:
                self.reservations[released_job.reservation].append(
                    released_job.priority(self.time)
                )
            else:
                self.queue.append(released_job.priority(self.time))
            released.append(i_job)

        for i_job in sorted(released, reverse=True):
            self.waiting_dependency.pop(i_job)

    def _check_qos_holds(self, running_jobs):
        for qos in self.qos_submit_held.keys():
            if not self.qos_submit_held[qos]:
                continue

            released = []
            for i_job, job in enumerate(self.qos_submit_held[qos]):
                if job.qos.hold_job_submit_usr(job):
                    continue

                # Pretend the job is now resubmitted
                job.submit_job(self.time)
                released.append(i_job)

                if job.dependency:
                    if not job.dependency.can_release(self.queue, running_jobs):
                        self.waiting_dependency.append(job.dependency_hold())
                        continue

                if job.qos.hold_job(job):
                    self.qos_held[job.qos].append(job.qos_resource_hold(self.time))
                    continue

                if job.reservation:
                    self.reservations[job.reservation].append(job.priority(self.time))
                else:
                    self.queue.append(job.priority(self.time))

            for i_job in sorted(released, reverse=True):
                self.qos_submit_held[qos].pop(i_job)

        for qos in self.qos_held.keys():
            # Dont need to check if we know nothing will be released
            if (
                not self.qos_held[qos] or
                not qos.job_quota_remaining or
                not qos.node_quota_remaining
            ):
                continue
            self.qos_held[qos] = self.priority_sorter.sort(self.qos_held[qos], self.time)

            released = []
            for i_job, job in enumerate(self.qos_held[qos]):
                if job.qos.hold_job_grp(job):
                    break

                # Lower priority jobs can be released if the job infront is only held by a per user
                # limit
                if job.qos.hold_job_usr(job):
                    continue

                if job.reservation:
                    self.reservations[job.reservation].append(job.priority(self.time))
                else:
                    self.queue.append(job.priority(self.time))

                released.append(i_job)

            for i_job in sorted(released, reverse=True):
                self.qos_held[qos].pop(i_job)

    def _verify_reservations(self, valid_reservations):
        removed_res, removed_res_cnt = set(), 0
        for job in self.all_jobs:
            if not job.reservation:
                # short qos is reservationrequired
                if job.qos.name == "short":
                    job.reservation = "shortqos"
                continue

            if job.reservation not in valid_reservations:
                removed_res.add(job.reservation)
                removed_res_cnt += 1
                job.reservation = ""
                job.ignore_in_eval = True

        print(
            "Missing reservation records for {} resulting in ignoring reservations for" \
            "{} jobs".format(removed_res, removed_res_cnt)
        )

    def _verify_dependencies(self):
        removed_dep_cnt, ignored_dep_cnt = 0, 0
        all_ids = { job.id for job in self.all_jobs }
        for job in self.all_jobs:
            if not job.dependency:
                # Dependency hidden in batch file or just removed
                if job.reason == "Dependency":
                    job.ignore_in_eval = True
                    ignored_dep_cnt += 1
                continue

            for dep_type, ids in job.dependency.conditions.items():
                intersection = ids.intersection(all_ids)
                job.dependency.conditions[dep_type] = intersection

                # Job arrays need to be expanded into individual JobIDs
                for unmatched_id in ids.difference(intersection):
                    array_ids = {
                        id for id in all_ids if re.match("{}_[0-9]".format(unmatched_id), id)
                    }
                    if array_ids:
                        if job.dependency.delimiter == "?":
                            raise NotImplemetedError(
                                "Not implemented expanding job array ids for OR dependencies"
                            )
                        job.dependency.conditions[dep_type].update(array_ids)

            job.dependency.conditions_met = not any(job.dependency.conditions.values())
            if job.dependency.conditions_met and not job.dependency.singleton:
                removed_dep_cnt += 1
                job.dependency = None
                if job.reason == "Dependency":
                    job.ignore_in_eval = True
                    ignored_dep_cnt += 1

        print(
            (
                "Removed {} dependencies that cannot be satisfied from workload trace\n".format(
                    removed_dep_cnt
                )
            ) +
            (
                "Ignored {} in evaulation due to dependency not being in SubmitLine or missing " \
                "from workload trace".format(
                    ignored_dep_cnt
                )
            )
        )

    def _prep_job_data(self, data_path):
        df_jobs = pd.read_csv(
            data_path, delimiter='|', lineterminator='\n', header=0,
            usecols=[
                "JobID", "Start", "End", "Submit", "Elapsed", "ConsumedEnergyRaw", "AllocNodes",
                "Timelimit", "ReqCPUS", "ReqNodes", "Group", "QOS", "ReqMem", "User", "Account",
                "Partition", "SubmitLine", "JobName", "Reason", "State"
            ]
        )
        df_jobs = df_jobs.loc[
            (df_jobs.Start != "Unknown") & (df_jobs.Start.notna()) & (df_jobs.End != "Unknown") &
            (df_jobs.End.notna()) & (df_jobs.AllocNodes != "0") & (df_jobs.AllocNodes != 0) &
            (df_jobs.Partition != "serial")
        ]

        df_jobs.Submit = pd.to_datetime(df_jobs.Submit, format="%Y-%m-%dT%H:%M:%S")
        df_jobs.Start = pd.to_datetime(df_jobs.Start, format="%Y-%m-%dT%H:%M:%S")
        df_jobs.End = pd.to_datetime(df_jobs.End, format="%Y-%m-%dT%H:%M:%S")
        df_jobs.Elapsed = df_jobs.End - df_jobs.Start
        df_jobs.Timelimit = df_jobs.Timelimit.apply(lambda row: timelimit_str_to_timedelta(row))

        df_jobs = df_jobs[~df_jobs.duplicated(subset="JobID", keep="first")]

        convert_to_raw(df_jobs, "AllocNodes")

        num_bad = len(
            df_jobs.loc[
                (df_jobs.ConsumedEnergyRaw.isna()) | (df_jobs.ConsumedEnergyRaw == 0.0) |
                (df_jobs.ConsumedEnergyRaw == "")
            ]
        )
        df_jobs.ConsumedEnergyRaw = df_jobs.apply(
            lambda row: (
                float(row.ConsumedEnergyRaw) if (
                    row.ConsumedEnergyRaw == row.ConsumedEnergyRaw and # ie. not nan
                    row.ConsumedEnergyRaw != 0.0 and row.ConsumedEnergyRaw != ""
                ) else float(550 * row.AllocNodes * row.Elapsed.total_seconds())
            ),
            axis=1
        )
        df_jobs["Power"] = df_jobs.apply(
            lambda row: (
                (
                    float(row.ConsumedEnergyRaw) / row.Elapsed.total_seconds()
                ) if row.Elapsed.total_seconds() != 0 else 0.0
            ),
            axis=1
        )
        num_bad += len(df_jobs.loc[(df_jobs.Power >= 10000000)])
        for i, anomalous_row in df_jobs.loc[(df_jobs.Power >= 10000000)].iterrows():
            df_jobs.at[i, "Power"] = 550 * df_jobs.at[i, "AllocNodes"]
        print("Salvaged {} jobs with bad ConsumedEnergyRaw".format(num_bad))

        df_jobs["TruePowerPerNode"] = df_jobs.apply(
            lambda row: float(row.Power) / float(row.AllocNodes), axis=1
        )

        df_jobs["DependencyArg"] = df_jobs.SubmitLine.apply(
            lambda row: get_sbatch_cli_arg(row, long="--dependency", short="-d")
        )
        df_jobs["ReservationArg"] = df_jobs.SubmitLine.apply(
            lambda row: get_sbatch_cli_arg(row, long="--reservation")
        )
        df_jobs["BeginArg"] = df_jobs.SubmitLine.apply(
            lambda row: get_sbatch_cli_arg(row, long="--begin", short="-b")
        )

        print("{} heterogeneous JobIDs converted to regular JobIDs".format(
            len(df_jobs.loc[(df_jobs.JobID.str.contains("+", regex=False))])
        ))
        df_jobs.JobID = df_jobs.JobID.apply(
            lambda row: str(int(row.split("+")[0]) + int(row.split("+")[1])) if "+" in row else row
        )

        # Some error in slurm accounting, can correct for case of one other user in account
        num_fixed = 0
        for i, anomalous_row in df_jobs.loc[(df_jobs.User == "00:00:00")].iterrows():
            acc_users = df_jobs.loc[(df_jobs.Account == anomalous_row.Account)].User.unique()
            if len(acc_users) == 2:
                num_fixed += 1
                df_jobs.at[i, "User"] = (
                    acc_users[1] if acc_users[0] == "00:00:00" else acc_users[0]
                )
        print("Corrected {} of {} users with name 00:00:00".format(
            num_fixed, len(df_jobs.loc[(df_jobs.User == "00:00:00")])
        ))

        return df_jobs


class QOS:
    def __init__(self, name, priority, grp_nodes, grp_jobs, usr_nodes, usr_jobs, usr_submit):
        self.name = name
        self.priority = priority

        # TODO This is dumb, implement in a way that means in these cases the quota is just not tracked
        grp_jobs = 1000000000000 if grp_jobs == -1 else grp_jobs
        grp_nodes = 1000000000000 if grp_nodes == -1 else grp_nodes
        usr_jobs = 1000000000000 if usr_jobs == -1 else usr_jobs
        usr_nodes = 1000000000000 if usr_nodes == -1 else usr_nodes
        usr_submit = 1000000000000 if usr_submit == -1 else usr_submit

        self.grp_quotas = []
        self.job_quota_remaining = grp_jobs
        self.node_quota_remaining = grp_nodes

        self.usr_job_quota_remaining = defaultdict(lambda: usr_jobs)
        self.usr_node_quota_remaining = defaultdict(lambda: usr_nodes)

        self.usr_submit_quota_remaining = defaultdict(lambda: usr_submit)

    def job_submitted(self, user):
        self.usr_submit_quota_remaining[user] -= 1

    def job_launched(self, user, nodes):
        self.job_quota_remaining -= 1
        self.node_quota_remaining -= nodes
        self.usr_job_quota_remaining[user] -= 1
        self.usr_node_quota_remaining[user] -= nodes

    def job_ended(self, user, nodes):
        self.job_quota_remaining += 1
        self.node_quota_remaining += nodes
        self.usr_job_quota_remaining[user] += 1
        self.usr_node_quota_remaining[user] += nodes
        self.usr_submit_quota_remaining[user] += 1

    def hold_job(self, job):
        return (
            not self.job_quota_remaining or
            job.nodes > self.node_quota_remaining or
            not self.usr_job_quota_remaining[job.user] or
            job.nodes > self.usr_node_quota_remaining[job.user]
        )

    def hold_job_grp(self, job):
        return not self.job_quota_remaining or job.nodes > self.node_quota_remaining

    def hold_job_usr(self, job):
        return (
            not self.usr_job_quota_remaining[job.user] or
            job.nodes > self.usr_node_quota_remaining[job.user]
        )

    def hold_job_submit_usr(self, job):
        return not self.usr_submit_quota_remaining[job.user]


class Job:
    def __init__(
        self, id, submit : datetime, nodes, runtime : timedelta, reqtime: timedelta, node_power,
        true_node_power, true_job_start, user, account, qos, partition, dependency_arg, name,
        reason, reservation_arg, begin_arg
    ):
        self.id = id
        self.nodes = nodes
        self.runtime = runtime
        self.reqtime = reqtime
        self.node_power = node_power
        self.true_node_power = true_node_power
        self.true_submit = submit
        self.submit = submit
        self.true_job_start = true_job_start
        self.user = user
        self.account = account
        self.qos = qos
        self.partition = partition
        self.name = name
        self.dependency = Dependency(dependency_arg, user, name) if dependency_arg else None
        self.reservation = reservation_arg

        # Some features are not relevant for scheduluing (AssocMaxCpuMinutesPerJobLimit means for
        # archer that the user hasnt been allocated time yet, reservations, jobs held by user, ...)
        # and some I cant implemented with available data (JobArrayTaskLimit is usually specified
        # in batch script). Want to have these jobs in simulation but don't want to include them in
        # evaluation stage

        self.reason = reason
        self.ignore_in_eval = (
            reason in [
                "AssocMaxCpuMinutesPerJobLimit", "ReqNodeNotAvail", "BeginTime", "JobHeldUser",
                "DependencyNeverSatisfied", "JobArrayTaskLimit"
            ] or
            begin_arg
        )

        self.launch_time = None
        self.start = None
        self.end = None
        self.assigned_nodes = []

        self.state = JobState.FUTURE

    def init_dependency(self, jid_to_job):
        if self.dependency:
            self.dependency.convert_jids_to_jobs(jid_to_job)

    def submit_job(self, time=None):
        if time:
            self.submit = time
        self.qos.job_submitted(self.user)
        return self

    def start_job(self, time : datetime):
        self.start = time
        self.end = time + self.runtime
        self.endlimit = time + self.reqtime
        self.state = JobState.RUNNING
        return self

    def assign_node(self, node):
        node.set_busy()
        self.assigned_nodes.append(node)
        node.running_job = self
        if len(self.assigned_nodes) >= self.nodes:
            return True
        return False

    def end_job(self):
        for node in self.assigned_nodes:
            node.set_free()
            node.running_job = None
        self.qos.job_ended(self.user, self.nodes)
        self.state = JobState.COMPLETED
        return self

    def qos_submit_hold(self):
        self.state = JobState.QOS_SUBMIT
        return self

    def dependency_hold(self):
        self.state = JobState.DEPENDENCY
        return self

    # The earliest stage when job starts accruing age
    def qos_resource_hold(self, time):
        self.state = JobState.QOS_RESOURCES
        self.launch_time = time
        return self

    # When the QOS recognises the job as 'running' for resource limits accounting
    def priority(self, time):
        self.state = JobState.PRIORITY
        if not self.launch_time:
            self.launch_time = time
        self.qos.job_launched(self.user, self.nodes)
        return self


class Dependency:
    def __init__(self, dependency_args, user, name):
        self.job_user_name = (user, name)
        self.delimiter = "?" if "?" in dependency_args else ","

        self.conditions = {}
        self.singleton = False
        for condition in dependency_args.split(self.delimiter):
            if condition == "singleton":
                self.singleton = True
                continue

            dep_type = condition.split(":")[0]
            # NOTE after can can take a +time after job_id, just going to ignore these for now
            if "+" in condition:
                print(condition)
            jobs = { job_id.split("+")[0] for job_id in condition.split(":")[1:] }
            # Jobs in trace all ran so can assume these conditions are met and treat all the same
            if dep_type == "afterok" or dep_type == "afternotok" or dep_type == "afterany":
                self.conditions["afterany"] = jobs
                continue

            if dep_type == "after":
                self.conditions[dep_type] = jobs
                continue

            raise NotImplementedError("Unrecognised dep_type {}".format(dep_type))

        self.submitted_relevant = "after" in self.conditions.keys()
        self.finished_relevant = "afterany" in self.conditions.keys()

        self.conditions_met = not any(self.conditions.values())

    def convert_jids_to_jobs(self, jid_to_job):
        for key in self.conditions.keys():
            self.conditions[key] = { jid_to_job[jid] for jid in self.conditions[key] }

    def can_release(self, queued_jobs, running_jobs):
        if not self.conditions_met:
            if "afterany" in self.conditions.keys():
                for job in list(self.conditions["afterany"]):
                    if job.state != JobState.COMPLETED:
                        continue
                    self.conditions["afterany"].remove(job)
                    if self.delimiter == "?":
                        self.conditions_met = True
                        break

            if "after" in self.conditions.keys():
                for job in list(self.conditions["after"]):
                    if job.state != JobState.COMPLETED and job.state != RUNNING:
                        continue
                    self.conditions["after"].remove(job)
                    if self.delimiter == "?":
                        self.conditions_met = True
                        break

            self.conditions_met = not any(self.conditions.values())

        if self.conditions_met and not self.singleton:
            return True

        if self.conditions_met and self.singleton:
            launched_jobs = running_jobs + queued_jobs
            if self.job_user_name in { (job.user, job.name) for job in launched_jobs }:
                return False
            return True

        return False


class JobState(Enum):
    FUTURE = 1
    PRIORITY = 2
    RUNNING = 3
    COMPLETED = 4
    QOS_SUBMIT = 5
    DEPENDENCY = 6
    QOS_RESOURCES = 7


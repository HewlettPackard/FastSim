import re
from enum import Enum
import datetime; from datetime import timedelta
from collections import defaultdict

import pandas as pd

from helpers import get_sbatch_cli_arg, timelimit_str_to_timedelta, convert_to_raw


# TODO Rework the resource limits implementation, there is a lot of overlap that could be removed

# TODO Interaction between queue, priority sorter, and resource limits has become a bit of mess,
# clean


class Queue:
    def __init__(
        self, job_data, partitions, qos_dump, considered_partitions, priority_sorter=None
    ):
        self.priority_sorter = priority_sorter

        self.qoss = self._read_in_qos(qos_dump)
        self.assoc_limits = {}

        df_jobs = self._prep_job_data(job_data, considered_partitions)
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
        self.all_jobs.sort(key=lambda job: (job.submit, job.id), reverse=True)
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

        assoclimit_created = {}
        for assoc, node in priority_sorter.fairtree.assocs.items():
            if assoc[::2] in assoclimit_created:
                self.assoc_limits[assoc] = assoclimit_created[assoc[::2]]
            else:
                self.assoc_limits[assoc] = AssocLimit(node.max_jobs, node.max_submit)
                if node.partition is None:
                    assoclimit_created[assoc[::2]] = self.assoc_limits[assoc]

        for qos in self.qoss.values():
            qos.set_assoc_limits(self.assoc_limits)

    def next_newjob(self):
        try:
            return self.all_jobs[-1].submit
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
                self.priority_sorter.sort(self.queue, self.time)
            return

        try:
            # resubmit = defaultdict(list)
            while self.all_jobs[-1].submit <= self.time:
                new_job = self.all_jobs.pop()

                # NOTE This association is no longer allowed to run jobs. This was not true in the 
                # past since the job is in the workload trace. For now just skip these jobs and
                # keep this in mind. Could also give assoc a default allocation in this case.
                if (
                    (
                        ResourceLimit.ASSOC_JOBS in new_job.qos.controlled_by_assoc and
                        not self.assoc_limits[new_job.assoc].assoc_job_quota
                    ) or
                    (
                        ResourceLimit.ASSOC_SUBMIT in new_job.qos.controlled_by_assoc and
                        not self.assoc_limits[new_job.assoc].assoc_submit_quota
                    )
                ):
                    continue


                # NOTE commit 6b6482b has alternate implementation where submit jobs are held until
                # the user's next submission. This works slightly better as entire sustem
                # metrics but makes the wait times by qos an project worse
                if new_job.qos.hold_job_submit(new_job):
                    self.qos_submit_held[new_job.qos].append(new_job.qos_submit_hold())
                    continue

                # if new_job.qos.hold_job_submit(new_job):
                #     if new_job.is_dependency_target:
                #         self.qos_submit_held[new_job.qos].append(new_job.qos_submit_hold())
                #         continue

                #     resubmit[new_job.user].append(new_job)

                #     continue

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

        # earliest_resubmit = self.time + timedelta(hours=1)
        # for usr, resubmit_jobs in resubmit.items():
        #     for i_job_rev, job in enumerate(reversed(self.all_jobs)):
        #         if job.user != usr or job.submit < earliest_resubmit:
        #             continue
        #         for job_resubmit in resubmit_jobs:
        #             job_resubmit.submit = job.submit
        #         self.all_jobs[-i_job_rev:-i_job_rev] = reversed(resubmit_jobs)
        #         break
        #     if resubmit_jobs[0].submit == self.time:
        #         for job_resubmit in resubmit_jobs:
        #             job.ignore_in_eval = True
        #             self.qos_submit_held[job_resubmit.qos].append(job_resubmit.qos_submit_hold())

        if len(self.queue) != pre_step_priority_len:
            self.priority_sorter.sort(self.queue, self.time)

        for res, res_queue in self.reservations.items():
            if len(res_queue) != pre_step_res_priority_len.get(res, 0):
                self.priority_sorter.sort(res_queue, self.time)

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

        for i_job in reversed(released):
            self.waiting_dependency.pop(i_job)

    def _check_qos_holds(self, running_jobs):
        for qos in self.qos_submit_held.keys():
            if not self.qos_submit_held[qos]:
                continue

            # Pretending that the user resubmits in order of original submission as soon as allowed
            released, users_waiting = [], set()
            for i_job, job in enumerate(self.qos_submit_held[qos]):
                if job.user in users_waiting:
                    continue

                if job.qos.hold_job_submit(job):
                    users_waiting.add(job.user)
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

            for i_job in reversed(released):
                self.qos_submit_held[qos].pop(i_job)

        for qos in self.qos_held.keys():
            if not self.qos_held[qos]:
                continue

            self.priority_sorter.sort(self.qos_held[qos], self.time)

            released = []
            for i_job_rev, job in enumerate(reversed(self.qos_held[qos])):
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

                released.append(-(i_job_rev + 1))

            for i_job in reversed(released):
                self.qos_held[qos].pop(i_job)

    def _verify_reservations(self, valid_reservations):
        removed_res, removed_res_cnt = set(), 0
        for job in self.all_jobs:
            if not job.reservation:
                # short qos is reservation required
                if job.qos.name == "short":
                    job.reservation = "shortqos"
                else:
                    job.reservation = ""
                continue

            if job.reservation not in valid_reservations:
                removed_res.add(job.reservation)
                removed_res_cnt += 1
                job.reservation = ""
                job.ignore_in_eval = True

        print(
            "Missing reservation records for {} resulting in ignoring reservations for " \
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

    def _prep_job_data(self, data_path, considered_partitions):
        df_jobs = pd.read_csv(
            data_path, delimiter='|', lineterminator='\n', header=0, encoding="ISO-8859-1",
            usecols=[
                "JobID", "Start", "End", "Submit", "Elapsed", "ConsumedEnergyRaw", "AllocNodes",
                "Timelimit", "ReqCPUS", "ReqNodes", "Group", "QOS", "ReqMem", "User", "Account",
                "Partition", "SubmitLine", "JobName", "Reason", "State"
            ],
        )
        df_jobs = df_jobs.loc[
            (df_jobs.Start != "Unknown") & (df_jobs.Start.notna()) & (df_jobs.End != "Unknown") &
            (df_jobs.End.notna()) & (df_jobs.AllocNodes != "0") & (df_jobs.AllocNodes != 0) &
            (df_jobs.Partition.isin(considered_partitions)) & (df_jobs.Timelimit.notna())
        ]

        df_jobs.Submit = pd.to_datetime(df_jobs.Submit, format="%Y-%m-%dT%H:%M:%S")
        df_jobs.Start = pd.to_datetime(df_jobs.Start, format="%Y-%m-%dT%H:%M:%S")
        df_jobs.End = pd.to_datetime(df_jobs.End, format="%Y-%m-%dT%H:%M:%S")
        df_jobs.Elapsed = df_jobs.End - df_jobs.Start
        df_jobs.Timelimit = df_jobs.Timelimit.apply(lambda row: timelimit_str_to_timedelta(row))

        with_dupes = len(df_jobs)
        df_jobs = df_jobs[~df_jobs.duplicated(subset="JobID", keep="first")]
        print("{} duplicate ids removed".format(len(df_jobs) - with_dupes))

        convert_to_raw(df_jobs, "AllocNodes")

        num_bad = len(
            df_jobs.loc[
                (df_jobs.ConsumedEnergyRaw.isna()) | (df_jobs.ConsumedEnergyRaw == 0.0) |
                (df_jobs.ConsumedEnergyRaw == "")
            ]
        )
        if num_bad / len(df_jobs) < 0.25:
            mean_power_per_node = df_jobs.loc[
                (df_jobs.ConsumedEnergyRaw.notna()) & (df_jobs.ConsumedEnergyRaw != 0.0) &
                (df_jobs.ConsumedEnergyRaw != "")
            ].apply(
                lambda row: (
                    float(row.ConsumedEnergyRaw) / row.Elapsed.total_seconds() / row.AllocNodes
                    if row.Elapsed.total_seconds() != 0
                    else 0.0
                ),
                axis=1
            ).mean
            df_jobs.ConsumedEnergyRaw = df_jobs.apply(
                lambda row: (
                    float(row.ConsumedEnergyRaw)
                    if (
                        row.ConsumedEnergyRaw == row.ConsumedEnergyRaw and
                        row.ConsumedEnergyRaw != 0.0 and
                        row.ConsumedEnergyRaw != ""
                    )
                    else float(mean_power_per_node * row.AllocNodes * row.Elapsed.total_seconds())
                ),
                axis=1
            )
        else:
            print(
                "!!!More than 25% of jobs do not have a valid ConsumedEnergy,"
                "setting all ConsumedEnergies to zero!!!"
            )
            df_jobs = df_jobs.assign(ConsumedEnergyRaw=0.0)
            mean_power_per_node = 0.0

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
            df_jobs.at[i, "Power"] = mean_power_per_node * df_jobs.at[i, "AllocNodes"]
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
    
        df_jobs.JobID = df_jobs.JobID.apply(lambda row: str(row))    
        print("{} heterogeneous JobIDs converted to regular JobIDs".format(
            len(df_jobs.loc[(df_jobs.JobID.str.contains("+", regex=False))])
        ))
        df_jobs.JobID = df_jobs.JobID.apply(
            lambda row: str(int(row.split("+")[0]) + int(row.split("+")[1])) if "+" in row else row
        )

        # Some error in slurm accounting, can correct for case of one other user in account
        num_broken, num_fixed = len(df_jobs.loc[(df_jobs.User == "00:00:00")]), 0
        for i, anomalous_row in df_jobs.loc[(df_jobs.User == "00:00:00")].iterrows():
            acc_users = df_jobs.loc[(df_jobs.Account == anomalous_row.Account)].User.unique()
            if len(acc_users) == 2:
                num_fixed += 1
                df_jobs.at[i, "User"] = (
                    acc_users[1] if acc_users[0] == "00:00:00" else acc_users[0]
                )
        print("Corrected {} of {} users with name 00:00:00".format(num_fixed, num_broken))

        print("{} Jobs in workload trace".format(len(df_jobs)))

        return df_jobs

    def _read_in_qos(self, qos_dump):
        df_qos = pd.read_csv(
            qos_dump,  delimiter='|', lineterminator='\n', header=0, encoding="ISO-8859-1"
        )

        qoss = {}

        for _, row in df_qos.iterrows():
            qoss[row.Name] = QOS(
                row.Name,
                int(row.Priority),
                (
                    None
                    if pd.isna(row.GrpTRES) or "node=" not in row.GrpTRES
                    else int(row.GrpTRES.split("node=")[1].split(",")[0])
                ),
                None if pd.isna(row.GrpJobs) else int(row.GrpJobs),
                None if pd.isna(row.GrpSubmit) else int(row.GrpSubmit),
                (
                    None
                    if pd.isna(row.MaxTRESPU) or "node=" not in row.MaxTRESPU
                    else int(row.MaxTRESPU.split("node=")[1].split(",")[0])
                ),
                None if pd.isna(row.MaxJobsPU) else int(row.MaxJobsPU),
                None if pd.isna(row.MaxJobs) else int(row.MaxJobs),
                None if pd.isna(row.MaxSubmitPU) else int(row.MaxSubmitPU),
                None if pd.isna(row.MaxSubmit) else int(row.MaxSubmit)
            )

        # print(
        #     "Name: Priority GrpTRES GrpJobs GrpSubmit MaxTRESPU MaxJobsPU MaxJobs MaxSubmitPU"
        #     "MaxSubmit"
        # )
        # for name, qos in qoss.items():
        #     print(name, end=": ")
        #     print(
        #         qos.priority, qos.node_quota_remaining, qos.job_quota_remaining,
        #         qos.submit_quota_remaining, qos.usr_node_quota_remaining["dummy"],
        #         qos.usr_job_quota_remaining["dummy"], qos.assoc_job_quota_remaining["dummy"],
        #         qos.usr_submit_quota_remaining["dummy"], qos.assoc_submit_quota_remaining["dummy"],
        #         sep=" "
        #     )

        return qoss


# NOTE Ignoring any QOS linked to the partition that could override this (not applicable for LUMI
# or ARCHER). Also ignoring per account resource limits.
class QOS:
    def __init__(
        self, name, priority, grp_nodes, grp_submit, grp_jobs, usr_nodes, usr_jobs, assoc_jobs,
        usr_submit, assoc_submit
    ):
        self.name = name
        self.priority = priority

        self.assoc_limits = {}

        self.tracked_limits = set()
        if grp_jobs is not None:
            self.tracked_limits.add(ResourceLimit.GRP_JOBS)
        if grp_nodes is not None:
            self.tracked_limits.add(ResourceLimit.GRP_NODES)
        if grp_submit is not None:
            self.tracked_limits.add(ResourceLimit.GRP_SUBMIT)
        if usr_jobs is not None:
            self.tracked_limits.add(ResourceLimit.USR_JOBS)
        if usr_nodes is not None:
            self.tracked_limits.add(ResourceLimit.USR_NODES)
        if usr_submit is not None:
            self.tracked_limits.add(ResourceLimit.USR_SUBMIT)
        if assoc_jobs is not None:
            self.tracked_limits.add(ResourceLimit.ASSOC_JOBS)
        if assoc_submit is not None:
            self.tracked_limits.add(ResourceLimit.ASSOC_SUBMIT)

        self.controlled_by_assoc = {
            limit
            for limit in [ResourceLimit.ASSOC_JOBS, ResourceLimit.ASSOC_SUBMIT]
                if limit not in self.tracked_limits
                }

        self.job_quota_remaining = grp_jobs
        self.node_quota_remaining = grp_nodes
        self.submit_quota_remaining = grp_submit

        self.usr_job_quota_remaining = defaultdict(lambda: usr_jobs)
        self.usr_node_quota_remaining = defaultdict(lambda: usr_nodes)
        self.usr_submit_quota_remaining = defaultdict(lambda: usr_submit)

        self.assoc_job_quota_remaining = defaultdict(lambda: assoc_jobs)
        self.assoc_submit_quota_remaining = defaultdict(lambda: assoc_submit)

    def set_assoc_limits(self, assoc_limits):
        self.assoc_limits = assoc_limits

    def job_submitted(self, job):
        if ResourceLimit.GRP_SUBMIT in self.tracked_limits:
            self.submit_quota_remaining -= 1
        if ResourceLimit.USR_SUBMIT in self.tracked_limits:
            self.usr_submit_quota_remaining[job.user] -= 1
        if ResourceLimit.ASSOC_SUBMIT in self.tracked_limits:
            self.assoc_submit_quota_remaining[job.assoc] -= 1

        self.assoc_limits[job.assoc].job_submitted()

    def job_launched(self, job):
        if ResourceLimit.GRP_JOBS in self.tracked_limits:
            self.job_quota_remaining -= 1
        if ResourceLimit.GRP_NODES in self.tracked_limits:
            self.node_quota_remaining -= job.nodes
        if ResourceLimit.USR_JOBS in self.tracked_limits:
            self.usr_job_quota_remaining[job.user] -= 1
        if ResourceLimit.USR_NODES in self.tracked_limits:
            self.usr_node_quota_remaining[job.user] -= job.nodes
        if ResourceLimit.ASSOC_JOBS in self.tracked_limits:
            self.assoc_job_quota_remaining[job.assoc] -= 1

        self.assoc_limits[job.assoc].job_launched()

    def job_ended(self, job):
        if ResourceLimit.GRP_JOBS in self.tracked_limits:
            self.job_quota_remaining += 1
        if ResourceLimit.GRP_NODES in self.tracked_limits:
            self.node_quota_remaining += job.nodes
        if ResourceLimit.GRP_SUBMIT in self.tracked_limits:
            self.submit_quota_remaining += 1
        if ResourceLimit.USR_JOBS in self.tracked_limits:
            self.usr_job_quota_remaining[job.user] += 1
        if ResourceLimit.USR_NODES in self.tracked_limits:
            self.usr_node_quota_remaining[job.user] += job.nodes
        if ResourceLimit.USR_SUBMIT in self.tracked_limits:
            self.usr_submit_quota_remaining[job.user] += job.nodes
        if ResourceLimit.ASSOC_JOBS in self.tracked_limits:
            self.assoc_job_quota_remaining[job.assoc] += 1
        if ResourceLimit.ASSOC_SUBMIT in self.tracked_limits:
            self.assoc_submit_quota_remaining[job.assoc] += 1

        self.assoc_limits[job.assoc].job_ended()

    def hold_job(self, job):
        return self.hold_job_grp(job) or self.hold_job_usr(job)

    def hold_job_grp(self, job):
        if (
            ResourceLimit.GRP_JOBS in self.tracked_limits and
            not self.job_quota_remaining
        ):
            return True
        if (
            ResourceLimit.GRP_NODES in self.tracked_limits and
            job.nodes > self.node_quota_remaining
        ):
            return True

        return False

    def hold_job_usr(self, job):
        if (
            ResourceLimit.USR_JOBS in self.tracked_limits and
            not self.usr_job_quota_remaining[job.user]
        ):
            return True
        if (
            ResourceLimit.USR_NODES in self.tracked_limits and
            job.nodes > self.usr_node_quota_remaining[job.user]
        ):
            return True
        if (
            ResourceLimit.ASSOC_JOBS in self.tracked_limits and
            not self.assoc_job_quota_remaining[job.assoc]
        ):
            return True

        return self.assoc_limits[job.assoc].hold_job(self.controlled_by_assoc)

    def hold_job_submit(self, job):
        return self.hold_job_submit_grp(job) or self.hold_job_submit_usr(job)

    def hold_job_submit_grp(self, job):
        if (
            ResourceLimit.GRP_SUBMIT in self.tracked_limits and
            not self.submit_quota_remaining
        ):
            return True

        return False

    def hold_job_submit_usr(self, job):
        if (
            ResourceLimit.USR_SUBMIT in self.tracked_limits and
            not self.usr_submit_quota_remaining[job.user]
        ):
            return True
        if (
            ResourceLimit.ASSOC_SUBMIT in self.tracked_limits and
            not self.assoc_submit_quota_remaining[job.assoc]
        ):
            return True

        return self.assoc_limits[job.assoc].hold_job_submit(self.controlled_by_assoc)


# NOTE This is only setup for user association limits. Slurm can have limits set on account
# associations and cluster association which can override the associtions below them. Implementing
# this would required extending this class for accounts also and having it interact with the
# assoctree so it knows what checks to pass on to its children. Each user assoc would then have
# a single assoc limit class with some of the qoutas refering to the parent account AssocLimit
# class, like the QOS does to this class currently. Would not need to check which resource limits
# overidde what for each job since the association relationship is constant
class AssocLimit:
    def __init__(self, assoc_jobs, assoc_submit):
        self.tracked_limits = set()
        if assoc_jobs is not None:
            self.tracked_limits.add(ResourceLimit.ASSOC_JOBS)
        if assoc_submit is not None:
            self.tracked_limits.add(ResourceLimit.ASSOC_SUBMIT)

        self.assoc_job_quota_remaining = assoc_jobs
        self.assoc_submit_quota_remaining = assoc_submit

        self.assoc_job_quota = assoc_jobs
        self.assoc_submit_quota = assoc_submit

    def job_submitted(self):
        if ResourceLimit.ASSOC_SUBMIT in self.tracked_limits:
            self.assoc_submit_quota_remaining -= 1

    def job_launched(self):
        if ResourceLimit.ASSOC_JOBS in self.tracked_limits:
            self.assoc_job_quota_remaining -= 1

    def job_ended(self):
        if ResourceLimit.ASSOC_SUBMIT in self.tracked_limits:
            self.assoc_submit_quota_remaining += 1
        if ResourceLimit.ASSOC_JOBS in self.tracked_limits:
            self.assoc_job_quota_remaining += 1

    def hold_job(self, limits):
        if (
            ResourceLimit.ASSOC_JOBS in self.tracked_limits and
            ResourceLimit.ASSOC_JOBS in limits
            and not self.assoc_job_quota_remaining
        ):
            return True

        return False

    def hold_job_submit(self, limits):
        if (
            ResourceLimit.ASSOC_SUBMIT in self.tracked_limits and
            ResourceLimit.ASSOC_SUBMIT in limits
            and not self.assoc_submit_quota_remaining
        ):
            return True

        return False

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
        # Dependency may be submitted incorrectly (typo or wrong format)
        if (
            dependency_arg is None or
            all(
                dep_type not in dependency_arg
                    for dep_type in [
                        "after:", "afterany:", "afterburstbuffer", "aftercorr", "afternotok",
                        "afterok", "singleton"
                    ]
            )
        ):
            self.dependency = None
        else:
            self.dependency =  Dependency(dependency_arg, user, name)
        self.reservation = reservation_arg

        self.assoc = (self.user, self.partition, self.account)

        self.is_dependency_target = False

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
        self.assigned_nodes = set()

        self.state = JobState.FUTURE

        self.planned_block = None

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Job):
            return self.id == other.id
        return False

    def init_dependency(self, jid_to_job):
        if self.dependency:
            self.dependency.convert_jids_to_jobs(jid_to_job)

            for jobs in self.dependency.conditions.values():
                for job in jobs:
                    job.is_dependency_target = True

    def submit_job(self, time=None):
        if time:
            self.submit = time
        self.qos.job_submitted(self)
        return self

    def start_job(self, time : datetime):
        self.start = time
        self.end = time + self.runtime
        self.endlimit = time + self.reqtime
        self.state = JobState.RUNNING
        return self

    def assign_node(self, node):
        node.set_busy()
        self.assigned_nodes.add(node)
        node.running_job = self
        if len(self.assigned_nodes) >= self.nodes:
            return True
        return False

    def end_job(self):
        for node in self.assigned_nodes:
            node.set_free()
            node.running_job = None
        self.qos.job_ended(self)
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
        self.qos.job_launched(self)
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
            # if "+" in condition:
            #     print("!!!Some jobs have dependencies with +time offsets!!!")
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
                    if job.state != JobState.COMPLETED and job.state != JobState.RUNNING:
                        continue
                    self.conditions["after"].remove(job)
                    if self.delimiter == "?":
                        self.conditions_met = True
                        break

            self.conditions_met = not any(self.conditions.values())

        if self.conditions_met and not self.singleton:
            return True

        # NOTE Might be inefficient to keep makeing lauched_jobs in the case of multiple singletons
        # Singletons are rare in ARCHER2 data so not worrying for now
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


class ResourceLimit(Enum):
    GRP_NODES = 1
    GRP_JOBS = 2
    GRP_SUBMIT = 3
    USR_JOBS = 4
    USR_NODES = 5
    USR_SUBMIT = 6
    ASSOC_SUBMIT = 7
    ASSOC_JOBS = 8


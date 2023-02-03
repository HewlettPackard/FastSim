import re
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

from globals import *


class Job():
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

        self.launch_time = submit
        self.start = None
        self.end = None
        self.assigned_nodes = []

    def submit_job(self, time=None):
        if time:
            self.submit = time
        self.qos.job_submitted(self.user)
        return self

    def start_job(self, time : datetime):
        self.start = time
        self.end = time + self.runtime
        self.endlimit = time + self.reqtime
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
        return self

    # When job starts accruing age
    def launch(self, time):
        self.launch_time = time
        return self

    # When the QOS recognises the job as 'running' for resource limits accounting
    def launch_into_priority(self):
        self.qos.job_launched(self.user, self.nodes)
        return self


class Queue():
    def __init__(self, df_jobs, init_time, priority_sorter, valid_reservations=[]):
        self.priority_sorter = priority_sorter
        self.time = init_time

        self.system = None

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

        self.all_jobs = [
            Job(
                job_row.JobID, job_row.Submit, job_row.AllocNodes, job_row.Elapsed,
                job_row.Timelimit, job_row.PowerPerNode, job_row.TruePowerPerNode, job_row.Start,
                job_row.User, job_row.Account, self.qoss[job_row.QOS], job_row.Partition,
                job_row.DependencyArg, job_row.JobName, job_row.Reason, job_row.ReservationArg,
                job_row.BeginArg
            ) for _, job_row in df_jobs.iterrows()
        ]
        self.all_jobs.sort(key=lambda job: (job.submit, job.id))
        self._verify_dependencies()
        self.queue = []

        self.waiting_dependency = []

        self.qos_held = { qos : [] for qos in self.qoss.values() }
        self.qos_submit_held = { qos : [] for qos in self.qoss.values() }

        self._verify_reservations(valid_reservations)
        self.reservations = defaultdict(list)

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

    def set_system(self, system):
        self.system = system

    def _check_dependencies(self):
        released = []
        for i_job, job in enumerate(self.waiting_dependency):
            job.dependency.update_finished(self.system.finished_jobs_step)
            job.dependency.update_submitted(self.system.submitted_jobs_step)

            if not job.dependency.can_release(self.queue, self.system.running_jobs):
                continue

            released_job = self.waiting_dependency[i_job]
            if released_job.qos.hold_job(released_job):
                self.qos_held[released_job.qos].append(released_job.launch(self.time))
            elif released_job.reservation:
                self.reservations[released_job.reservation].append(
                    released_job.launch(self.time).launch_into_priority()
                )
            else:
                self.queue.append(released_job.launch(self.time).launch_into_priority())
            released.append(i_job)

        for i_job in sorted(released, reverse=True):
            self.waiting_dependency.pop(i_job)

        self.system.finished_jobs_step = []
        self.system.submitted_jobs_step = []

    def _check_qos_holds(self):
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
                    job.dependency.update_finished(self.system.job_history)
                    job.dependency.update_submitted(
                        self.system.running_jobs + self.system.job_history
                    )
                    if not job.dependency.can_release(self.queue, self.system.running_jobs):
                        self.waiting_dependency.append(job)
                        continue

                if job.qos.hold_job(job):
                    self.qos_held[job.qos].append(job.launch(self.time))
                    continue

                if job.reservation:
                    self.reservations[job.reservation].append(job.launch_into_priority())
                else:
                    self.queue.append(job.launch_into_priority())

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
                    self.reservations[job.reservation].append(job.launch_into_priority())
                else:
                    self.queue.append(job.launch_into_priority())

                released.append(i_job)

            for i_job in sorted(released, reverse=True):
                self.qos_held[qos].pop(i_job)

    def step(self, time, retained):
        self.time = time

        self._check_dependencies()
        self._check_qos_holds()

        if self.time < self.next_newjob():
            # Still need to sort as there may be new jobs from dependency and qos releases
            self.queue[retained:] = self.priority_sorter.sort(self.queue[retained:], self.time)
            return

        try:
            while self.all_jobs[0].submit <= self.time:
                new_job = self.all_jobs.pop(0)

                if new_job.qos.hold_job_submit_usr(new_job):
                    self.qos_submit_held[new_job.qos].append(new_job)
                    continue

                new_job.submit_job()

                if new_job.dependency:
                    new_job.dependency.update_finished(self.system.job_history)
                    new_job.dependency.update_submitted(
                        self.system.running_jobs + self.system.job_history
                    )
                    if not new_job.dependency.can_release(self.queue, self.system.running_jobs):
                        self.waiting_dependency.append(new_job)
                        continue

                if new_job.qos.hold_job(new_job):
                    self.qos_held[new_job.qos].append(new_job.launch(self.time))
                    continue

                if new_job.reservation:
                    self.reservations[new_job.reservation].append(new_job.launch(self.time))
                    continue

                self.queue.append(new_job.launch(self.time).launch_into_priority())

        except IndexError: # No more new jobs
            pass

        self.queue[retained:] = self.priority_sorter.sort(self.queue[retained:], self.time)
        for reservation in list(self.reservations.keys()):
            self.reservations[reservation] = (
                self.priority_sorter.sort(self.reservations[reservation], self.time)
            )

    def next_newjob(self):
        try:
            return self.all_jobs[0].submit
        except IndexError:
            return datetime.max


class Dependency():
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
            job_ids = { job_id.split("+")[0] for job_id in condition.split(":")[1:] }
            # Jobs in trace all ran so can assume these conditions are met and treat all the same
            if dep_type == "afterok" or dep_type == "afternotok" or dep_type == "afterany":
                self.conditions["afterany"] = job_ids
                continue

            if dep_type == "after":
                self.conditions[dep_type] = job_ids
                continue

            raise NotImplementedError("Unrecognised dep_type {}".format(dep_type))

        self.submitted_relevant = "after" in self.conditions.keys()
        self.finished_relevant = "afterany" in self.conditions.keys()

        self.conditions_met = not any(self.conditions.values())

    def update_finished(self, jobs):
        # return
        if not self.finished_relevant or self.conditions_met:
            return

        for job in jobs:
            if job.id in self.conditions["afterany"]:
                self.conditions["afterany"].remove(job.id)
                if self.delimiter == "?":
                    self.conditions_met = True
                    return

    def update_submitted(self, jobs):
        # return
        if not self.submitted_relevant or self.conditions_met:
            return

        for job in jobs:
            if job.id in self.conditions["after"]:
                self.conditions["after"].remove(job.id)
                if self.delimiter == "?":
                    self.conditions_met = True
                    return

    def can_release(self, queued_jobs, running_jobs):
        if not self.conditions_met:
            self.conditions_met = not any(self.conditions.values())
            if not self.conditions_met:
                return False

        if self.conditions_met and not self.singleton:
            return True

        if self.conditions_met and self.singleton:
            launched_jobs = running_jobs + queued_jobs
            if self.job_user_name in { (job.user, job.name) for job in launched_jobs }:
                return False
            return True

        return False


class Partition():
    def __init__(self, name, priority_tier, priority_weight):
        self.name = name
        self.priority_tier = priority_tier
        self.priority_weight = priority_weight # Normalised s.t. partition with greatest has 1

        self.nodes = []
        self.planned_nodes = []
        self.always_available_nodes = []
        self.backfill_only_nodes = []

        self.num_available = 0
        self.num_available_only_backfill = 0

    def add_node(self, node):
        node.partitions.append(self)
        self.nodes.append(node)
        self.nodes.sort(key=lambda node: (node.weight, node.id)) # Small weights get priority
        if not node.job_end_restriction:
            self.always_available_nodes.append(node)
            self.num_available += node.free
        else:
            self.backfill_only_nodes.append(node)
            self.num_available_only_backfill += node.free

    def available_nodes(self, backfill=False):
        if backfill:
            return self.num_available + self.num_available_only_backfill

        return self.num_available

    def set_planned(self):
        for node in self.nodes:
            if node.free:
                node.set_planned()
                self.planned_nodes.append(node)

    def set_unplanned(self):
        while self.planned_nodes:
            self.planned_nodes.pop(0).set_unplanned()


class QOS():
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
        self.possible_to_release = True

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


class Node():
    def __init__(
        self, num, weight=0, down_schedule=[], reservation_schedule=[], job_end_restriction=None
    ):
        self.id = num
        self.weight = weight

        self.free = True
        self.running_job = None

        self.down_schedule = down_schedule
        self.down = False
        self.up_time = None

        self.reservation_schedule = reservation_schedule
        self.reservation = ""
        self.unreserved_time = None
        self.job_end_restriction = job_end_restriction

        self.planned = False

        self.partitions = []

    def set_planned(self):
        self.planned = True
        self.free = False
        for partition in self.partitions:
            if self.job_end_restriction: # NOTE This is so janky but Im desparate, fix in refactor
                partition.num_available_only_backfill -= 1
                continue
            partition.num_available -= 1

    def set_unplanned(self):
        self.planned = False
        self.free = True
        for partition in self.partitions:
            if self.job_end_restriction:
                partition.num_available_only_backfill += 1
                continue
            partition.num_available += 1

    def set_reserved(self, reservation_name, end_time):
        self.reservation = reservation_name
        self.unreserved_time = end_time
        if self.down or not self.free:
            return
        self.free = False
        for partition in self.partitions:
            if self.job_end_restriction:
                partition.num_available_only_backfill -= 1
                continue
            partition.num_available -= 1

    def set_unreserved(self):
        self.reservation = ""
        self.unreserved_time = None
        if self.down or self.running_job:
            return
        self.free = True
        for partition in self.partitions:
            if self.job_end_restriction:
                partition.num_available_only_backfill += 1
                continue
            partition.num_available += 1

    def set_down(self, up_time):
        self.down = True
        self.up_time = up_time
        # If job is already running it is allowed to finish
        if not self.free:
            return
        self.free = False
        for partition in self.partitions:
            if self.job_end_restriction:
                partition.num_available_only_backfill -= 1
                continue
            partition.num_available -= 1

    def set_up(self):
        self.down = False
        self.up_time = None
        if self.reservation:
            return
        self.free = True
        for partition in self.partitions:
            if self.job_end_restriction:
                partition.num_available_only_backfill += 1
                continue
            partition.num_available += 1

    def set_free(self):
        if self.down or self.reservation:
            return
        self.free = True
        for partition in self.partitions:
            if self.job_end_restriction:
                partition.num_available_only_backfill += 1
                continue
            partition.num_available += 1

    def set_busy(self):
        self.free = False
        for partition in self.partitions:
            if self.job_end_restriction:
                partition.num_available_only_backfill -= 1
                continue
            partition.num_available -= 1


class Archer2():
    def __init__(
        self, init_time : datetime, nodes=None, partitions=None, backfill_opts={},
        low_freq_condition=lambda queue: False, low_freq_calc=None, low_freq_reqtime_factor=1.0,
    ):
        self.power_usage = 0 # MW
        self.init_time = init_time
        self.time = init_time
        self.backfill_opts = backfill_opts
        if "resolution" not in self.backfill_opts:
            self.backfill_opts["resolution"] = timedelta(minutes=1) # 1min for Archer2
        if "max_job_test" not in self.backfill_opts:
            self.backfill_opts["max_job_test"] = 1000 # 1000 for Archer2
        self.low_freq_condition = low_freq_condition
        self.low_freq_calc = low_freq_calc
        self.low_freq_reqtime_factor = low_freq_reqtime_factor

        self.running_jobs = []

        self.power_history = [self.power_usage] # MW
        self.idle_history = [5860] # %
        self.queue_size = 0
        self.queue_size_history = [0]
        self.times = [self.time]
        self.bd_slowdowns = []
        self.job_history = []
        self.finished_jobs_step = []
        self.submitted_jobs_step = []
        self.total_energy = 0.0 # GJ

        self.partitions = partitions
        self.nodes = nodes
        # self.node_down_order = [
        #     node for node in nodes if node.down_schedule
        # ].sort(key=lambda node: node.down_schedule[0])
        self.node_down_order = []
        self.down_nodes = []
        self.node_reservation_order = []
        self.reserved_nodes = []

        self.nodes_free = 5860

    def set_nodes_partitions(self, nodes, partitions):
        self.nodes = nodes
        self.partitions = partitions
        self.node_down_order = sorted(
            [ node for node in nodes if node.down_schedule ],
            key=lambda node: node.down_schedule[0][0]
        )
        self.node_reservation_order = sorted(
            [ node for node in nodes if node.reservation_schedule ],
            key=lambda node: node.reservation_schedule[0][0]
        )

    def has_space(self, job : Job, backfill=False):
        return (
            True if self.available_nodes(job.partition, backfill=backfill) >= job.nodes else False
        )

    def available_nodes(self, partition="", backfill=False):
        if partition:
            return self.partitions[partition].available_nodes(backfill=backfill)

        return self.nodes_free

    def next_event(self):
        if not self.running_jobs:
            return datetime.max

        self.running_jobs.sort(key=lambda job: job.end)

        return self.running_jobs[0].end

    # NOTE: make low_freq_reqtime_factor and general reqtime scaling factor as it may be
    # interesting to know if it can be tuned to a systems workload eg. users tend to overestimate
    # time so might be beneficial to reduce the reqtimes
    def get_backfill_jobs(self, queue : Queue):
        backfill_now = []

        # if self.backfill_opts["EASY"]:
        if False:
            free_nodes = self.available_nodes()
            for job in sorted(self.running_jobs, key=lambda job: job.endlimit):
                free_nodes += job.nodes
                if free_nodes >= queue.queue[0].nodes:
                    shadow_time = job.endlimit
                    extra_nodes = free_nodes - queue.queue[0].nodes
                    break

            for i_job, job in enumerate(
                list(queue.queue)[:max(len(queue.queue), self.backfill_opts["max_job_test"])]
            ):
                # Shadow time too short and no extra nodes -> not possible to backfill anymore
                if shadow_time <= self.time + self.backfill_opts["resolution"] and not extra_nodes:
                    return backfill_now

                pass

            return backfill_now

        free_blocks = defaultdict(set)
        free_blocks_ready_intervals = set()
        for node in self.nodes:
            if not node.free:
                continue
            if not node.job_end_restriction:
                free_blocks[(self.time, datetime.max)].add(node)
                free_blocks_ready_intervals.add((self.time, datetime.max))
                continue
            free_blocks[(self.time, node.job_end_restriction(self.time))].add(node)
            free_blocks_ready_intervals.add((self.time, node.job_end_restriction(self.time)))

        if not free_blocks_ready_intervals:
            return backfill_now
        # free_blocks[(self.time, datetime.max)] = {
        #     node for node in self.nodes if node.free and not node.job_end_restriction
        # }
        for job in self.running_jobs:
            # Some jobs exceeding timelimit and some with zero reqtime
            if job.endlimit <= self.time:
                job.endlimit = self.time + self.backfill_opts["resolution"]
            for node in job.assigned_nodes:
                if not node.job_end_restriction:
                    free_blocks[(job.endlimit, datetime.max)].add(node)
                    continue
                end_restriction = node.job_end_restriction(self.time)
                if job.endlimit >= end_restriction:
                    continue
                free_blocks[(job.endlimit, end_restriction)].add(node)
            # free_blocks[(job.endlimit, datetime.max)].update(job.assigned_nodes)

        min_required_block_time = (
            self.time +
            (
                min(queue.queue, key=lambda job: job.reqtime).reqtime *
                self.low_freq_reqtime_factor + self.backfill_opts["resolution"]
            )
        )

        # free_blocks_ready_intervals = (
        #     { (self.time, datetime.max) } if self.available_nodes() else set()
        # )
        # max_block_time = datetime.max
        max_block_time = max(free_blocks_ready_intervals, key=lambda interval: interval[1])[1]
        partition_num_tested = defaultdict(int)
        # Dont consider jobs that require partitions with no available nodes
        partition_maxes = {
            partition : (
                self.backfill_opts["max_job_test"] if (
                    self.available_nodes(partition, backfill=True)
                ) else 0
            ) for partition in self.partitions.keys()
        }
        for i_job, job in enumerate(queue.queue):
            # Mimics bf not seeing jobs submitted after it gets initial lock
            # if BACKFILL_OPTS["continue"] and job.submit > self.time - BACKFILL_OPTS["interval"]:
            #     continue
            # break if no blocks or only <= min blocks available for immediate backfill
            if max_block_time < min_required_block_time:
                break

            # Empiracal max jobs before reaching bf_max_time
            if sum(partition_num_tested.values()) > BACKFILL_OPTS["max_test_timelimit"]:
                break
            if partition_num_tested[job.partition] >= partition_maxes[job.partition]:
                continue
            partition_num_tested[job.partition] += 1

            reqtime = job.reqtime * self.low_freq_reqtime_factor + self.backfill_opts["resolution"]
            # Only need to plan nodes for jobs that may be relevant to immediate scheduling
            if self.time + reqtime > max_block_time:
                continue

            free_nodes = 0
            selected_intervals = {}

            for interval, nodes in sorted(free_blocks.items(), key=lambda entry: entry[0][0]):
                valid_nodes = {
                    node for node in nodes if self.partitions[job.partition] in node.partitions
                }
                if valid_nodes:
                    selected_intervals[interval] = valid_nodes
                    free_nodes += len(valid_nodes)
                    latest_interval = interval

                if job.nodes <= free_nodes:
                    # usage_block_end = min(selected_intervals.keys(), key=lambda key: key[1])[1]
                    usage_block_start = max(selected_intervals.keys(), key=lambda key: key[0])[0]

                    for interval in list(selected_intervals.keys()):
                        if usage_block_start + reqtime > interval[1]:
                            free_nodes -= len(selected_intervals.pop(interval))

                    if  job.nodes > free_nodes:
                        continue

                    usage_block_end = usage_block_start + reqtime

                    # Remove nodes we don't need from the latest interval added
                    # Don't want to pop randomly for reproducibility between runs
                    for node in sorted(selected_intervals[latest_interval])[:free_nodes-job.nodes]:
                        selected_intervals[latest_interval].remove(node)

                    if usage_block_start == self.time:
                        backfill_now.append(
                            (
                                i_job,
                                { node for nodes in selected_intervals.values() for node in nodes }
                            )
                        )

                    for key, nodes in selected_intervals.items():
                        free_blocks[key] -= nodes

                        # The original ready now block has been broken and the interval needs
                        # redefining
                        if key[0] == self.time:
                            if not free_blocks[key]:
                                free_blocks_ready_intervals.remove((key[0], key[1]))
                            if key[0] != usage_block_start:
                                free_blocks_ready_intervals.add((key[0], usage_block_start))

                        if key[0] != usage_block_start:
                            free_blocks[(key[0], usage_block_start)].update(nodes)
                        if key[1] != usage_block_end:
                            free_blocks[(usage_block_end, key[1])].update(nodes)

                        # These nodes are now reserved, delete block if this leaves nothing left
                        if not free_blocks[key]:
                            free_blocks.pop(key)

                    if not free_blocks_ready_intervals:
                        return backfill_now
                    max_block_time = max(
                        free_blocks_ready_intervals, key=lambda interval: interval[1]
                    )[1]

                    break

        return backfill_now

    def submit_jobs(self, queue : Queue, sched=True, backfill=True):
        partitions_full = set()
        low_freqs = self.low_freq_condition(self.queue_size)

        self.submitted_jobs_step = []

        if sched:
            for reservation, res_queue in queue.reservations.items():
                if not res_queue:
                    continue

                free_nodes = [
                    node for node in self.nodes if (
                        node.reservation == reservation and not node.running_job
                    )
                ]
                while res_queue and res_queue[0].nodes < len(free_nodes):
                    job = res_queue.pop(0)
                    self.running_jobs.append(job.start_job(self.time))
                    self.power_usage += job.true_node_power * job.nodes / 1e+6
                    self.bd_slowdowns.append(
                        max((job.end - job.submit) / max(job.runtime, BD_THRESHOLD), 1)
                    )
                    self.total_energy += (
                        job.true_node_power * job.nodes * job.runtime.total_seconds() / 1e+9
                    )

                    for _ in range(job.nodes):
                        node = free_nodes.pop(0)
                        job.assigned_nodes.append(node)
                        node.running_job = job

                    self.submitted_jobs_step.append(job)

            jobs_submitted = []
            for i_job, job in enumerate(queue.queue):
                if job.partition in partitions_full:
                    continue

                if self.has_space(job):
                    if low_freqs:
                        power_factor, time_factor = self.low_freq_calc.get_factors()
                        job_ready.runtime *= time_factor
                        job_ready.reqtime *= self.low_freq_reqtime_factor
                        job_ready.true_node_power *= power_factor
                    self.submit(job.start_job(self.time))
                    jobs_submitted.append(i_job)
                else:
                    self.partitions[job.partition].set_planned()
                    partitions_full.add(job.partition)
                    if len(partitions_full) == len(self.partitions):
                        break

            for i in sorted(jobs_submitted, reverse=True):
                self.submitted_jobs_step.append(queue.queue.pop(i))

            for partition in self.partitions.values():
                partition.set_unplanned()

        if backfill and queue.queue and self.available_nodes(backfill=True):
            backfill_now = self.get_backfill_jobs(queue)
            for i_job, nodes in backfill_now:
                job_ready = queue.queue[i_job]
                if low_freqs:
                    power_factor, time_factor = self.low_freq_calc.get_factors()
                    job_ready.runtime *= time_factor
                    job_ready.reqtime *= self.low_freq_reqtime_factor
                    job_ready.true_node_power *= power_factor
                if not self.submit(job_ready.start_job(self.time), nodes=nodes, backfill=True):
                    print(
                        self.available_nodes(), self.available_nodes(backfill=True), i,
                        backfill_now
                    )

            # if backfill_now:
            #     print([ (i_job, len(nodes)) for i_job, nodes in backfill_now ])

            for i_job, _ in sorted(backfill_now, key=lambda job_nodes: job_nodes[0], reverse=True):
               self.submitted_jobs_step.append(queue.queue.pop(i_job))


        self.queue_size = len(queue.queue)

    def submit(self, job : Job, nodes=None, backfill=False):
        if self.has_space(job, backfill=backfill):
            self.running_jobs.append(job)
            self.nodes_free -= job.nodes
            self.power_usage += job.true_node_power * job.nodes / 1e+6
            self.bd_slowdowns.append(
                max((job.end - job.submit)/max(job.runtime, BD_THRESHOLD), 1)
            )
            self.total_energy += (
                job.true_node_power * job.nodes * job.runtime.total_seconds() / 1e+9
            )

            if nodes:
                for node in nodes:
                    if node.running_job:
                        raise Exception(" ")
                    if not node.free:
                        raise Exception(" ")
                    job.assign_node(node)

            else:
                for node in self.partitions[job.partition].always_available_nodes:
                    if node.free:
                        if job.assign_node(node):
                            break

            return True

        print("No free nodes, job not submitted")
        return False

    def _check_down_nodes(self):
        try:
            while self.down_nodes and self.down_nodes[0].up_time <= self.time:
                node = self.down_nodes.pop(0)
                was_free = node.free
                node.set_up()
                if not was_free and node.free:
                    self.nodes_free += 1
        except:
            print(self.down_nodes[0].id, self.down_nodes, self.down_nodes[0].up_time, self.time)
            raise TypeError

        while self.node_down_order and self.node_down_order[0].down_schedule[0][0] <= self.time:
            node = self.node_down_order[0]
            # If already down delay this new downtime until the next up to not interfere (this
            # happens because my DOWN implementation waits for current running job to finish)
            if node.down:
                node.down_schedule[0][0] = node.up_time
                self.node_down_order.sort(key=lambda node: node.down_schedule[0][0])
                continue

            down_schedule = node.down_schedule.pop(0)
            if not len(node.down_schedule):
                self.node_down_order.pop(0)
            else:
                self.node_down_order.sort(key=lambda node: node.down_schedule[0][0])

            up_time = self.time + down_schedule[1]

            if node.running_job:
                if down_schedule[2] == "DOWN":
                    up_time += node.running_job.end - self.time
                if up_time <= node.running_job.end:
                    continue

            was_free = node.free
            node.set_down(up_time)
            if was_free and not node.free:
                self.nodes_free -= 1

            self.down_nodes.append(node)
            self.down_nodes.sort(key=lambda node: node.up_time)

    def _check_reservations(self):
        while self.reserved_nodes and self.reserved_nodes[0].unreserved_time <= self.time:
            node = self.reserved_nodes.pop(0)
            was_free = node.free
            node.set_unreserved()
            if not was_free and node.free:
                self.nodes_free += 1

        while (
            self.node_reservation_order and
            self.node_reservation_order[0].reservation_schedule[0][0] <= self.time
        ):
            node = self.node_reservation_order[0]

            reservation_schedule = node.reservation_schedule.pop(0)
            if not len(node.reservation_schedule):
                self.node_reservation_order.pop(0)
            else:
                self.node_reservation_order.sort(key=lambda node: node.reservation_schedule[0][0])

            was_free = node.free
            node.set_reserved(reservation_schedule[2], reservation_schedule[1])
            if was_free and not node.free:
                self.nodes_free -= 1

            self.reserved_nodes.append(node)
            self.reserved_nodes.sort(key=lambda node: node.unreserved_time)

    def step(self, time):
        self.time = time

        self._check_down_nodes()
        self._check_reservations()

        self.running_jobs.sort(key=lambda job: job.end)

        self.finished_jobs_step = []
        while self.running_jobs and self.running_jobs[0].end <= self.time:
            self.finished_jobs_step.append(self.running_jobs.pop(0))
            self.finished_jobs_step[-1].end_job()
            self.job_history.append(self.finished_jobs_step[-1])
            self.nodes_free += sum(
                1 for node in self.finished_jobs_step[-1].assigned_nodes if node.free
            )
            self.power_usage -= (
                self.finished_jobs_step[-1].true_node_power *
                self.finished_jobs_step[-1].nodes /
                1e+6
            )

        self.idle_history.append(self.available_nodes())
        # XXX Need reservations in denominator
        # self.occupancy_history.append(
        #     1 -
        #     (
        #         self.available_nodes() /
        #         (5860 - sum(1 for node in self.down_nodes if not node.free))
        #     )
        # )
        self.power_history.append(self.power_usage)
        self.queue_size_history.append(self.queue_size)
        self.times.append(self.time)

        return self.finished_jobs_step # Need this for fair share usage accounting


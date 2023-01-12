from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

from globals import *


class Job():
    def __init__(
        self, id, submit : datetime, nodes, runtime : timedelta, reqtime: timedelta, node_power,
        true_node_power, true_job_start, user, account, qos, partition, dependency_arg, name
    ):
        self.id = id
        self.nodes = nodes
        self.runtime = runtime
        self.reqtime = reqtime
        self.node_power = node_power
        self.true_node_power = true_node_power
        self.submit = submit
        self.true_job_start = true_job_start
        self.user = user
        self.account = account
        self.qos = qos
        self.partition = partition
        self.name = name

        self.dependency = Dependency(dependency_arg, user, name) if dependency_arg else None

        self.launch_time = submit
        self.start = None
        self.end = None
        self.assigned_nodes = []

    def start_job(self, time : datetime):
        self.start = time
        self.end = time + self.runtime
        self.endlimit = time + self.reqtime

        self.qos.job_started(self.user, self.nodes)
        return self

    def end_job(self):
        for node in self.assigned_nodes:
            node.set_free()
        self.qos.job_ended(self.user, self.nodes)
        return self

    def relaunch(self, time):
        self.launch_time = time
        return self


class Queue():
    def __init__(self, df_jobs, init_time, priority_sorter):
        self.priority_sorter = priority_sorter
        self.time = init_time

        self.system = None

        # hardcoded for now
        # Don't have a proper way to simulate reservations as I can't see a history of deleted ones
        self.qoss = {
            "normal" : QOS("normal", 0, -1, -1, -1, -1),
            "reservation" : QOS("reservation", 1, -1, -1, -1, -1),
            "taskfarm" : QOS("taskfarm", 1, -1, 128, 512, 32),
            "standard" : QOS("standard", 1, -1, -1, 1024, 16),
            "short" : QOS("short", 1, -1, -1, 32, 4),
            "long" : QOS("long", 1, 2048, -1, 512, 16),
            "largescale" : QOS("largescale", 1, -1, -1, -1, 1),
            "highmem" : QOS("highmem", 1, -1, -1, 256, 16),
            "lowpriority" : QOS("lowpriority", 0.002, -1, -1, 2048, 16)
        }

        self.all_jobs = [
            Job(
                job_row.JobID, job_row.Submit, job_row.AllocNodes, job_row.Elapsed,
                job_row.Timelimit, job_row.PowerPerNode, job_row.TruePowerPerNode, job_row.Start,
                job_row.User, job_row.Account, self.qoss[job_row.QOS], job_row.Partition,
                job_row.DependencyArg, job_row.JobName
            ) for _, job_row in df_jobs.sort_values("Submit").iterrows()
        ]
        self._verify_dependencies()
        self.queue = []

        self.waiting_dependency = []

    def _verify_dependencies(self):
        all_ids = { job.id for job in self.all_jobs }
        for job in self.all_jobs:
            if not job.dependency:
                continue

            for dep_type, ids in job.dependency.conditions.items():
                job.dependency.conditions[dep_type] = ids - all_ids

            job.dependency.conditions_met = not any(job.dependency.conditions.values())
            if job.dependency.conditions_met and not job.dependency.singleton:
                job.dependency = None
                print("removed")

    def set_system(self, system):
        self.system = system

    def check_dependencies(self):
        for i_job, job in enumerate(self.waiting_dependency):
            job.dependency.update_finished(self.system.finished_jobs_step)
            job.dependency.update_submitted(self.system.submitted_jobs_step)
            if job.dependency.can_release(self.queue, self.system.running_jobs):
                self.queue.append(self.waiting_dependency.pop(i_job).relaunch(self.time))
                print("released")

    def step(self, t_step, retained):
        self.check_dependencies()

        self.time += t_step

        if self.time < self.next_newjob():
            return

        try:
            while self.all_jobs[0].submit <= self.time:
                new_job = self.all_jobs.pop(0)
                if new_job.dependency:
                    new_job.dependency.update_finished(self.system.job_history)
                    new_job.dependency.update_submitted(
                        self.system.running_jobs + self.system.job_history
                    )
                    if not new_job.dependency.can_release(self.queue, self.system.running_jobs):
                        self.waiting_dependency.append(new_job)
                self.queue.append(new_job)
        except IndexError: # No more new jobs
            pass

        self.queue[retained:] = self.priority_sorter.sort(self.queue[retained:], self.time)

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
            job_ids = { int(job_id) for job_id in condition.split(":")[1:] }
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
        if not self.finished_relevant or self.conditions_met:
            return

        for job in jobs:
            if job.id in self.conditions["afterany"]:
                self.conditions["afterany"].remove(job.id)
                if self.delimiter == "?":
                    self.conditions_met = True
                    return

    def update_submitted(self, jobs):
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

        self.num_available = 0

    def add_node(self, node):
        node.partitions.append(self)
        self.nodes.append(node)
        self.nodes.sort(key=lambda node: node.weight) # Small weights get priority
        self.num_available += node.free

    def available_nodes(self):
        return self.num_available


class QOS():
    def __init__(self, name, priority, grp_nodes, grp_jobs, usr_nodes, usr_jobs):
        self.name = name
        self.priority = priority

        # TODO This is dumb, implement in a way that means in these cases the quota is just not tracked
        grp_jobs = 1000000000000 if grp_jobs == -1 else grp_jobs
        grp_nodes = 1000000000000 if grp_nodes == -1 else grp_nodes
        usr_jobs = 1000000000000 if usr_jobs == -1 else usr_jobs
        usr_nodes = 1000000000000 if usr_nodes == -1 else usr_nodes

        self.grp_quotas = []
        self.job_quota_remaining = grp_jobs
        self.node_quota_remaining = grp_nodes

        self.usr_job_quota_remaining = defaultdict(lambda: usr_jobs)
        self.usr_node_quota_remaining = defaultdict(lambda: usr_nodes)

    def job_started(self, user, nodes):
        self.job_quota_remaining -= 1
        self.node_quota_remaining -= nodes
        self.usr_job_quota_remaining[user] -= 1
        self.usr_node_quota_remaining[user] -= nodes

    def job_ended(self, user, nodes):
        self.job_quota_remaining += 1
        self.node_quota_remaining += nodes
        self.usr_job_quota_remaining[user] += 1
        self.usr_node_quota_remaining[user] += nodes

    def hold_job(self, user, nodes):
        return (
            not self.job_quota_remaining or nodes > self.node_quota_remaining or
            not self.usr_job_quota_remaining[user] or nodes > self.usr_node_quota_remaining[user]
        )


class Node():
    def __init__(self, num, weight=0):
        self.id = num
        self.weight = weight
        self.free = True

        self.partitions = []

    def set_free(self):
        self.free = True
        for partition in self.partitions:
            partition.num_available += 1

    def set_busy(self):
        self.free = False
        for partition in self.partitions:
            partition.num_available -= 1


class Archer2():
    def __init__(
        self, init_time : datetime, node_down_mean=0, backfill_opts={},
        low_freq_condition=lambda queue: False, low_freq_calc=None, low_freq_reqtime_factor=1.0
    ):
        self.power_usage = 0 # MW
        self.node_down_mean = node_down_mean
        self.init_time = init_time
        self.time = init_time
        self.backfill_opts = backfill_opts
        if "min_block_width" not in self.backfill_opts:
            self.backfill_opts["min_block_width"] = timedelta(minutes=1) # 1min for Archer2
        if "max_job_test" not in self.backfill_opts:
            self.backfill_opts["max_job_test"] = 1000 # 1000 for Archer2
        self.low_freq_condition = low_freq_condition
        self.low_freq_calc = low_freq_calc
        self.low_freq_reqtime_factor = low_freq_reqtime_factor

        self.running_jobs = []

        self.power_history = [self.power_usage] # MW
        self.occupancy_history = [0] # %
        self.queue_size = 0
        self.queue_size_history = [0]
        self.times = [self.time]
        self.bd_slowdowns = []
        self.job_history = []
        self.finished_jobs_step = []
        self.submitted_jobs_step = []
        self.total_energy = 0.0 # GJ

        self.partitions = {
            "standard" : Partition("standard", 1, 1.0), "highmem" : Partition("highmem", 1, 1.0)
        }
        self.nodes = []
        for i in range(5276):
            self.nodes.append(Node(i, 0))
            self.partitions["standard"].add_node(self.nodes[-1])
        for i in range(5276, 5860):
            self.nodes.append(Node(i, 1000))
            self.partitions["standard"].add_node(self.nodes[-1])
            self.partitions["highmem"].add_node(self.nodes[-1])

        self.nodes_free = 5860
        self.nodes_drained = 0
        self.nodes_drained_carryover = 0

    def has_space(self, job : Job):
        return True if self.available_nodes(job.partition) >= job.nodes else False

    def available_nodes(self, partition=""):
        if partition:
            return self.partitions[partition].available_nodes()

        return self.nodes_free

        # return self.nodes_free - self.nodes_drained

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
                if shadow_time <= self.time + self.backfill_opts["min_block_width"] and not extra_nodes:
                    return backfill_now

                pass

            return backfill_now

        free_blocks = defaultdict(set)
        free_blocks[(self.time, datetime.max)] = { node for node in self.nodes if node.free }
        for job in self.running_jobs:
            # Some jobs exceeding timelimit and some with zero reqtime
            if job.endlimit <= self.time:
                job.endlimit = self.time + 0.1 * job.reqtime + timedelta(minutes=1)
            free_blocks[(job.endlimit, datetime.max)].update(job.assigned_nodes)

        min_required_block_time = max( # min
            self.time + self.backfill_opts["min_block_width"],
            self.time + (
                min(queue.queue, key=lambda job: job.reqtime).reqtime *
                self.low_freq_reqtime_factor
            )
        )

        free_blocks_ready_intervals = (
            { (self.time, datetime.max) } if self.available_nodes() else set()
        )
        max_block_time = datetime.max
        partition_num_tested = defaultdict(int)
        # Dont consider jobs that require partitions with no available nodes
        partition_maxes = {
            partition : (
                self.backfill_opts["max_job_test"] if self.available_nodes(partition) else 0
            ) for partition in self.partitions.keys()
        }
        for i_job, job in enumerate(queue.queue):
            if job.qos.hold_job(job.user, job.nodes):
                continue
            # break if no blocks or only <= min blocks available for immediate backfill
            if not free_blocks_ready_intervals:
                break
            max_block_time = max(free_blocks_ready_intervals, key=lambda interval: interval[1])[1]
            if max_block_time < min_required_block_time:
                break

            if partition_num_tested[job.partition] >= partition_maxes[job.partition]:
                continue
            partition_num_tested[job.partition] += 1

            reqtime = job.reqtime * self.low_freq_reqtime_factor
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

                if job.nodes <= free_nodes:
                    usage_block_end = min(selected_intervals.keys(), key=lambda key: key[1])[1]
                    usage_block_start = max(selected_intervals.keys(), key=lambda key: key[0])[0]

                    # Nodes do not remain available for this jobs runtime, find new block
                    if usage_block_start + reqtime > usage_block_end:
                        selected_intervals = {}
                        free_nodes = 0
                        continue

                    usage_block_end = usage_block_start + reqtime

                    if usage_block_start == self.time:
                        backfill_now.append(i_job)

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

                    break

        return backfill_now

    def submit_jobs(self, queue : Queue):
        partitions_full = set()
        low_freqs = self.low_freq_condition(self.queue_size)

        self.submitted_jobs_step = []

        jobs_submitted = []
        for i_job, job in enumerate(queue.queue):
            if job.partition in partitions_full:
                continue

            if self.has_space(job):
                if job.qos.hold_job(job.user, job.nodes):
                    continue
                if low_freqs:
                    power_factor, time_factor = self.low_freq_calc.get_factors()
                    job_ready.runtime *= time_factor
                    job_ready.reqtime *= self.low_freq_reqtime_factor
                    job_ready.true_node_power *= power_factor
                self.submit(job.start_job(self.time))
                jobs_submitted.append(i_job)
            else:
                partitions_full.add(job.partition)
                if len(partitions_full) == len(self.partitions):
                    break

        for i in sorted(jobs_submitted, reverse=True):
            self.submitted_jobs_step.append(queue.queue.pop(i))

        queue.check_dependencies()

        if queue.queue and self.available_nodes():
            backfill_now = self.get_backfill_jobs(queue)
            problemo = False
            for i in backfill_now:
                job_ready = queue.queue[i]
                if low_freqs:
                    power_factor, time_factor = self.low_freq_calc.get_factors()
                    job_ready.runtime *= time_factor
                    job_ready.reqtime *= self.low_freq_reqtime_factor
                    job_ready.true_node_power *= power_factor
                problemo = not self.submit(job_ready.start_job(self.time))
                if problemo:
                    print(self.available_nodes(), i, backfill_now)
            if problemo:
                print("=====")

            for i in sorted(backfill_now, reverse=True):
               self.submitted_jobs_step.append((queue.queue.pop(i)))

        self.queue_size = len(queue.queue)

    def submit(self, job : Job):
        if self.has_space(job):
            self.running_jobs.append(job)
            self.nodes_free -= job.nodes
            self.power_usage += job.true_node_power * job.nodes / 1e+6
            self.bd_slowdowns.append(
                max((job.end - job.launch_time)/max(job.runtime, BD_THRESHOLD), 1)
            )
            self.total_energy += (
                job.true_node_power * job.nodes * job.runtime.total_seconds() / 1e+9
            )

            for node in self.partitions[job.partition].nodes:
                if node.free:
                    node.set_busy()
                    job.assigned_nodes.append(node)
                    if len(job.assigned_nodes) >= job.nodes:
                        break

            return True

        print("No free nodes, job not submitted")
        return False

    def step(self, t_step : timedelta):
        self.time += t_step

        self.running_jobs.sort(key=lambda job: job.end)

        self.finished_jobs_step = []
        while self.running_jobs and self.running_jobs[0].end <= self.time:
            self.finished_jobs_step.append(self.running_jobs.pop(0))
            self.finished_jobs_step[-1].end_job()
            self.job_history.append(self.finished_jobs_step[-1])
            self.nodes_free += self.finished_jobs_step[-1].nodes
            self.power_usage -= (
                self.finished_jobs_step[-1].true_node_power *
                self.finished_jobs_step[-1].nodes /
                1e+6
            )

        # Resample drained nodes every 12 hour at most
        if self.time.hour != (self.time - t_step).hour and not self.time.hour % 12:
            num_drain = max(
                (
                    round(np.random.normal(
                        loc=self.node_down_mean, scale=self.node_down_mean / 2
                    )) +
                    self.nodes_drained_carryover
                ),
                0
            )
            if num_drain <= self.nodes_free:
                self.nodes_drained = num_drain
                self.nodes_drained_carryover = 0
            else:
                self.nodes_drained = self.nodes_free
                self.nodes_drained_carryover = num_drain - self.nodes_free

        self.occupancy_history.append(1 - (self.available_nodes()/(5860 - self.nodes_drained)))
        self.power_history.append(self.power_usage)
        self.queue_size_history.append(self.queue_size)
        self.times.append(self.time)

        return self.finished_jobs_step # Need this for fair share usage accounting


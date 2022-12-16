from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

from globals import *


class Job():
    def __init__(
        self, submit : datetime, nodes, runtime : timedelta, reqtime: timedelta, node_power,
        true_node_power, true_job_start
    ):
        self.nodes = nodes
        self.runtime = runtime
        self.reqtime = reqtime
        self.node_power = node_power
        self.true_node_power = true_node_power
        self.submit = submit
        self.true_job_start = true_job_start

        self.start = None
        self.end = None

    def start_job(self, time : datetime):
        self.start = time
        self.end = time + self.runtime
        self.endlimit = time + self.reqtime
        return self


class Queue():
    def __init__(self, df_jobs, init_time, priority_sorter):
        self.priority_sorter = priority_sorter
        self.time = init_time

        self.all_jobs = [
            Job(
                job_row.Submit, job_row.AllocNodes, job_row.Elapsed, job_row.Timelimit,
                job_row.PowerPerNode, job_row.TruePowerPerNode, job_row.Start
            ) for _, job_row in df_jobs.sort_values("Submit").iterrows()
        ]
        self.queue = []

    def step(self, t_step, retained):
        self.time += t_step

        if self.time < self.next_newjob():
            return

        try:
            while self.all_jobs[0].submit <= self.time:
                self.queue.append(self.all_jobs.pop(0))
        except IndexError:
            pass

        self.queue[retained:] = self.priority_sorter.sort(self.queue[retained:], self.time)

    def next_newjob(self):
        try:
            return self.all_jobs[0].submit
        except IndexError:
            return datetime.max


class Archer2():
    def __init__(
        self, init_time : datetime, baseline_power=1500, slurmtocab_factor=1.0, node_down_mean=0,
        backfill_opts={}, low_freq_condition=lambda queue: False, low_freq_calc=None,
        low_freq_reqtime_factor=1.0
    ):
        self.power_usage = 0 # MW
        self.slurmtocab_factor = slurmtocab_factor
        self.node_down_mean = node_down_mean
        self.init_time = init_time
        self.time = init_time
        self.backfill_opts = backfill_opts
        if "min_block_width" not in self.backfill_opts:
            self.backfill_opts["min_block_width"] = timedelta(minutes=5) # 1min for Archer2
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

        self.nodes_free = 5860
        self.nodes_drained = 0
        self.nodes_drained_carryover = 0
        self.sorted = False

    def has_space(self, job : Job):
        return True if self.available_nodes() >= job.nodes else False

    def available_nodes(self):
        return self.nodes_free - self.nodes_drained

    def next_event(self):
        if not self.running_jobs:
            return datetime.max

        if not self.sorted:
            self.running_jobs.sort(key=lambda job: job.end)
            self.sorted = True

        return self.running_jobs[0].end

    def get_backfill_jobs(self, queue : Queue):
        backfill_now = []

        free_blocks = defaultdict(int)
        free_blocks[(self.time, datetime.max)] = self.available_nodes()
        for job in self.running_jobs:
            # Some jobs exceeding timelimit and some with zero reqtime
            if job.endlimit <= self.time:
                job.endlimit = self.time + 0.1 * job.reqtime + timedelta(minutes=1)
            free_blocks[(job.endlimit, datetime.max)] += job.nodes

        min_required_block_time = min(
            self.time + self.backfill_opts["min_block_width"],
            self.time + (
                min(queue.queue, key=lambda job: job.reqtime).reqtime *
                self.low_freq_reqtime_factor
            )
        )

        # max_block_time = datetime.max
        free_blocks_ready_intervals = (
            { (self.time, datetime.max) } if self.available_nodes() else set()
        )
        max_block_time = datetime.max
        for i_job, job in enumerate(
            list(queue.queue)[:max(len(queue.queue), self.backfill_opts["max_job_test"])]
        ):
            reqtime = job.reqtime * self.low_freq_reqtime_factor
            # Only need to plan nodes for jobs that may be relevant to immediate scheduling
            if self.time + reqtime > max_block_time:
                continue

            free_nodes = 0
            selected_intervals = {}

            # break if no blocks or only <= min blocks available for immediate backfill
            if not free_blocks_ready_intervals:
                break
            max_block_time = max(free_blocks_ready_intervals, key=lambda interval: interval[1])[1]
            if max_block_time < min_required_block_time:
                break

            for interval, nodes in sorted(free_blocks.items(), key=lambda entry: entry[0][0]):
                selected_intervals[interval] = nodes
                free_nodes += nodes

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

                    for key in selected_intervals.keys():
                        if key[0] == self.time:
                            free_blocks_ready_intervals.remove((key[0], key[1]))
                            if key[0] != usage_block_start:
                                free_blocks_ready_intervals.add((key[0], usage_block_start))

                        if key[0] != usage_block_start:
                            free_blocks[(key[0], usage_block_start)] += free_blocks[key]
                        if key[1] != usage_block_end:
                            free_blocks[(usage_block_end, key[1])] += free_blocks[key]

                        free_blocks.pop(key)

                    if free_nodes - job.nodes:
                        free_blocks[(usage_block_start, usage_block_end)] += free_nodes - job.nodes
                        if usage_block_start == self.time:
                            free_blocks_ready_intervals.add((key[0], usage_block_end))

                    break

        return backfill_now

    def submit_jobs(self, queue : Queue):
        low_freqs = self.low_freq_condition(queue)
        for job in list(queue.queue):
            if self.has_space(job):
                if low_freqs:
                    power_factor, time_factor = self.low_freq_calc.get_factors()
                    queue.queue[0].runtime *= time_factor
                    queue.queue[0].true_node_power *= power_factor
                self.submit(queue.queue.pop(0).start_job(self.time))
            else:
                break

        if queue.queue and self.available_nodes():
            backfill_now = self.get_backfill_jobs(queue)
            for i in sorted(backfill_now, reverse=True):
                if low_freqs:
                    power_factor, time_factor = self.low_freq_calc.get_factors()
                    queue.queue[i].runtime *= max(time_factor, 1)
                    queue.queue[i].true_node_power *= min(power_factor, 1)
                if not self.submit(queue.queue.pop(i).start_job(self.time)):
                    print(self.available_nodes(), i, backfill_now)

        self.queue_size = len(queue.queue)

    def submit(self, job : Job):
        if self.has_space(job):
            self.running_jobs.append(job)
            self.nodes_free -= job.nodes
            self.power_usage += (job.true_node_power * job.nodes * self.slurmtocab_factor) / 1e+6
            self.bd_slowdowns.append(
                max((job.end - job.submit)/max(job.runtime, BD_THRESHOLD), 1)
            )
            self.job_history.append(job)
            self.sorted = False
            return True
        else:
            print("No free nodes, job not submitted")
            return False

    def step(self, t_step : timedelta):
        self.time += t_step

        if not self.sorted:
            self.running_jobs.sort(key=lambda job: job.end)
            self.sorted = True

        while self.running_jobs and self.running_jobs[0].end <= self.time:
            job = self.running_jobs.pop(0)
            self.nodes_free += job.nodes
            self.power_usage -= (job.true_node_power * job.nodes * self.slurmtocab_factor) / 1e+6

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

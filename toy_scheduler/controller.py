import os, time, copy
import datetime; from datetime import timedelta
from collections import defaultdict, Counter
import dill as pickle

import pandas as pd

from config import get_config
from partition import Partitions
from job_queue import Queue
from priority_sorters import MFPrioritySorter
from fairshare import FairTree


# TODO Currently nodes can only be in one reservation's free blocks, so if a node is going
# unreserved soon the backfiller for no reservation jobs will not be see it

# TODO Implement REPLACE_DOWN, this is used for the shortqos reservation. Shouldn't, be too hard
# the current implementation. I think the slurm implementation only replaces the nodes if there are
# idle nodes at that moment of attempting to schedule on that node? This is not how my scheduler
# works so I will just replace at the moment of going only if there are idle nodes at that time.
# This should in practice be pretty much the same

# TODO Job requeuing for down nodes.


class Controller:
    def __init__(self, config_file):
        self.config = get_config(config_file)

        self.partitions = Partitions(
            self.config.slurm_conf, self.config.considered_partitions,
            self.config.node_events_dump, self.config.reservations_dump
        )

        self.queue = Queue(
            self.config.job_dump, self.partitions, self.config.qos_dump,
            self.config.considered_partitions
        )
        self.init_time = self.queue.time
        self.time = self.queue.time

        active_usrs = { job.user for job in self.queue.all_jobs }
        self.fairtree = FairTree(
            self.config.assocs_dump, self.config.PriorityCalcPeriod,
            self.config.PriorityDecayHalfLife, self.init_time, active_usrs,
            self.config.approx_excess_assocs, self.partitions
        )
        priority_sorter = MFPrioritySorter(
            self.init_time, self.config.PriorityWeightJobSize, self.config.PriorityWeightAge,
            self.config.PriorityWeightFairshare, self.config.PriorityMaxAge,
            self.config.PriorityWeightPartition, self.config.PriorityWeightQOS,
            len({ partition.priority_tier for partition in self.partitions.partitions }) == 1
        )
        priority_sorter.fairtree = self.fairtree
        self.queue.set_priority_sorter(priority_sorter)

        self.num_sched_test_step = 0
        self.num_bf_test_step = 0

        self.times = [self.time]
        self.power_usage = 0
        self.total_energy = 0.0

        self.job_history = []
        self.running_jobs = []

        self.down_nodes = []
        self.node_down_order = sorted(
            [ node for node in self.partitions.nodes if node.down_schedule ],
            key=lambda node: node.down_schedule[-1][0],
            reverse=True
        )
        self.reserved_nodes = []
        self.node_reservation_order = sorted(
            [ node for node in self.partitions.nodes if node.reservation_schedule ],
            key=lambda node: node.reservation_schedule[-1][0],
            reverse=True
        )

        # [next_event, submitted, cleared, start, end, nodes, name]
        if self.config.hpe_restrictlongjobs_sliding_reservations == "const":
            self.sliding_reservations = [
                [
                    submitted, submitted, submitted + timedelta(hours=1),
                    submitted + timedelta(hours=1, minutes=5),
                    submitted + timedelta(days=365, hours=1, minutes=5),
                    self.partitions.hpe_restrictlong_nodes, "HPE_RestrictLongJobs"
                ] for submitted in [
                    (
                        self.init_time.replace(minute=0, second=0) - timedelta(minutes=5) +
                        timedelta(hours=hr_num)
                    ) for hr_num in (
                        range(int(
                            (
                                max(
                                    self.queue.all_jobs,
                                    key=lambda job: job.true_job_start
                                ).true_job_start -
                                self.time
                            ).total_seconds() /
                            (60 * 60)
                        ))
                    )
                ]
            ]
            self.sliding_reservations.sort(key=lambda res: res[0], reverse=True)

        elif self.config.hpe_restrictlongjobs_sliding_reservations:
            with open(self.config.hpe_restrictlongjobs_sliding_reservations, "rb") as f:
                hpe_restrictlong_res = pickle.load(f)

            for submitted in sorted(hpe_restrictlong_res):
                print("{}: {}".format(submitted, len(hpe_restrictlong_res[submitted])))

            nid_to_node = { node.id : node for node in self.partitions.nodes }
            self.sliding_reservations = [
                [
                    submitted, submitted, submitted + timedelta(hours=1),
                    submitted + timedelta(hours=1, minutes=5),
                    submitted + timedelta(days=365, hours=1, minutes=5),
                    [ nid_to_node[nid] for nid in hpe_restrictlong_res[submitted] ],
                    "HPE_RestrictLongJobs"
                ] for submitted in sorted(hpe_restrictlong_res)
            ]

        else:
            self.sliding_reservations = []

        self.step_cnt = 0
        self.previous_print_hour = self.time.hour

    def _next_job_finish(self):
        if not self.running_jobs:
            return datetime.datetime.max

        return self.running_jobs[-1].end

    def run_sim(self, max_steps=0):
        # import numpy as np
        sim_start = time.time()

        times_sched, times_bf, times_sched_bf = [], [], []

        # TODO Non-defer implementation
        previous_small_sched = self.time
        next_bf_time = self.time + self.config.bf_interval
        next_sched_time = self.time + self.config.sched_interval
        next_fairtree_time = self.time + self.config.PriorityCalcPeriod
        while self.queue.all_jobs or self.queue.queue or self.running_jobs:
            # start = time.time()
            self.time = min(
                next_bf_time, next_sched_time, next_fairtree_time, self._next_job_finish(),
                self.queue.next_newjob()
            )

            sched, sched_depth, bf, fairtree = False, None, False, False

            if self.time == next_fairtree_time:
                fairtree = True
                next_fairtree_time += self.config.PriorityCalcPeriod

            if self.time == next_sched_time:
                sched = True
                next_sched_time += self.config.sched_interval
            elif (
                self.time > previous_small_sched + self.config.sched_min_interval and
                self.time == self._next_job_finish()
            ):
                sched = True
                sched_depth = self.config.default_queue_depth
                previous_small_sched = self.time

            if self.time == next_bf_time:
                bf = True
                next_bf_time += self.config.bf_interval

            self._step(sched, sched_depth, bf, fairtree)
            self.step_cnt += 1

            if max_steps and self.step_cnt > max_steps:
                break
            # if bf and not sched:
            #     times_bf.append(time.time() - start)
            # if sched and not bf:
            #     times_sched.append(time.time() - start)
            # if sched and bf:
            #     times_sched_bf.append(time.time() - start)
            # if self.step_cnt % 1000 == 0:
            #     print("bf: ", np.mean(times_bf))
            #     print("sched: ", np.mean(times_sched))
            #     print("sched & bf: ", np.mean(times_sched_bf))

        elapsed = time.time() - sim_start
        print(
            "Sim complete in {} hr {} mins".format(
                int(elapsed // (60 * 60)), int((elapsed % (60 * 60)) // 60)
            )
        )

    def _step(self, sched, sched_depth, bf, fairtree):
        self._check_finished_jobs()

        self._check_down_nodes()
        self._check_reservations()

        self.queue.step(self.time, self.running_jobs)

        if fairtree:
            self.fairtree.fairshare_calc(self.running_jobs, self.time)

        pre_submit_running_jobs_len = len(self.running_jobs)

        if sched:
            self.num_sched_test_step = 0
            sched_depth = sched_depth if sched_depth is not None else (
                sum(len(res_queue) for res_queue in self.queue.reservations.values()) +
                len(self.queue.queue)
            )

            self._sched_reservations(sched_depth)

            self._sched_main(sched_depth)

        if bf:
            self.partitions.clean_free_blocks(self.time, self.config.bf_resolution)

            self.num_bf_test_step = 0

            self._backfill()

        if len(self.running_jobs) != pre_submit_running_jobs_len:
            self.running_jobs.sort(key=lambda job: job.end, reverse=True)

        self.times.append(self.time)

        self._print_stats(sched, bf)

    def _submit(self, job, nodes=None):
        # NOTE Submit assumes that the current job's reservations have been removed before being
        # called. The idea is a job releases its reservations before scheduling is attempted and
        # gets new/the same reservations if it was not able to run at that moment
        self.running_jobs.append(job)
        self.power_usage += job.true_node_power * job.nodes / 1e+6
        self.total_energy += (
            job.true_node_power * job.nodes * job.runtime.total_seconds() / 1e+9
        )

        if job.assigned_nodes:
            raise Exception("UUGGHGHG")

        if nodes:
            for node in nodes:
                self.partitions.remove_free_block(node)
                node.interval_times[0] = job.endlimit
                self.partitions.add_free_block(node)
                job.assign_node(node)

        else:
            raise NotImplementedError()

        if len(job.assigned_nodes) != job.nodes:
            raise Exception("bruh")

        return True

    def _check_finished_jobs(self):
        while self.running_jobs and self.running_jobs[-1].end <= self.time:
            job = self.running_jobs.pop()
            self._end_job(job)
            self.fairtree.job_finish_usage_update(job)
            self.job_history.append(job)
            self.power_usage -= job.true_node_power * job.nodes / 1e+6
            self.total_energy += (
                job.true_node_power * job.nodes * job.runtime.total_seconds() / 1e+9
            )

    def _end_job(self, job):
        job.end_job()
        # Down nodes dont exist to free_blocks
        for node in job.assigned_nodes:
            if node.down:
                continue
            self.partitions.remove_free_block(node)
            node.interval_times[0] = self.time
            self.partitions.add_free_block(node)

    def _sched_reservations(self, sched_depth):
        for reservation, res_queue in self.queue.reservations.items():
            if not res_queue:
                continue

            # Reservation ended, delete stray jobs. Could be a problem if the same reservation
            # comes back but *shrug*
            # NOTE Once reservation is finished all intervals should've been popped
            if not self.partitions.free_blocks[reservation]:
                print(
                    "!!!\nReservation {} has no nodes, deleting {} job\n!!!".format(
                        reservation, len(self.queue.reservation[reservation])
                    )
                )
                self.queue.reservations[reservation] = []
                continue

            free_nodes_ready_now = {
                node
                for interval, nodes in self.partitions.free_blocks[reservation].items()
                    if interval[0] <= self.time
                    for node in nodes
                        if node.running_job is None
            }

            if not free_nodes_ready_now:
                continue

            max_job_end = max(
                free_nodes_ready_now, key=lambda node: node.interval_times[1]
            ).interval_times[1]

            jobs_submitted = []

            for i_job_rev, job in enumerate(reversed(res_queue)):
                if self.num_sched_test_step >= sched_depth:
                    continue
                self.num_sched_test_step += 1

                job_end = self.time + job.reqtime

                # Check for plnd nodes that the job could run on now
                valid_nodes = {
                    node
                    for node in self._yield_planned_nodes_ready_now(job, job_end)
                        if node in free_nodes_ready_now
                }
                n_nodes = len(valid_nodes)

                # Planned nodes either don't exist or have been messed up by overrunning jobs and
                # there are no more suitable nodes
                if n_nodes < job.nodes and job_end > max_job_end:
                    if n_nodes:
                        for node in valid_nodes:
                            free_nodes_ready_now.remove(node)
                        if not free_nodes_ready_now:
                            break
                        max_job_end = max(
                            free_nodes_ready_now, key=lambda node: node.interval_times[1]
                        ).interval_times[1]
                    continue


                if n_nodes:
                    max_node = max(valid_nodes, key=lambda node: (node.weight, node.id))
                    max_node_weight = (max_node, (max_node.weight, max_node.id))
                else:
                    max_node_weight = None

                for node in free_nodes_ready_now:
                    if job_end > node.interval_times[1]:
                        continue

                    if not n_nodes:
                        valid_nodes.add(node)
                        n_nodes += 1
                        max_node_weight = node, (node.weight, node.id)
                        continue
                    # Have enough nodes now, so only accept more favourable nodes
                    if n_nodes >= job.nodes:
                        node_weight = (node.weight, node.id)
                        if node_weight > max_node_weight[1]:
                            continue
                        valid_nodes.remove(max_node_weight[0])
                        max_node_weight = (node, node_weight)
                    valid_nodes.add(node)
                    n_nodes += 1

                for node in valid_nodes:
                    free_nodes_ready_now.remove(node)

                if len(valid_nodes) == job.nodes:
                    self.partitions.clear_planned_blocks(job)

                    self._submit(job.start_job(self.time), nodes=valid_nodes)
                    jobs_submitted.append(-(i_job_rev + 1))

                if not free_nodes_ready_now:
                    break

                max_job_end = max(
                    free_nodes_ready_now, key=lambda node: node.interval_times[1]
                ).interval_times[1]

            for i_job in reversed(jobs_submitted):
                res_queue.pop(i_job)

    def _sched_main(self, sched_depth):
        free_nodes_ready_now = {
            node
            for interval, nodes in self.partitions.free_blocks[""].items()
                if interval[0] <= self.time
                for node in nodes
                    if node.running_job is None
        }
        if not free_nodes_ready_now:
            return
        max_job_end = max(
            free_nodes_ready_now, key=lambda node: node.interval_times[1]
        ).interval_times[1]

        jobs_submitted = []

        for i_job_rev, job in enumerate(reversed(self.queue.queue)):
            if self.num_sched_test_step >= sched_depth:
                break
            self.num_sched_test_step += 1

            job_end = self.time + job.reqtime

            # Check for plnd nodes that the job could run on now
            # NOTE Reason for "if node in free_nodes_ready_now". Some of nodes may have been given
            # to a higher prio job earlier in this loop so shouldn't be considered. Although the
            # job they were given to  didn't get enough nodes to run and so interval_times has not
            # been updated, we should consider these nodes spoken for
            valid_nodes = {
                node
                for node in self._yield_planned_nodes_ready_now(job, job_end)
                    if node in free_nodes_ready_now
            }
            n_nodes = len(valid_nodes)

            # Planned nodes either don't exist or have been messed up by overrunning jobs and there
            # are no more suitable nodes
            if n_nodes < job.nodes and job_end > max_job_end:
                if n_nodes:
                    for node in valid_nodes:
                        free_nodes_ready_now.remove(node)
                    if not free_nodes_ready_now:
                        break
                    max_job_end = max(
                        free_nodes_ready_now, key=lambda node: node.interval_times[1]
                    ).interval_times[1]
                continue

            if n_nodes:
                max_node = max(valid_nodes, key=lambda node: (node.weight, node.id))
                max_node_weight = (max_node, (max_node.weight, max_node.id))
            else:
                max_node_weight = None

            for node in free_nodes_ready_now:
                if job.partition not in node.partitions:
                    continue

                if job_end > node.interval_times[1]:
                    continue

                if not n_nodes:
                    valid_nodes.add(node)
                    n_nodes += 1
                    max_node_weight = node, (node.weight, node.id)
                    continue
                # Have enough nodes now, so only accept more favourable nodes
                if n_nodes >= job.nodes:
                    node_weight = (node.weight, node.id)
                    if node_weight > max_node_weight[1]:
                        continue
                    valid_nodes.remove(max_node_weight[0])
                    max_node_weight = (node, node_weight)
                valid_nodes.add(node)
                n_nodes += 1

            if len(valid_nodes) == job.nodes:
                # NOTE New ready now nodes may be available after this but it's going to rare so
                # I won't bother, they will be picked up in the next sched anyway
                self.partitions.clear_planned_blocks(job)

                self._submit(job.start_job(self.time), nodes=valid_nodes)
                jobs_submitted.append(-(i_job_rev + 1))

            for node in valid_nodes:
                free_nodes_ready_now.remove(node)

            if not free_nodes_ready_now:
                break

            max_job_end = max(
                free_nodes_ready_now, key=lambda node: node.interval_times[1]
            ).interval_times[1]

        for i_job in reversed(jobs_submitted):
            self.queue.queue.pop(i_job)

    def _yield_planned_nodes_ready_now(self, job, job_end):
        if job.planned_block is None:
            return

        for node in job.planned_block[1]:
            # Node not ready now
            if node.interval_times[0] != self.time:
                continue
            # An earlier reservation is in the way
            if node.interval_times[1] != job.planned_block[0][0]:
                continue
            # Some cases where overrruning jobs can cause this to happen
            if node.interval_times[3] < job_end:
                continue

            yield node

    def _backfill(self):
        for node in self.partitions.nodes:
            if len(node.interval_times) > 2:
                node.interval_times = [node.interval_times[0], node.interval_times[-1]]
                while node.jobs_plnd:
                    node.jobs_plnd.pop().planned_block = None

        for reservation, res_queue in self.queue.reservations.items():
            if not res_queue:
                continue

            backfill_now = self._get_backfill_jobs(res_queue, reservation=reservation)

            for i_job, nodes in backfill_now:
                job_ready = self.queue.reservations[reservation][i_job]
                self._submit(job_ready.start_job(self.time), nodes=nodes)

            for i_job, _ in reversed(backfill_now):
               self.queue.reservations[reservation].pop(i_job)

        if self.queue.queue:
            backfill_now = self._get_backfill_jobs(self.queue.queue)

            for i_job, nodes in backfill_now:
                job_ready = self.queue.queue[i_job]
                self._submit(job_ready.start_job(self.time), nodes=nodes)

            # NOTE i_job here are negative so need to pop deepest indices first
            for i_job, _ in reversed(backfill_now):
               self.queue.queue.pop(i_job)

    def _get_backfill_jobs(self, queue, reservation=""):
        backfill_now = []
        window_end = self.time + self.config.bf_window

        # Stop backfilling when conditions met:
        # - No jobs with a reqtime that could fit on nodes that are avaible to run the job now
        # - All busy nodes that will go idle before the next bf_interval has at least one future
        #   reservation
        # This is done to speed up sim. There may be some small discrepancy since one future
        # reservations may not be enough for node to have correct behaviour in the sched cycles
        # between now and next backfill eg. job that crashes immediately or an smaller job that
        # could've fit in before the single reservation. Although I imagine this is going to be
        # quite rare so worth it for the speed up
        crit_time = self.time + self.config.bf_interval
        nodes_needing_resv = set()
        nodes_free_now = set()

        free_blocks = defaultdict(set)

        for interval, nodes in self.partitions.free_blocks[reservation].items():
            for node in nodes:
                free_blocks[interval].add(node)
                if interval[0] == self.time:
                    nodes_free_now.add(node)
                elif node.running_job.end <= crit_time:
                    nodes_needing_resv.add(node)

        if not nodes_needing_resv and not nodes_free_now:
            return backfill_now

        num_nodes_needing_resv = len(nodes_needing_resv)

        if nodes_free_now:
            max_reqtime_run_now = min(
                (
                    max(
                        nodes_free_now, key=lambda node: node.interval_times[1]
                    ).interval_times[1] -
                    self.time
                ),
                self.config.bf_window
            )
        else:
            max_reqtime_run_now = timedelta()

        if nodes_needing_resv:
            max_reqtime_nodes_needing_resv = min(
                (
                    max(
                        nodes_needing_resv, key=lambda node: node.interval_times[1]
                    ).interval_times[1] -
                    self.time
                ),
                self.config.bf_window
            )
        else:
            max_reqtime_nodes_needing_resv = timedelta()

        max_reqtime = max(max_reqtime_nodes_needing_resv, max_reqtime_run_now)

        for i_job_rev, job in enumerate(reversed(queue)):
            if self.num_bf_test_step >= self.config.bf_max_job_test:
                continue
            self.num_bf_test_step += 1

            reqtime = job.reqtime + self.config.bf_resolution

            if reqtime > max_reqtime:
                continue

            num_free_nodes = 0
            selected_intervals, unsorted_intervals = defaultdict(set), set()

            # NOTE Earliest starting first, then sort by earliest end
            sorted_free_blocks = sorted(free_blocks.items(), key=lambda block: block[0])
            usage_block_start = sorted_free_blocks[0][0][0]
            usage_block_end = usage_block_start + reqtime

            for i_block, (interval, nodes) in enumerate(sorted_free_blocks):
                if interval[1] >= usage_block_end and interval[1] >= interval[0] + reqtime:
                    valid_nodes = { node for node in nodes if job.partition in node.partitions }
                    if valid_nodes:
                        selected_intervals[interval] = valid_nodes
                        unsorted_intervals.add(interval)
                        num_free_nodes += len(valid_nodes)
                        usage_block_start = interval[0]
                        usage_block_end = interval[0] + reqtime
                        if usage_block_end > window_end:
                            break

                # Only move forward once we have gathered all valid nodes that
                # share an interval start time
                if not i_block + 1 == len(sorted_free_blocks):
                    if sorted_free_blocks[i_block + 1][0][0] != interval[0]:
                        if job.nodes > num_free_nodes:
                            unsorted_intervals.clear()
                            continue
                    else:
                        continue

                if job.nodes <= num_free_nodes:
                    # Remove blocks that do not remain free long enough for job to run
                    # NOTE Still need to check this since usage_block end has moved forward from
                    # first blocks
                    for selected_interval in list(selected_intervals.keys()):
                        if usage_block_end > selected_interval[1]:
                            num_free_nodes -= len(selected_intervals.pop(selected_interval))

                    if job.nodes > num_free_nodes:
                        unsorted_intervals.clear()
                        continue

                    # Need to sort this last bunch of nodes by weight to choose which to run on
                    possible_node_intervals = []
                    for unsorted_interval in unsorted_intervals:
                        unsorted_nodes = selected_intervals.pop(unsorted_interval)
                        num_free_nodes -= len(unsorted_nodes)
                        for unsorted_node in unsorted_nodes:
                            possible_node_intervals.append((unsorted_node, unsorted_interval))
                    possible_node_intervals.sort(
                        key=lambda node_interval: (node_interval[0].weight, node_interval[0].id),
                        reverse=True
                    )
                    for node_interval in possible_node_intervals[num_free_nodes - job.nodes:]:
                        selected_intervals[node_interval[1]].add(node_interval[0])

                    # Run the job
                    if usage_block_start == self.time:
                        backfill_now.append(
                            (
                                -(i_job_rev + 1),
                                { node for nodes in selected_intervals.values() for node in nodes }
                            )
                        )
                        for nodes in selected_intervals.values():
                            nodes_free_now -= nodes
                    # Reserve the requred blocks for this job
                    else:
                        job.planned_block = ((usage_block_start, usage_block_end), set())
                        for nodes in selected_intervals.values():
                            for node in nodes:
                                job.planned_block[1].add(node)
                                for i_time, time in enumerate(node.interval_times):
                                    if time > usage_block_start:
                                        node.interval_times.insert(i_time, usage_block_start)
                                        node.interval_times.insert(i_time + 1, usage_block_end)
                                        break
                                # node.interval_times.insert(-1, usage_block_start)
                                # node.interval_times.insert(-1, usage_block_end)
                                # node.interval_times[1:-1] = sorted(node.interval_times[1:-1])
                                node.jobs_plnd.add(job)

                    recompute_max_reqtime_run_now = False
                    recompute_max_reqtime_nodes_needing_resv = False

                    for selected_interval, nodes in selected_intervals.items():
                        free_blocks[selected_interval] -= nodes

                        if selected_interval[0] != usage_block_start:
                            free_blocks[(selected_interval[0], usage_block_start)].update(nodes)
                        if selected_interval[1] != usage_block_end:
                            free_blocks[(usage_block_end, selected_interval[1])].update(nodes)

                        if not free_blocks[selected_interval]:
                            free_blocks.pop(selected_interval)

                        if selected_interval[0] == self.time:
                            recompute_max_reqtime_run_now = True

                        if not nodes.isdisjoint(nodes_needing_resv):
                            recompute_max_reqtime_nodes_needing_resv = True

                    if recompute_max_reqtime_run_now:
                        if nodes_free_now:
                            max_reqtime_run_now = min(
                                (
                                    max(
                                        nodes_free_now, key=lambda node: node.interval_times[1]
                                    ).interval_times[1] -
                                    self.time
                                ),
                                self.config.bf_window
                            )
                        else:
                            max_reqtime_run_now = timedelta()
                        max_reqtime = max(max_reqtime_nodes_needing_resv, max_reqtime_run_now)

                    if recompute_max_reqtime_nodes_needing_resv:
                        max_reqtime_nodes_needing_resv = min(
                            (
                                max(
                                    nodes_needing_resv,
                                    key=lambda node: node.interval_times[1]
                                ).interval_times[1] -
                                self.time
                            ),
                            self.config.bf_window
                        )
                        max_reqtime = max(max_reqtime_nodes_needing_resv, max_reqtime_run_now)

                    break

        return backfill_now

    def _check_down_nodes(self):
        while self.down_nodes and self.down_nodes[-1].up_time <= self.time:
            node = self.down_nodes.pop()
            node.set_up()
            if node.reservation:
                node.interval_times[1] = node.unreserved_time
            elif node.reservation_schedule:
                node.interval_times[1] = node.reservation_schedule[-1][0]
            else:
                node.interval_times[1] = datetime.datetime.max
            node.interval_times[0] = self.time
            self.partitions.add_free_block(node)

        while self.node_down_order and self.node_down_order[-1].down_schedule[-1][0] <= self.time:
            node = self.node_down_order[-1]
            # If already down delay this new downtime until the next up to not interfere (this
            # happens because my DOWN implementation waits for current running job to finish)
            if node.down:
                node.down_schedule[-1][0] = node.up_time
                node.down_schedule.sort(key=lambda schedule: schedule[0], reverse=True)
                self.node_down_order.sort(key=lambda node: node.down_schedule[-1][0], reverse=True)
                continue

            # Delay down until after current running job has finished, this call will happen
            # after _check_fininshed_jobs() and before and sched calls so this down will not be
            # getting perpetually delayed
            if node.down_schedule[-1][2] == "DOWN" and node.running_job:
                node.down_schedule[-1][0] = node.running_job.end
                node.down_schedule.sort(key=lambda schedule: schedule[0], reverse=True)
                self.node_down_order.sort(key=lambda node: node.down_schedule[-1][0], reverse=True)
                continue

            down_schedule = node.down_schedule.pop()
            if not len(node.down_schedule):
                self.node_down_order.pop()
            else:
                self.node_down_order.sort(key=lambda node: node.down_schedule[-1][0], reverse=True)

            up_time = self.time + down_schedule[1]

            if node.running_job and up_time <= node.running_job.end:
                continue

            # Cancel all plans and remove all free_blocks
            self.partitions.clear_planned_blocks(node)

            self.partitions.remove_free_block(node)

            node.set_down(up_time)

            node.interval_times = [datetime.datetime.max, datetime.datetime.max]

            self.down_nodes.append(node)
            self.down_nodes.sort(key=lambda node: node.up_time, reverse=True)

    def _check_reservations(self):
        # Destroy and spawn new sliding reservations
        while self.sliding_reservations and self.sliding_reservations[-1][0] <= self.time:
            _, submit, clear, start, end, nodes, name = self.sliding_reservations[-1]
            for node in nodes:
                if submit is not None: # At a event where submitting new block
                    node.reservation_schedule.append((start, end, name))
                    node.reservation_schedule.sort(key=lambda schedule: schedule[0], reverse=True)

                    if node.down:
                        continue

                    self.partitions.remove_free_block(node)

                    # Remove any plans that this new reservation invalidates, for hpelongjobs
                    # something is broken if this every happens as the window slides forwards
                    while len(node.interval_times) > 2:
                        if node.interval_times[-2] > node.reservation_schedule[-1][0]:
                            node.interval_times.pop(-2)
                            node.interval_times.pop(-2)
                            raise Exception("Bruh?")
                        else:
                            break

                    node.interval_times[-1] = node.reservation_schedule[-1][0]
                    self.partitions.add_free_block(node)
                else: # At event where clearing block that is about to start
                    for i_res_sched, res_sched in enumerate(node.reservation_schedule):
                        if res_sched[2] != name:
                            continue
                        node.reservation_schedule.pop(i_res_sched)

                        if node.down:
                            break

                        self.partitions.remove_free_block(node)

                        if node.reservation_schedule:
                            node.interval_times[-1] = node.reservation_schedule[-1][0]
                        else:
                            node.interval_times[-1] = datetime.datetime.max

                        self.partitions.add_free_block(node)

                        break

            if not submit: # Finished this window
                self.sliding_reservations.pop()
            else: # Created this window, need to clear it next
                self.sliding_reservations[-1][0] = clear
                self.sliding_reservations[-1][1] = None
                # If clear is same time as next submit, want to clear first
                self.sliding_reservations.sort(
                    key=lambda sliding_res: (sliding_res[0], sliding_res[1] is not None),
                    reverse=True
                )

        # Set/unset reservation
        while self.reserved_nodes and self.reserved_nodes[-1].unreserved_time <= self.time:
            node = self.reserved_nodes.pop()

            if len(node.interval_times) > 2:
                raise Exception(
                    "Nodes leaving a reservation should not have anything planned"
                    "in the current implementation"
                )

            if node.down:
                node.set_unreserved()
                continue

            self.partitions.remove_free_block(node)

            node.set_unreserved()
            if node.reservation_schedule:
                node.interval_times[-1] = node.reservation_schedule[-1][0]
            else:
                node.interval_times[-1] = datetime.datetime.max

            self.partitions.add_free_block(node)

        while (
            self.node_reservation_order and
            self.node_reservation_order[-1].reservation_schedule[-1][0] <= self.time
        ):
            node = self.node_reservation_order[-1]

            reservation_schedule = node.reservation_schedule.pop()
            if not len(node.reservation_schedule):
                self.node_reservation_order.pop()
            else:
                self.node_reservation_order.sort(
                    key=lambda node: node.reservation_schedule[-1][0], reverse=True
                )

            if len(node.interval_times) > 2:
                raise Exception(
                    "Nodes entering a reservation should not have anything planned"
                    "in the current implementation"
                )

            if node.down:
                node.set_reserved(reservation_schedule[2], reservation_schedule[1])

                self.reserved_nodes.append(node)
                self.reserved_nodes.sort(key=lambda node: node.unreserved_time, reverse=True)

                break

            self.partitions.remove_free_block(node)

            node.set_reserved(reservation_schedule[2], reservation_schedule[1])
            node.interval_times[-1] = node.unreserved_time
            self.partitions.add_free_block(node)

            self.reserved_nodes.append(node)
            self.reserved_nodes.sort(key=lambda node: node.unreserved_time, reverse=True)

    def _print_stats(self, sched, bf):
        if not (self.time.hour != self.previous_print_hour and not self.time.hour % 3):
            return

        self.previous_print_hour = self.time.hour
        print(
            "{} (step {}):\n".format(self.time, self.step_cnt) +
            # "Idle Nodes = {} (num in free_blocks_ready {}) (highmem {})\t" \
            "Idle Nodes = {} (".format(sum(1 for node in self.partitions.nodes if node.free)) +
            " ".join(
                "{}={}".format(
                    partition.name, sum(1 for node in partition.nodes if node.free)
                ) for partition in self.partitions.partitions
            ) +
            ")\t" + 
            "NodesReserved = {} (Idle = {})\tNodesHPE_RestrictLongJobs = {} (Idle = {})\t" \
            "NodesDown = {}\tPower = {:.4f} MW\n".format(
                sum(1 for node in self.partitions.nodes if node.reservation),
                sum(
                   1 for node in self.partitions.nodes if (
                        node.reservation and not node.running_job and not node.down
                    )
                ),
                sum(
                    1 for node in self.partitions.nodes if (
                        node.reservation_schedule and
                        "HPE_RestrictLongJobs" in [
                            res_sched[2] for res_sched in node.reservation_schedule
                        ]
                    )
                ),
                sum(
                    1 for node in self.partitions.nodes if (
                        node.reservation_schedule and
                        "HPE_RestrictLongJobs" in [
                            res_sched[2] for res_sched in node.reservation_schedule
                        ] and
                        not node.running_job and not node.down
                    )
                ),
                sum(1 for node in self.down_nodes if not node.running_job),
                self.power_usage
            ) +
            "QueueSize = {} (held by priority {} (partition ".format(
                (
                    len(self.queue.queue) +
                    len(self.queue.waiting_dependency) +
                    sum(len(jobs) for jobs in self.queue.qos_held.values())
                ),
                len(self.queue.queue)
            ) +
            " ".join(
                "{}={}".format(
                    partition.name,
                    sum(1 for job in self.queue.queue if job.partition is partition)
                ) for partition in self.partitions.partitions
            ) +
            " qos lowpriority {}) ".format(
                sum(1 for job in self.queue.queue if job.qos.name == "lowpriority")
            ) +
            "dependency {} qos holds {} (".format(
                len(self.queue.waiting_dependency),
                sum(len(jobs) for jobs in self.queue.qos_held.values())
            ) +
            " ".join(
                "{}={}".format(
                    qos.name, len(jobs)
                ) for qos, jobs in self.queue.qos_held.items() if len(jobs)
            ) +
            ") qos submit holds {} (".format(
                sum(len(jobs) for jobs in self.queue.qos_submit_held.values())
            ) +
            " ".join(
                "{}={}".format(
                    qos.name, len(jobs)
                ) for qos, jobs in self.queue.qos_submit_held.items() if len(jobs)
            ) +
            " " +
            " ".join(
                "{}={}".format(
                    (assoc[0], assoc[1].name, assoc[2]), 
                    sum(
                        1
                        for jobs in self.queue.qos_submit_held.values()
                            for job in jobs
                                if job.assoc == assoc
                    )
                )
                for assoc in set(
                    job.assoc
                    for jobs in self.queue.qos_submit_held.values()
                        for job in jobs
                )
            ) +
            "))\tRunningJobs = {}\n".format(len(self.running_jobs))
        )

        # if self.queue.queue:
            # print("ID - QOS - Age - Size - FairShare - Total")
            # print("User - FairShare")
            # for job in self.queue.queue:
                # qos_factor = int(self.queue.priority_sorter._qos_priority(job))
                # age_factor = int(self.queue.priority_sorter._age_priority(job))
                # size_factor = int(self.queue.priority_sorter._size_priority(job))
                # fairshare_factor = int(self.queue.priority_sorter._fairshare_priority(job))
                # print(
                #     job.id, qos_factor, age_factor, size_factor, fairshare_factor,
                #     qos_factor + age_factor + size_factor + fairshare_factor, sep=" - "
                # )
                # print(job.user, self.queue.priority_sorter._fairshare_priority(job))

        # max_endtime_ready_num_nodes = defaultdict(int)
        # for interval, nodes in self.partitions.free_blocks[""].items():
        #     if interval[0] > self.time:
        #         continue
        #     max_endtime_ready_num_nodes[interval[1]] += len(nodes)

        # for max_endtime, num_nodes in sorted(
        #     max_endtime_ready_num_nodes.items(), key=lambda pair: pair[0]
        # ):
        #     print(
        #         "{}: {}".format(
        #             (
        #                 max_endtime - self.time
        #                 if max_endtime != datetime.datetime.max
        #                 else datetime.datetime.max
        #             ),
        #             num_nodes
        #         )
        #     )
        # if any(job.partition.name == "standard" for job in self.queue.queue):
        #     next_job = self.queue.queue[-1]
        #     print(
        #         next_job.reqtime, next_job.nodes, next_job.partition.name, next_job.qos.name,
        #         next_job.true_submit, next_job.true_job_start,
        #         (
        #             (next_job.planned_block[0],  len(next_job.planned_block[1]))
        #             if next_job.planned_block else
        #             next_job.planned_block
        #         )
        #     )
        #     print(
        #         min(
        #             (
        #                 (
        #                     job.reqtime, job.nodes,
        #                     (
        #                         (job.planned_block[0],  len(job.planned_block[1]))
        #                         if job.planned_block is not None else
        #                         job.planned_block
        #                     )
        #                 )
        #                 for job in self.queue.queue
        #                     if job.partition.name == "standard"
        #             ),
        #             key=lambda job_data: (job_data[0], -job_data[1])
        #         )
        #     )
        # print("sched", sched, "bf", bf, "sched_steps", self.num_sched_test_step)
        # print()


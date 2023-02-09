import os, time, copy
import datetime; from datetime import timedelta
from collections import defaultdict, Counter
import dill as pickle

import pandas as pd

from config import get_config
from partition import Partitions
from job_queue import Queue
from priority_sorters import MFPrioritySorter


class Controller:
    def __init__(self, config_file):
        self.config = get_config(config_file)

        self.partitions = Partitions(self.config.node_events_dump, self.config.reservations_dump)

        self.queue = Queue(self.config.job_dump, self.partitions)
        self.init_time = self.queue.time
        self.time = self.queue.time
        priority_sorter = MFPrioritySorter(
            self.config.assocs_dump, timedelta(minutes=5), timedelta(days=2), self.init_time, 100,
            500, 300, timedelta(days=14), 0, 10000,
            len({ partition.priority_tier for partition in self.partitions.partitions }) == 1
        )
        self.queue.set_priority_sorter(priority_sorter)
        self.fairtree = priority_sorter.fairtree

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
        if self.config.hpe_restrictlongjobs_sliding_reservations:
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

        self.step_cnt = 0
        self.previous_print_hour = self.time.hour

    def _next_job_finish(self):
        if not self.running_jobs:
            return datetime.datetime.max

        return self.running_jobs[-1].end

    def run_sim(self):
        import numpy as np
        sim_start = time.time()

        times_sched, times_bf, times_sched_bf = [], [], []

        # TODO Non-defer implementation
        previous_small_sched = self.time
        next_bf_time = self.time + self.config.bf_interval
        next_sched_time = self.time + self.config.sched_interval
        next_fairtree_time = self.time + self.config.PriorityCalcPeriod
        while self.queue.all_jobs or self.queue.queue or self.running_jobs:
            start = time.time()
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
            if bf and not sched:
                times_bf.append(time.time() - start)
            if sched and not bf:
                times_sched.append(time.time() - start)
            if sched and bf:
                times_sched_bf.append(time.time() - start)
            if self.step_cnt % 1000 == 0:
                print("bf: ", np.mean(times_bf))
                print("sched: ", np.mean(times_sched))
                print("sched & bf: ", np.mean(times_sched_bf))

        elapsed = time.time() - sim_start
        print(
            "Sim complete in {} hr {} mins".format(
                elapsed // (60 * 60), (elapsed % (60 * 60)) // 60
            )
        )

    def _step(self, sched, sched_depth, bf, fairtree):
        # NOTE Regardless of sched or bf still need to:
        # - check for down nodes so the they can go down closer to the correct time rather than in
        # groups at sched/bf steps
        # - finish jobs when they finish so that dependecies/qos submit holds can start accruing
        # time at the correct time
        # - release jobs from qos submit and dependencies holds so that they can start accruing
        # time at the correct time

        self._check_finished_jobs()

        self._check_down_nodes()
        self._check_reservations()

        # NOTE Changes to dependencies and qos implementations should mean I won't have to pass all
        # of this
        self.queue.step(self.time, self.running_jobs)

        if fairtree:
            self.fairtree.fairshare_calc(self.running_jobs, self.time)

        pre_submit_running_jobs_len = len(self.running_jobs)

        # Process free_blocks ready to use by schedulers:
        # - check all ready now intervals for nodes that are still running, move these to a new
        # interval that extents the endlimit
        # - populate free_blocks_ready intervals
        if sched or bf:
            self.partitions.free_blocks_ready_intervals = defaultdict(list)
            for res, free_blocks in self.partitions.free_blocks.items():
                for interval in list(free_blocks):
                    if interval[0] > self.time: # TODO use longest over-run of reqtime for these checks
                        continue

                    for node in list(free_blocks[interval]):
                        if not node.running_job:
                            continue
                        node.running_job.endlimit = self.time + self.config.bf_resolution
                        free_blocks[interval].remove(node)
                        node.free_block_interval = (node.running_job.endlimit, interval[1])
                        free_blocks[node.free_block_interval].add(node)

                    if not free_blocks[interval]:
                        free_blocks.pop(interval)
                        continue

                    # Merge all free blocks free blocks all starting at the current time
                    # if interval[0] != self.time:
                    #     free_blocks[(self.time, interval[1])].update(free_blocks[interval])
                    #     free_blocks.pop(interval)

                    self.partitions.free_blocks_ready_intervals[res].append(interval)

        if sched:
            self.num_sched_test_step = 0
            sched_depth = sched_depth if sched_depth is not None else (
                sum(len(res_queue) for res_queue in self.queue.reservations.values()) +
                len(self.queue.queue)
            )

            self._sched_reservations(sched_depth)

            self._sched_main(sched_depth)

        if bf:
            self.num_bf_test_step = 0
            self._backfill()

        if len(self.running_jobs) != pre_submit_running_jobs_len:
            self.running_jobs.sort(key=lambda job: job.end, reverse=True)

        self.times.append(self.time)

        self._print_stats()

    def _submit(self, job, nodes=None):
        self.running_jobs.append(job)
        self.power_usage += job.true_node_power * job.nodes / 1e+6
        self.total_energy += (
            job.true_node_power * job.nodes * job.runtime.total_seconds() / 1e+9
        )

        if nodes:
            for node in nodes:
                self.partitions.free_blocks[node.reservation][node.free_block_interval].remove(
                    node
                )
                if not self.partitions.free_blocks[node.reservation][node.free_block_interval]:
                    self.partitions.free_blocks[node.reservation].pop(node.free_block_interval)
                node.free_block_interval = (job.endlimit, node.free_block_interval[1])
                self.partitions.free_blocks[node.reservation][node.free_block_interval].add(node)
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
            self.partitions.free_blocks[node.reservation][node.free_block_interval].remove(
                node
            )
            if not self.partitions.free_blocks[node.reservation][node.free_block_interval]:
                self.partitions.free_blocks[node.reservation].pop(node.free_block_interval)
            node.free_block_interval = (self.time, node.free_block_interval[1])
            self.partitions.free_blocks[node.reservation][node.free_block_interval].add(node)

    def _sched_reservations(self, sched_depth):
        for reservation, res_queue in self.queue.reservations.items():
            if not res_queue:
                continue

            # Reservation ended, delete stray jobs. Could be a problem if the same reservation
            # comes back but *shrug*
            if not self.partitions.reservations[reservation]:
                print(
                    "!!!\nReservation {} has no nodes, deleting {} job\n!!!".format(
                        reservation, len(self.queue.reservation[reservation])
                    )
                )
                self.queue.reservations[reservation] = []
                continue

            # NOTE Reservations must be a single partition so don't need to worry about order
            free_nodes_res = [
                node for node in self.partitions.reservations[reservation] if (
                    not node.running_job and not node.down
                )
            ]
            jobs_submitted = []
            for i_job_rev, job in enumerate(reversed(res_queue)):
                valid_nodes, i_free_node = [], len(free_nodes_res)
                while len(valid_nodes) < job.nodes and i_free_node:
                    i_free_node -= 1
                    node = free_nodes_res[i_free_node]
                    if self.time + job.reqtime > node.unreserved_time:
                        continue
                    valid_nodes.append(free_nodes_res.pop(i_free_node))

                if len(valid_nodes) == job.nodes:
                    self._submit(job.start_job(self.time), nodes=valid_nodes)
                    jobs_submitted.append(-(i_job_rev + 1))
                else:
                    break

            for i_job in reversed(jobs_submitted):
               self.queue.reservations[reservation].pop(i_job)

    def _sched_main(self, sched_depth):
        partition_free_nodes = self.partitions.get_partition_free_nodes()

        # NOTE All of this weird counting is to prevent iterating over and verifying that jobs
        # cannot run on nodes we can know before hand that they can't. This makes the sim much
        # faster when the HPE_RestrictLongJobs reservation is included
        partition_always_free_num_nodes = defaultdict(int)
        partition_free_time_num_nodes = defaultdict(lambda: defaultdict(int))
        for partition, free_nodes in partition_free_nodes.items():
            for node in free_nodes:
                if node.reservation_schedule:
                    partition_free_time_num_nodes[partition][node.reservation_schedule[-1][0]] += 1
                else:
                    partition_always_free_num_nodes[partition] += 1

        partitions_with_space = {
            partition for partition, free_nodes in partition_free_nodes.items() if free_nodes
        }
        partitions_full = set()
        max_reqtime = (
            max(
                self.partitions.free_blocks_ready_intervals[""], key=lambda interval: interval[1]
            )[1] -
            self.time
        )
        jobs_submitted = []
        # NOTE This implementation is a little weird since I need to maintain node ordering
        # according to its weight in a partition
        for i_job_rev, job in enumerate(reversed(self.queue.queue)):
            if self.num_sched_test_step >= sched_depth:
                break
            self.num_sched_test_step += 1

            if job.partition in partitions_full:
                continue

            if job.partition not in partitions_with_space:
                continue

            if (
                not partition_always_free_num_nodes[job.partition] and
                self.time + job.reqtime >= max(partition_free_time_num_nodes[job.partition].keys())
            ):
                continue

            valid_nodes, i_free_node = [], len(partition_free_nodes[job.partition])
            while len(valid_nodes) < job.nodes and i_free_node:
                i_free_node -= 1
                node = partition_free_nodes[job.partition][i_free_node]
                if (
                    node.reservation_schedule and
                    self.time + job.reqtime > node.reservation_schedule[-1][0]
                ):
                    continue
                valid_nodes.append(partition_free_nodes[job.partition].pop(i_free_node))

                # Node is going to be allocated/planned, update counters and shared partitions
                for partition in node.partitions:
                    res_sched = node.reservation_schedule
                    if res_sched:
                        partition_free_time_num_nodes[partition][res_sched[-1][0]] -= 1
                        if not partition_free_time_num_nodes[partition][res_sched[-1][0]]:
                            partition_free_time_num_nodes[partition].pop(res_sched[-1][0])
                    else:
                        partition_always_free_num_nodes[partition] -= 1

                    if partition != job.partition:
                        partition_free_nodes[partition].remove(node)

                    if not partition_free_nodes[partition]:
                        partitions_with_space.remove(partition)

            if len(valid_nodes) == job.nodes:
                self._submit(job.start_job(self.time), nodes=valid_nodes)
                jobs_submitted.append(-(i_job_rev + 1))

            if not partitions_with_space:
                break

        for i_job in reversed(jobs_submitted):
            self.queue.queue.pop(i_job)

    def _backfill(self):
        for reservation, res_queue in self.queue.reservations.items():
            if not res_queue:
                continue

            backfill_now = self._get_backfill_jobs(
                {
                    node for node in self.partitions.reservations[reservation] if (
                        not node.running_job and not node.down
                    )
                },
                queue=res_queue,
                running_jobs=(
                    {
                        node.running_job for node in self.partitions.reservations[reservation] if node.reservation == reservation
                    } -
                    { None }
                ),
                reservation=True
            )
            for i_job, nodes in backfill_now:
                job_ready = res_queue[i_job]
                self._submit(job_ready.start_job(self.time), nodes=nodes)

            # if backfill_now:
            #     print(reservation, [ (i_job, len(nodes)) for i_job, nodes in backfill_now ])

            for i_job, _ in reversed(backfill_now):
               self.queue.reservations[reservation].pop(i_job)

        if self.queue.queue:
            backfill_now = self._get_backfill_jobs(self.partitions.get_free_nodes())
            for i_job, nodes in backfill_now:
                job_ready = self.queue.queue[i_job]
                self._submit(job_ready.start_job(self.time), nodes=nodes)

            # if backfill_now:
            #     print([ (i_job, len(nodes)) for i_job, nodes in backfill_now ])

            # NOTE i_job here are negative so need to pop deepest indices first
            for i_job, _ in reversed(backfill_now):
               self.queue.queue.pop(i_job)

    # def _get_backfill_jobs(self, partition_free_nodes):
    # NOTE if reservation is specified, expects free_nodes and running_jobs to only relate to nodes
    # in the reservation
    def _get_backfill_jobs(self, free_nodes, queue=None, running_jobs=None, reservation=False):
    # def _get_backfill_jobs(self, queue, reservation=""):
        backfill_now = []

#         running_jobs = running_jobs if running_jobs is not None else self.running_jobs
#         queue = queue if queue else self.queue.queue

#         # NOTE start of EASY bf implementation in git history

#         free_blocks = defaultdict(set)
#         free_blocks_ready_intervals = set()
#         for node in free_nodes:
#             # Backfilling for reservation queue
#             if reservation:
#                 free_blocks[(self.time, node.unreserved_time)].add(node)
#                 free_blocks_ready_intervals.add((self.time, node.unreserved_time))
#                 continue

#             # Free block will end at an upcoming reservation
#             if node.reservation_schedule:
#                 free_blocks[(self.time, node.reservation_schedule[-1][0])].add(node)
#                 free_blocks_ready_intervals.add((self.time, node.reservation_schedule[-1][0]))
#                 continue

#             free_blocks[(self.time, datetime.datetime.max)].add(node)
#             free_blocks_ready_intervals.add((self.time, datetime.datetime.max))

#         if not free_blocks_ready_intervals:
#             return backfill_now

        window_end = self.time + self.config.bf_window
#         # NOTE When running_jobs is specified assume it includes only jobs from the given
#         # reservation
#         for job in running_jobs:
#             # Some jobs exceeding timelimit and some with zero reqtime
#             if job.endlimit <= self.time:
#                 job.endlimit = self.time + self.config.bf_resolution

#             for node in job.assigned_nodes:
#                 if reservation:
#                     free_blocks[(job.endlimit, node.unreserved_time)].add(node)
#                     continue

#                 # NOTE Should check if there is a scheduled reservation coming after a current
#                 # reservation has finished rather than setting to datetime.max
#                 if node.reservation:
#                     if node.unreserved_time > window_end:
#                         break
#                     free_blocks[(node.unreserved_time, datetime.datetime.max)].add(node)
#                     continue

#                 if not node.reservation_schedule:
#                     free_blocks[(job.endlimit, datetime.datetime.max)].add(node)
#                     continue
#                 end_restriction = node.reservation_schedule[-1][0]
#                 if job.endlimit >= end_restriction:
#                     continue
#                 free_blocks[(job.endlimit, end_restriction)].add(node)

        # if running_jobs:
        #     alt_free_blocks = defaultdict(set)
        #     res = "" if not reservation else job.reservation
        #     for interval, nodes in self.partitions.free_blocks[res].items():
        #         if interval[0] > window_end:
        #             continue
        #         if interval[0] > self.time:
        #             for node in nodes:
        #                 if node.running_job not in running_jobs:
        #                     continue
        #                 alt_free_blocks[interval].add(node)
        #             continue
        #         if interval[0] <= self.time:
        #             for node in nodes:
        #                 if node.running_job:
        #                     alt_free_blocks[(self.time + self.config.bf_resolution, interval[1])].add(node)
        #                     continue
        #                 alt_free_blocks[(self.time, interval[1])].add(node)
        #     free_blocks_no_down = { interval : { node for node in nodes if not node.down } for interval, nodes in free_blocks.items() if not all(node.down for node in nodes) }
        #     if alt_free_blocks != free_blocks_no_down:
        #         print(sum(len(nodes) for nodes in alt_free_blocks.values()), sum(len(nodes) for nodes in free_blocks_no_down.values()))
        #         if { node for nodes in alt_free_blocks.values() for node in nodes } != {  node for nodes in free_blocks_no_down.values() for node in nodes }:
        #             print("ughhhhhhhhhhhhhhhhhhhhhhhhh")
        #         print(len(alt_free_blocks), len(free_blocks_no_down))
        #         print("ugh")
        #         print(len(free_nodes), len(running_jobs), reservation)
        #         print(self.time)
        #         print({ interval for interval in alt_free_blocks } - { interval for interval in free_blocks_no_down })
        #         print({ interval for interval in free_blocks_no_down } - { interval for interval in alt_free_blocks })

        if not reservation:
            reservation = ""
        if not queue:
            queue = self.queue.queue

        free_blocks = defaultdict(set, { (max(interval[0], self.time), interval[1]) : set(nodes) for interval, nodes in self.partitions.free_blocks[reservation].items() if interval[0] < window_end })
        # free_blocks = copy.deepcopy(self.partitions.free_blocks[reservation])
        # free_blocks = self.partitions.free_blocks[reservation]
        # free_blocks_ready_intervals = copy.copy(self.partitions.free_blocks_ready_intervals[reservation])
        free_blocks_ready_intervals = { interval for interval in free_blocks.keys() if interval[0] <= self.time }

        if not free_blocks_ready_intervals:
            return backfill_now

        min_required_block_time = (
            self.time + min(queue, key=lambda job: job.reqtime).reqtime + self.config.bf_resolution
        )
        max_block_time = max(free_blocks_ready_intervals, key=lambda interval: interval[1])[1]

        for i_job_rev, job in enumerate(reversed(queue)):
            # break if no blocks or only <= min blocks available for immediate backfill
            if max_block_time < min_required_block_time:
                break

            if self.num_bf_test_step >= self.config.bf_max_job_test:
                break
            self.num_bf_test_step += 1

            reqtime = job.reqtime + self.config.bf_resolution
            # Only need to plan nodes for jobs that may be relevant to immediate scheduling
            if self.time + reqtime > max_block_time:
                continue

            num_free_nodes = 0
            selected_intervals = {}

            # NOTE Earliest starting first, then sort by earliest end for reproducibility
            for interval, nodes in sorted(free_blocks.items(), key=lambda block: block[0]):
                valid_nodes = { node for node in nodes if job.partition in node.partitions }
                if valid_nodes:
                    selected_intervals[interval] = valid_nodes
                    num_free_nodes += len(valid_nodes)

                if job.nodes <= num_free_nodes:
                    usage_block_start = max(
                        selected_intervals.keys(),
                        key=lambda selected_interval: selected_interval[0]
                    )[0]
                    usage_block_end = usage_block_start + reqtime

                    # Remove blocks that do not remain free long enough for job to run
                    for selected_interval in list(selected_intervals.keys()):
                        if usage_block_end > selected_interval[1]:
                            num_free_nodes -= len(selected_intervals.pop(selected_interval))

                    if job.nodes > num_free_nodes:
                        continue

                    # Remove nodes we don't need from the latest interval added, sort first for
                    # reproducibility
                    if num_free_nodes != job.nodes:
                        selected_intervals[interval] = set(
                            sorted(
                                selected_intervals[interval],
                                key=lambda node: (node.weight, node.id)
                            )[num_free_nodes - job.nodes:]
                        )

                    if usage_block_start == self.time:
                        backfill_now.append(
                            (
                                -(i_job_rev + 1),
                                { node for nodes in selected_intervals.values() for node in nodes }
                            )
                        )

                    for selected_interval, nodes in selected_intervals.items():
                        free_blocks[selected_interval] -= nodes

                        # The original ready now block has been broken and the interval needs
                        # redefining
                        if selected_interval[0] <= self.time:
                            if not free_blocks[selected_interval]:
                                free_blocks_ready_intervals.remove(selected_interval)
                            if selected_interval[0] != usage_block_start:
                                free_blocks_ready_intervals.add(
                                    (selected_interval[0], usage_block_start)
                                )

                        # Dont consider free blocks starting beyond bf_max_window
                        if selected_interval[0] < window_end:
                            if selected_interval[0] != usage_block_start:
                                free_blocks[(selected_interval[0], usage_block_start)].update(
                                    nodes
                                )
                            if (
                                selected_interval[1] != usage_block_end and
                                usage_block_end < window_end
                            ):
                                free_blocks[(usage_block_end, selected_interval[1])].update(nodes)

                        # These nodes are now planned, delete block if this leaves nothing left
                        if not free_blocks[selected_interval]:
                            free_blocks.pop(selected_interval)

                    if not free_blocks_ready_intervals:
                        return backfill_now
                    max_block_time = max(
                        free_blocks_ready_intervals, key=lambda interval: interval[1]
                    )[1]

                    break

        return backfill_now

    def _check_down_nodes(self):
        while self.down_nodes and self.down_nodes[-1].up_time <= self.time:
            node = self.down_nodes.pop()
            node.set_up()
            self.partitions.free_blocks[node.reservation][node.free_block_interval].remove(node)
            if not self.partitions.free_blocks[node.reservation][node.free_block_interval]:
                self.partitions.free_blocks[node.reservation].pop(node.free_block_interval)
            if node.reservation:
                node.free_block_interval = (self.time, node.unreserved_time)
            elif node.reservation_schedule:
                node.free_block_interval = (self.time, node.reservation_schedule[-1][0])
            else:
                node.free_block_interval = (self.time, datetime.datetime.max)
            self.partitions.free_blocks[node.reservation][node.free_block_interval].add(node)

        while self.node_down_order and self.node_down_order[-1].down_schedule[-1][0] <= self.time:
            node = self.node_down_order[-1]
            # If already down delay this new downtime until the next up to not interfere (this
            # happens because my DOWN implementation waits for current running job to finish)
            if node.down:
                node.down_schedule[-1][0] = node.up_time
                self.node_down_order.sort(key=lambda node: node.down_schedule[-1][0], reverse=True)
                continue

            down_schedule = node.down_schedule.pop()
            if not len(node.down_schedule):
                self.node_down_order.pop()
            else:
                self.node_down_order.sort(key=lambda node: node.down_schedule[-1][0], reverse=True)

            up_time = self.time + down_schedule[1]

            if node.running_job:
                if down_schedule[2] == "DOWN":
                    up_time += node.running_job.end - self.time
                if up_time <= node.running_job.end:
                    continue

            node.set_down(up_time)
            self.partitions.free_blocks[node.reservation][node.free_block_interval].remove(node)
            if not self.partitions.free_blocks[node.reservation][node.free_block_interval]:
                self.partitions.free_blocks[node.reservation].pop(node.free_block_interval)
            node.free_block_interval = (datetime.datetime.max, datetime.datetime.max)
            self.partitions.free_blocks[node.reservation][node.free_block_interval].add(node)

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
                    self.partitions.free_blocks[node.reservation][node.free_block_interval].remove(
                        node
                    )
                    if not self.partitions.free_blocks[node.reservation][node.free_block_interval]:
                        self.partitions.free_blocks[node.reservation].pop(node.free_block_interval)
                    node.free_block_interval = (
                        node.free_block_interval[0], node.reservation_schedule[-1][0]
                    )
                    self.partitions.free_blocks[node.reservation][node.free_block_interval].add(
                        node
                    )
                else: # At event where clearing block that is about to start
                    for i_res_sched, res_sched in enumerate(node.reservation_schedule):
                        if res_sched[2] != name:
                            continue
                        node.reservation_schedule.pop(i_res_sched)
                        (
                            self.partitions.free_blocks
                        )[node.reservation][node.free_block_interval].remove(node)
                        if node.reservation_schedule:
                            node.free_block_interval = (
                                node.free_block_interval[0], node.reservation_schedule[-1][0]
                            )
                        else:
                            node.free_block_interval = (
                                node.free_block_interval[0], datetime.datetime.max
                            )
                        (
                            self.partitions.free_blocks
                        )[node.reservation][node.free_block_interval].add(node)

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
            self.partitions.remove_node_reservation(node)
            self.partitions.free_blocks[node.reservation][node.free_block_interval].remove(node)
            if not self.partitions.free_blocks[node.reservation][node.free_block_interval]:
                self.partitions.free_blocks[node.reservation].pop(node.free_block_interval)
            node.set_unreserved()
            if node.reservation_schedule:
                node.free_block_interval = (
                    max(self.time, node.free_block_interval[0]), node.reservation_schedule[-1][0]
                )
            else:
                node.free_block_interval = (
                    max(self.time, node.free_block_interval[0]), datetime.datetime.max
                )
            self.partitions.free_blocks[node.reservation][node.free_block_interval].add(node)

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

            self.partitions.free_blocks[node.reservation][node.free_block_interval].remove(node)
            if not self.partitions.free_blocks[node.reservation][node.free_block_interval]:
                self.partitions.free_blocks[node.reservation].pop(node.free_block_interval)
            node.set_reserved(reservation_schedule[2], reservation_schedule[1])
            node.free_block_interval = (
                max(self.time, node.free_block_interval[0]), node.unreserved_time
            )
            self.partitions.free_blocks[node.reservation][node.free_block_interval].add(node)
            self.partitions.add_node_reservation(node)

            self.reserved_nodes.append(node)
            self.reserved_nodes.sort(key=lambda node: node.unreserved_time, reverse=True)

    def _print_stats(self):
        if not (self.time.hour != self.previous_print_hour and not self.time.hour % 3):
            return

        self.previous_print_hour = self.time.hour
        print(
            "{} (step {}):\n".format(self.time, self.step_cnt) +
            "Idle Nodes = {} (highmem {})\tNodesReserved = {} (Idle = {})\t" \
            "NodesHPE_RestrictLongJobs = {} (Idle = {})\t" \
            "NodesDown = {}\tPower = {:.4f} MW\n".format(
                sum(
                    1 for node in self.partitions.nodes if (
                        node.free and "standard" in [
                            partition.name for partition in node.partitions
                        ]
                    )
                ),
                sum(
                    1 for node in self.partitions.nodes if (
                        node.free and "highmem" in [
                            partition.name for partition in node.partitions
                        ]
                    )
                ),
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
            "QueueSize = {} (held by priority {} (partition highmem {} qos lowpriority {}) " \
            "dependency {} qos holds {} (".format(
                (
                    len(self.queue.queue) +
                    len(self.queue.waiting_dependency) +
                    sum(len(jobs) for jobs in self.queue.qos_held.values())
                ),
                len(self.queue.queue),
                sum(1 for job in self.queue.queue if job.partition == "highmem"),
                sum(1 for job in self.queue.queue if job.qos.name == "lowpriority"),
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
            "))\tRunningJobs = {}\n".format(len(self.running_jobs))
        )


import os
import datetime; from datetime import timedelta
from collections import defaultdict
import dill as pickle

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
            500, 300, timedelta(days=14), 0, 10000
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
            key=lambda node: node.down_schedule[0][0]
        )
        self.reserved_nodes = []
        self.node_reservation_order = sorted(
            [ node for node in self.partitions.nodes if node.reservation_schedule ],
            key=lambda node: node.reservation_schedule[0][0]
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
                    "HPE_RestricLongJobs"
                ] for submitted in sorted(hpe_restrictlong_res)
            ]

        else:
            self.sliding_reservations = []

        self.step_cnt = 0
        self.previous_print_hour = self.time.hour

    def _next_job_finish(self):
        if not self.running_jobs:
            return datetime.datetime.max

        return self.running_jobs[0].end

    def run_sim(self):
        # TODO Non-defer implementation
        previous_small_sched = self.time
        next_bf_time = self.time + self.config.bf_interval
        next_sched_time = self.time + self.config.sched_interval
        next_fairtree_time = self.time + self.config.PriorityCalcPeriod
        while self.queue.all_jobs or self.queue.queue or self.running_jobs:
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

        # NOTE Put these into methods
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
            self.running_jobs.sort(key=lambda job: job.end)

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
                job.assign_node(node)

        else:
            for node in job.partition.nodes:
                if node.free:
                    if job.assign_node(node):
                        break

        if len(job.assigned_nodes) != job.nodes:
            raise Exception("bruh")

        return True

    def _check_finished_jobs(self):
        while self.running_jobs and self.running_jobs[0].end <= self.time:
            job = self.running_jobs.pop(0)
            job.end_job()
            self.fairtree.job_finish_usage_update(job)
            self.job_history.append(job)
            self.power_usage -= job.true_node_power * job.nodes / 1e+6
            self.total_energy += (
                job.true_node_power * job.nodes * job.runtime.total_seconds() / 1e+9
            )

    def _sched_reservations(self, sched_depth):
        for reservation, res_queue in self.queue.reservations.items():
            if not res_queue:
                continue

            # Reservation ended, delete stray jobs. Could be a problem if the same reservation
            # comes back but *shrug*
            if not self.partitions.reservations[reservation]:
                self.queue.reservations[reservation] = []
                continue

            # TODO Refactor of scheduling in reserved blocks, want to be able to backfill if
            # needed, also might be work rethinking what node "free" means in context of
            # reservations
            # NOTE Reservations must be a single partition
            free_nodes_res = [
                node for node in self.partitions.reservations[reservation] if (
                    not node.running_job and not node.down
                )
            ]
            jobs_submitted = []
            for i_job, job in enumerate(res_queue):
                valid_nodes, i_free_node = [], 0
                while len(valid_nodes) < job.nodes and i_free_node < len(free_nodes_res):
                    node = free_nodes_res[i_free_node]
                    if self.time + job.reqtime > node.unreserved_time:
                        i_free_node += 1
                        continue
                    valid_nodes.append(free_nodes_res.pop(i_free_node))

                if len(valid_nodes) == job.nodes:
                    self._submit(job.start_job(self.time), nodes=valid_nodes)
                    jobs_submitted.append(i_job)
                else:
                    break

            for i_job in sorted(jobs_submitted, reverse=True):
               self.queue.reservations[reservation].pop(i_job)

                # while job.nodes < len(free_nodes_res):
                #     if self.num_sched_test_step >= sched_depth:
                #         break
                #     self.num_sched_test_step += 1

                #     job = res_queue.pop(0).start_job(self.time)
                #     self._submit(job, nodes=[ free_nodes_res.pop(0) for _ in range(job.nodes) ])

    def _sched_main(self, sched_depth):
        partition_free_nodes = self.partitions.get_partition_free_nodes()
        partitions_full = set()
        jobs_submitted = []
        # NOTE This implementation is a little weird since I need to maintain node ordering
        # according to its weight in a partition
        for i_job, job in enumerate(self.queue.queue):
            if self.num_sched_test_step >= sched_depth:
                break
            self.num_sched_test_step += 1

            if job.partition in partitions_full:
                continue

            valid_nodes, i_free_node = [], 0
            while (
                len(valid_nodes) < job.nodes and
                i_free_node < len(partition_free_nodes[job.partition])
            ):
                node = partition_free_nodes[job.partition][i_free_node]
                if (
                    node.reservation_schedule and
                    self.time + job.reqtime > node.reservation_schedule[0][0]
                ):
                    i_free_node += 1
                    continue
                valid_nodes.append(partition_free_nodes[job.partition].pop(i_free_node))
                for other_partition in set(node.partitions) - { job.partition }:
                    partition_free_nodes[other_partition].remove(node)

            if len(valid_nodes) == job.nodes:
                self._submit(job.start_job(self.time), nodes=valid_nodes)
                jobs_submitted.append(i_job)

            elif not partition_free_nodes[job.partition]:
                partitions_full.add(job.partition)
                if len(partitions_full) == len(self.partitions.partitions):
                    break

        for i_job in sorted(jobs_submitted, reverse=True):
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
                running_jobs={
                    node.running_job for node in self.partitions.reservations[reservation] if (
                        node.running_job
                    )
                },
                reservation=True
            )
            for i_job, nodes in backfill_now:
                job_ready = res_queue[i_job]
                self._submit(job_ready.start_job(self.time), nodes=nodes)

            # if backfill_now:
            #     print(reservation, [ (i_job, len(nodes)) for i_job, nodes in backfill_now ])

            for i_job, _ in sorted(backfill_now, key=lambda job_nodes: job_nodes[0], reverse=True):
               self.queue.reservations[reservation].pop(i_job)

        if self.queue.queue:
            backfill_now = self._get_backfill_jobs(self.partitions.get_free_nodes())
            for i_job, nodes in backfill_now:
                job_ready = self.queue.queue[i_job]
                self._submit(job_ready.start_job(self.time), nodes=nodes)

            # if backfill_now:
            #     print([ (i_job, len(nodes)) for i_job, nodes in backfill_now ])

            for i_job, _ in sorted(backfill_now, key=lambda job_nodes: job_nodes[0], reverse=True):
               self.queue.queue.pop(i_job)

    # def _get_backfill_jobs(self, partition_free_nodes):
    # NOTE if reservation is specified, expects free_nodes and running_jobs to only relate to nodes
    # in the reservation
    def _get_backfill_jobs(self, free_nodes, queue=None, running_jobs=None, reservation=False):
        backfill_now = []

        running_jobs = running_jobs if running_jobs else self.running_jobs
        queue = queue if queue else self.queue.queue

        # NOTE start of EASY bf implementation in git history

        free_blocks = defaultdict(set)
        free_blocks_ready_intervals = set()
        for node in free_nodes:
            # Backfilling for reservation queue
            if reservation:
                free_blocks[(self.time, node.unreserved_time)].add(node)
                free_blocks_ready_intervals.add((self.time, node.unreserved_time))
                continue

            # Free block will end at an upcoming reservation
            if node.reservation_schedule:
                free_blocks[(self.time, node.reservation_schedule[0][0])].add(node)
                free_blocks_ready_intervals.add((self.time, node.reservation_schedule[0][0]))
                continue

            free_blocks[(self.time, datetime.datetime.max)].add(node)
            free_blocks_ready_intervals.add((self.time, datetime.datetime.max))

        if not free_blocks_ready_intervals:
            return backfill_now

        window_end = self.time + self.config.bf_window
        # NOTE When running_jobs is specified assume it includes only jobs from the given
        # reservation
        for job in running_jobs:
            # Some jobs exceeding timelimit and some with zero reqtime
            if job.endlimit <= self.time:
                job.endlimit = self.time + self.config.bf_resolution

            for node in job.assigned_nodes:
                if reservation:
                    free_blocks[(job.endlimit, node.unreserved_time)].add(node)
                    continue

                # NOTE Should check if there is a scheduled reservation coming after a current
                # reservation has finished rather than setting to datetime.max
                if node.reservation:
                    if node.unreserved_time > window_end:
                        break
                    free_blocks[(node.unreserved_time, datetime.datetime.max)].add(node)
                    continue

                if not node.reservation_schedule:
                    free_blocks[(job.endlimit, datetime.datetime.max)].add(node)
                    continue
                end_restriction = node.reservation_schedule[0][0]
                if job.endlimit >= end_restriction:
                    continue
                free_blocks[(job.endlimit, end_restriction)].add(node)

        min_required_block_time = (
            self.time + min(queue, key=lambda job: job.reqtime).reqtime + self.config.bf_resolution
        )
        max_block_time = max(free_blocks_ready_intervals, key=lambda interval: interval[1])[1]

        for i_job, job in enumerate(queue):
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
                                    selected_intervals[interval], key=lambda node: node.id
                                )[num_free_nodes - job.nodes:]
                        )

                    if usage_block_start == self.time:
                        backfill_now.append(
                            (
                                i_job,
                                { node for nodes in selected_intervals.values() for node in nodes }
                            )
                        )

                    for selected_interval, nodes in selected_intervals.items():
                        free_blocks[selected_interval] -= nodes

                        # The original ready now block has been broken and the interval needs
                        # redefining
                        if selected_interval[0] == self.time:
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
        while self.down_nodes and self.down_nodes[0].up_time <= self.time:
            node = self.down_nodes.pop(0)
            node.set_up()

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

            node.set_down(up_time)

            self.down_nodes.append(node)
            self.down_nodes.sort(key=lambda node: node.up_time)

    def _check_reservations(self):
        # Destroy and spawn new sliding reservations
        while self.sliding_reservations and self.sliding_reservations[0][0] <= self.time:
            _, submit, clear, start, end, nodes, name = self.sliding_reservations[0]
            for node in nodes:
                if submit is not None: # At a event where submitting new block
                    node.reservation_schedule.append((start, end, name))
                    node.reservation_schedule.sort(key=lambda schedule: schedule[0])
                else: # At event where clearning block that is about to start
                    for i_res_sched, res_sched in enumerate(node.reservation_schedule):
                        if res_sched[2] == name:
                            node.reservation_schedule.pop(i_res_sched)
                            break

            if not submit: # Finished this window
                self.sliding_reservations.pop(0)
            else: # Created this window, need to clear it next
                self.sliding_reservations[0][0] = clear
                self.sliding_reservations[0][1] = None
                # If clear is same time as next submit, want to clear first
                self.sliding_reservations.sort(
                    key=lambda sliding_res: (sliding_res[0], sliding_res[1] is not None)
                )

        while self.reserved_nodes and self.reserved_nodes[0].unreserved_time <= self.time:
            node = self.reserved_nodes.pop(0)
            self.partitions.remove_node_reservation(node)
            node.set_unreserved()

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

            node.set_reserved(reservation_schedule[2], reservation_schedule[1])
            self.partitions.add_node_reservation(node)

            self.reserved_nodes.append(node)
            self.reserved_nodes.sort(key=lambda node: node.unreserved_time)

    def _print_stats(self):
        if not (self.time.hour != self.previous_print_hour and not self.time.hour % 3):
            return

        self.previous_print_hour = self.time.hour
        print(
            "{} (step {}):\n".format(self.time, self.step_cnt) +
            "Idle Nodes = {} (highmem {})\tNodesReserved = {}(Idle = {})\t" \
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
                        "HPE_RestricLongJobs" in [
                            res_sched[2] for res_sched in node.reservation_schedule
                        ]
                    )
                ),
                sum(
                    1 for node in self.partitions.nodes if (
                        node.reservation_schedule and
                        "HPE_RestricLongJobs" in [
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


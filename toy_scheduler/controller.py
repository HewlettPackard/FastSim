import datetime; from datetime import timedelta
from collections import defaultdict

from config import get_config
from partition import Partitions
from job_queue import Queue
from priority_sorters import MFPrioritySorter


class Controller():
    def __init__(self, config_file):
        self.config = get_config(config_file)

        self.partitions = Partitions(self.config.node_events_dump, self.config.reservations_dump)

        self.queue = Queue(
            self.config.job_dump, valid_reservations=self.partitions.valid_reservations
        )
        self.init_time = self.queue.time
        self.time = self.queue.time
        priority_sorter = MFPrioritySorter(
            self.config.assocs_dump, timedelta(minutes=5), timedelta(days=2), self.init_time, 100,
            500, 300, timedelta(days=14), 0, 10000
        )
        self.queue.set_priority_sorter(priority_sorter)
        self.fairtree = priority_sorter.fairtree

        self.times = [self.time]
        self.power_usage = 0
        self.total_energy = 0.0

        self.job_history = []
        self.running_jobs = []
        self.finished_jobs_step = []
        self.submitted_jobs_step = []

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

            sched, bf, fairtree = False, False, False

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
                previous_small_sched = self.time

            if self.time == next_bf_time:
                bf = True
                next_bf_time += self.config.bf_interval

            self._step(sched, bf, fairtree)
            self.step_cnt += 1

    def _step(self, sched, bf, fairtree):
        # NOTE Regardless of sched or bf still need to:
        # - check for down nodes so the they can go down closer to the correct time rather than in
        # groups at sched/bf steps
        # - finish jobs when they finish so that dependecies/qos submit holds can start accruing
        # time at the correct time
        # - release jobs from qos submit and dependencies holds so that they can start accruing
        # time at the correct time

        self._check_finished_jobs()
        for job in self.finished_jobs_step:
            self.fairtree.job_finish_usage_update(job)
            self.job_history.append(job)
            self.power_usage -= job.true_node_power * job.nodes / 1e+6
            self.total_energy += (
                job.true_node_power * job.nodes * job.runtime.total_seconds() / 1e+9
            )

        self._check_down_nodes()
        self._check_reservations()

        self.running_jobs.sort(key=lambda job: job.end)

        # NOTE Changes to dependencies and qos implementations should mean I won't have to pass all
        # of this
        self.queue.step(
            self.time, self.finished_jobs_step, self.submitted_jobs_step, self.running_jobs,
            self.job_history
        )

        if fairtree:
            self.fairtree.fairshare_calc(self.running_jobs, self.time)

        # NOTE Put these into methods
        self.submitted_jobs_step = []
        if sched:
            for reservation, res_queue in self.queue.reservations.items():
                if not res_queue:
                    continue

                free_nodes = [
                    node for node in self.partitions.nodes if (
                        node.reservation == reservation and not node.running_job
                    )
                ]
                while res_queue and res_queue[0].nodes < len(free_nodes):
                    job = res_queue.pop(0)
                    self.running_jobs.append(job.start_job(self.time))
                    self.power_usage += job.true_node_power * job.nodes / 1e+6
                    for _ in range(job.nodes):
                        node = free_nodes.pop(0)
                        job.assigned_nodes.append(node)
                        node.running_job = job

                    self.submitted_jobs_step.append(job)

            partitions_full = set()

            jobs_submitted = []
            for i_job, job in enumerate(self.queue.queue):
                if job.partition in partitions_full:
                    continue

                if (
                    self.partitions.get_partition_by_name(job.partition).available_nodes() >=
                    job.nodes
                ):
                    self._submit(job.start_job(self.time))
                    jobs_submitted.append(i_job)
                else:
                    self.partitions.get_partition_by_name(job.partition).set_planned()
                    partitions_full.add(job.partition)
                    if len(partitions_full) == len(self.partitions.partitions):
                        break

            for i in sorted(jobs_submitted, reverse=True):
                self.submitted_jobs_step.append(self.queue.queue.pop(i))

            for partition in self.partitions.partitions:
                partition.set_unplanned()

        if (
            bf and self.queue.queue and
            any(
                partition.available_nodes(backfill=True) for partition in (
                    self.partitions.partitions
                )
            )
        ):
            backfill_now = self._get_backfill_jobs()
            for i_job, nodes in backfill_now:
                job_ready = self.queue.queue[i_job]
                self._submit(job_ready.start_job(self.time), nodes=nodes)

            # if backfill_now:
            #     print([ (i_job, len(nodes)) for i_job, nodes in backfill_now ])

            for i_job, _ in sorted(backfill_now, key=lambda job_nodes: job_nodes[0], reverse=True):
               self.submitted_jobs_step.append(self.queue.queue.pop(i_job))

        if sched or bf:
            self.running_jobs.sort(key=lambda job: job.end)

        self.times.append(self.time)

        self._print_stats()

    def _print_stats(self):
        if not (self.time.hour != self.previous_print_hour and not self.time.hour % 3):
            return

        self.previous_print_hour = self.time.hour
        print(
            "{} (step {}):\n".format(self.time, self.step_cnt) +
            "Idle Nodes = {} (highmem {})\tNodesReserved = {}(Idle = {})\t" \
            "NodesHPE_RestrictLongJobs = {} (Idle = {})\t" \
            "NodesDown = {}\tPower = {:.4f} MW\n".format(
                self.partitions.get_partition_by_name("standard").available_nodes(backfill=True),
                self.partitions.get_partition_by_name("highmem").available_nodes(backfill=True),
                sum(1 for node in self.partitions.nodes if node.reservation),
                sum(
                   1 for node in self.partitions.nodes if (
                        node.reservation and not node.running_job and not node.down
                    )
                ),
                sum(1 for node in self.partitions.nodes if node.job_end_restriction),
                sum(
                    1 for node in self.partitions.nodes if (
                        node.job_end_restriction and not node.running_job and not node.down
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

    def _submit(self, job, nodes=None):
        self.running_jobs.append(job)
        self.power_usage += job.true_node_power * job.nodes / 1e+6
        self.total_energy += (
            job.true_node_power * job.nodes * job.runtime.total_seconds() / 1e+9
        )

        if nodes:
            for node in nodes:
                if node.running_job:
                    raise Exception("bruh")
                if not node.free:
                    raise Exception("bruh")
                job.assign_node(node)

        else:
            for node in (
                self.partitions.get_partition_by_name(job.partition).always_available_nodes
            ):
                if node.free:
                    if job.assign_node(node):
                        break

        if len(job.assigned_nodes) != job.nodes:
            raise Exception("bruh")

        return True

    def _get_backfill_jobs(self):
        backfill_now = []

        # NOTE Start of EASY backfilling implementation
        # if self.backfill_opts["EASY"]:
        #     free_nodes = self.available_nodes()
        #     for job in sorted(self.running_jobs, key=lambda job: job.endlimit):
        #         free_nodes += job.nodes
        #         if free_nodes >= queue.queue[0].nodes:
        #             shadow_time = job.endlimit
        #             extra_nodes = free_nodes - queue.queue[0].nodes
        #             break

        #     for i_job, job in enumerate(
        #         list(queue.queue)[:max(len(queue.queue), self.backfill_opts["max_job_test"])]
        #     ):
        #         # Shadow time too short and no extra nodes -> not possible to backfill anymore
        #         if shadow_time <= self.time + self.backfill_opts["resolution"] and not extra_nodes:
        #             return backfill_now

        #         pass

        #     return backfill_now

        free_blocks = defaultdict(set)
        free_blocks_ready_intervals = set()
        for node in self.partitions.nodes:
            if not node.free:
                continue
            if not node.job_end_restriction:
                free_blocks[(self.time, datetime.datetime.max)].add(node)
                free_blocks_ready_intervals.add((self.time, datetime.datetime.max))
                continue
            free_blocks[(self.time, node.job_end_restriction(self.time))].add(node)
            free_blocks_ready_intervals.add((self.time, node.job_end_restriction(self.time)))

        if not free_blocks_ready_intervals:
            return backfill_now
        for job in self.running_jobs:
            # Some jobs exceeding timelimit and some with zero reqtime
            if job.endlimit <= self.time:
                job.endlimit = self.time + self.config.bf_resolution
            for node in job.assigned_nodes:
                if not node.job_end_restriction:
                    free_blocks[(job.endlimit, datetime.datetime.max)].add(node)
                    continue
                end_restriction = node.job_end_restriction(self.time)
                if job.endlimit >= end_restriction:
                    continue
                free_blocks[(job.endlimit, end_restriction)].add(node)

        min_required_block_time = (
            self.time +
            min(self.queue.queue, key=lambda job: job.reqtime).reqtime + self.config.bf_resolution
        )

        max_block_time = max(free_blocks_ready_intervals, key=lambda interval: interval[1])[1]
        partition_num_tested = defaultdict(int)
        # Dont consider jobs that require partitions with no available nodes
        partition_maxes = {
            partition.name : (
                self.config.bf_max_job_test if partition.available_nodes(backfill=True) else 0
            ) for partition in self.partitions.partitions
        }
        for i_job, job in enumerate(self.queue.queue):
            # Mimics bf not seeing jobs submitted after it gets initial lock
            # if self.config.bf_continue and job.submit > self.time - self.config.bf_interval:
            #     continue
            # break if no blocks or only <= min blocks available for immediate backfill
            if max_block_time < min_required_block_time:
                break

            # Empirical max jobs before reaching bf_max_time
            # if sum(partition_num_tested.values()) > BACKFILL_OPTS["max_test_timelimit"]:
            #     break
            if partition_num_tested[job.partition] >= partition_maxes[job.partition]:
                continue
            partition_num_tested[job.partition] += 1

            reqtime = job.reqtime + self.config.bf_resolution
            # Only need to plan nodes for jobs that may be relevant to immediate scheduling
            if self.time + reqtime > max_block_time:
                continue

            free_nodes = 0
            selected_intervals = {}

            for interval, nodes in sorted(free_blocks.items(), key=lambda entry: entry[0][0]):
                valid_nodes = [
                        node for node in sorted(nodes, key=lambda node: node.id) if (
                        self.partitions.get_partition_by_name(job.partition) in node.partitions
                    )
                ]
                if valid_nodes:
                    selected_intervals[interval] = valid_nodes
                    free_nodes += len(valid_nodes)
                    latest_interval = interval

                if job.nodes <= free_nodes:
                    usage_block_start = max(selected_intervals.keys(), key=lambda key: key[0])[0]

                    for interval in list(selected_intervals.keys()):
                        if usage_block_start + reqtime > interval[1]:
                            free_nodes -= len(selected_intervals.pop(interval))

                    if job.nodes > free_nodes:
                        continue

                    usage_block_end = usage_block_start + reqtime

                    # Remove nodes we don't need from the latest interval added
                    for i in range(free_nodes - job.nodes):
                        selected_intervals[latest_interval].pop()

                    if usage_block_start == self.time:
                        backfill_now.append(
                            (
                                i_job,
                                { node for nodes in selected_intervals.values() for node in nodes }
                            )
                        )

                    for key, nodes in selected_intervals.items():
                        free_blocks[key] -= set(nodes)

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

    def _check_finished_jobs(self):
        self.finished_jobs_step = []
        while self.running_jobs and self.running_jobs[0].end <= self.time:
            self.finished_jobs_step.append(self.running_jobs.pop(0))
            self.finished_jobs_step[-1].end_job()

    def _check_down_nodes(self):
        try:
            while self.down_nodes and self.down_nodes[0].up_time <= self.time:
                node = self.down_nodes.pop(0)
                node.set_up()
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

            node.set_down(up_time)

            self.down_nodes.append(node)
            self.down_nodes.sort(key=lambda node: node.up_time)

    def _check_reservations(self):
        while self.reserved_nodes and self.reserved_nodes[0].unreserved_time <= self.time:
            node = self.reserved_nodes.pop(0)
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

            self.reserved_nodes.append(node)
            self.reserved_nodes.sort(key=lambda node: node.unreserved_time)


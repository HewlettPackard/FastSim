from collections import OrderedDict


class Backfiller:
    def __init__(self, config):
        self.bf_window = config.bf_window.total_seconds()
        self.bf_end_padding = (config.OverTimeLimit + config.KillWait).total_seconds()
        self.bf_resolution = config.bf_resolution.total_seconds()
        self.bf_max_relevant_start = (
            (config.bf_max_time - config.bf_yield_interval).total_seconds()
        )
        self.bf_try_per_lock_hold = int(
            config.bf_yield_interval.total_seconds() * config.approx_bf_try_per_sec
        )
        self.bf_max_lock_holds = int(
            config.bf_max_time / (config.bf_yield_interval + config.bf_yield_sleep)
        )
        self.bf_job_max_test = config.bf_job_max_test
        self.bf_loop_active = False

        self.bf_locks_remaining = None
        self.bf_free_blocks = None
        self.bf_time = None
        self.bf_nodes_free_now_max_reqtimes = None
        self.bf_max_reqtime = None
        self.bf_queue_min_reqtime = None
        self.bf_job_reqtimes = None
        self.bf_secs_past = None

    def prep_new_bf(self, queue):
        self.bf_locks_remaining = self.bf_max_lock_holds
        self.bf_time = self.time
        self.bf_secs_past = 0

        self.bf_free_blocks, self.bf_nodes_free_now_max_reqtimes = {}, {}

        for resv, free_block in self.partitions.free_blocks.items():
            self.bf_free_blocks[resv] = defaultdict(set)
            bf_free_blocks = self.bf_free_blocks[resv]
            self.bf_nodes_free_now_max_reqtimes[resv] = {}
            bf_nodes_free_now_max_reqtimes = self.bf_nodes_free_now_max_reqtimes[resv]

            for interval, nodes in free_block.items():
                if interval[1] == datetime.datetime.max:
                    interval_f = self.bf_window
                else:
                    interval_f = min(
                        (interval[1] - self.time).total_seconds(), self.bf_window
                    )

                if interval[0] <= self.time:
                    for node in nodes:
                        if node.running_job is not None:
                            interval_i = max(
                                (
                                    (node.running_job.endlimit - self.time).total_seconds() +
                                    self.bf_end_padding
                                ),
                                1
                            )

                        else:
                            interval_i = 0

                        bf_free_blocks[(interval_i, interval_f)].add(node)

                        if interval_i <= self.bf_max_relevant_start:
                            bf_nodes_free_now_max_reqtimes[node] = interval_f - interval_i

                else:
                    interval_i = (interval[0] - self.time).total_seconds()
                    if interval_i >= self.bf_window:
                        continue

                    bf_free_blocks[(interval_i, interval_f)].update(nodes)

                if interval_i <= self.bf_max_relevant_start:
                    for node in nodes:
                        bf_nodes_free_now_max_reqtimes[node] = interval_f - interval_i

        self.bf_max_reqtime = {
            resv : max(nodes_free_now_min_reqtimes.values()) if nodes_free_now_min_reqtimes else 0
            for resv, nodes_free_now_min_reqtimes in self.bf_nodes_free_now_max_reqtimes.items()
        }

        self.bf_queue, self.bf_job_ordered_reqtimes = [], {}

        # NOTE Only checking for the normal queue because resv queues are usually small. Should
        # really just have a single queue with reservation sorted to the front, this would clean
        # up some code
        max_test_from_normal_q = (
            self.bf_max_job_test - sum(len(q) for q in queue.reservations.values())
        )
        if max_test_from_normal_q > 0:
            self.bf_job_ordered_reqtimes[""] = {}
            for job in queue.queue[-max_test_from_normal_q:]:
                self.bf_queue.append(job)
                self.bf_job_ordered_reqtimes[""][job] = job.reqtime.total_seconds()
        for resv, resv_q in queue.reservations.items():
            self.bf_job_ordered_reqtimes[resv] = {}
            for job in resv_q:
                self.bf_queue.append(job)
                self.bf_job_ordered_reqtimes[resv][job] = job.reqtime.total_seconds()
        # Earliest reqtime job at front so I can call next(iter()) to grab it
        self.bf_job_ordered_reqtimes = {
            resv : OrderedDict(
                {
                    job : reqtime
                    for job, reqtime in sorted(
                        job_ordered_reqtimes.items(), key=lambda job_reqtime: job_reqtime[1]
                    )
                }
            )
            for resv, job_ordered_reqtimes in self.bf_job_ordered_reqtimes.items()
        }
        # print(self.bf_job_ordered_reqtimes)
        # print(next(iter(self.bf_job_ordered_reqtimes)))
        # To avoid needing to check if we are on the final iteration
        # for resv in self.bf_job_ordered_reqtimes:
        #     self.bf_job_ordered_reqtimes[resv]["dummy"] = self.bf_window_in_res

        # If there are no more jobs in the queue that could possibly be started now, we shouldnt
        # waste time backfilling. Still want to go throught the yield cycles and just pretend
        # we are actually backfilling. This is more likely to flick to true as the backfill
        # schedule fills up and jobs are processed from the queue.
        self.bf_done = {
            resv : (
                next(iter(job_ordered_reqtimes.values())) > self.bf_max_reqtime[resv]
                if job_ordered_reqtimes
                else True
            )
            for resv, job_ordered_reqtimes in self.bf_job_ordered_reqtimes.items()
        }

from collections import defaultdict
import datetime; from datetime import timedelta

import pandas as pd

from helpers import convert_nodelist_to_node_nums

from job_queue import Job


# NOTE Do I really need this partitions class, not doing anything but hiding the data prep atm
class Partitions:
    def __init__(self, nid_data, partition_data):
        self.partitions = {
            Partition(name, data["prio_tier"], data["prio_jobfactor"])
            for name, data in partition_data.items()
        }
        self.partitions_by_name = { partition.name : partition for partition in self.partitions }

        self.nodes = set()
        for nid, data in nid_data.items():
            node = Node(nid, data["weight"], data["down_schedule"], data["resv_schedule"])
            for p_name in data["partitions"]:
                self.partitions_by_name[p_name].add_node(node)
            self.nodes.add(node)

        print("Using partitions:")
        for partition in self.partitions:
            print(partition.name, " - ", len(partition.nodes), " nodes", sep="")
        print("With {} unique nodes total".format(len(self.nodes)))

        self.reservations = defaultdict(list)

        # { reservation : { interval : nodes, ... }, ... }
        self.free_blocks = defaultdict(lambda: defaultdict(set))
        for node in self.nodes:
            self.free_blocks[node.reservation][
                (node.interval_times[0], node.interval_times[-1])
            ].add(node)

    def remove_free_block(self, node):
        interval = (node.interval_times[0], node.interval_times[-1])
        self.free_blocks[node.reservation][interval].remove(node)
        if not self.free_blocks[node.reservation][interval]:
            self.free_blocks[node.reservation].pop(interval)

    def add_free_block(self, node):
        interval = (node.interval_times[0], node.interval_times[-1])
        self.free_blocks[node.reservation][interval].add(node)

    def clean_free_blocks(self, time, bf_resolution):
        # Give all overrunning jobs some extra time and set all intervals with a start time in the
        # past to a start time now

        for res, free_blocks in self.free_blocks.items():
            for interval in list(free_blocks):
                if interval[0] > time:
                    continue

                for node in list(free_blocks[interval]):
                    if node.running_job:
                        node.interval_times[0] = time + bf_resolution
                    elif interval[0] != time:
                        node.interval_times[0] = time
                    else:
                        continue

                    free_blocks[interval].remove(node)

                    # Guard against reservations overlapping with eachother. Remove if this shift
                    # causes the earliest reservation to fall entirely behind current time.
                    while (
                        len(node.interval_times) > 2 and
                        node.interval_times[0] >= node.interval_times[2]
                    ):
                        plnd_job = None
                        for job in node.jobs_plnd:
                            if job.planned_block[0][0] == node.interval_times[1]:
                                plnd_job = job
                                break
                        plnd_job.planned_block[1].remove(node)
                        if not plnd_job.planned_block[1]:
                            plnd_job.planned_block = None
                        node.jobs_plnd.remove(plnd_job)

                        node.interval_times.pop(2)
                        node.interval_times.pop(1)

                    free_blocks[(node.interval_times[0], node.interval_times[-1])].add(node)

                if not free_blocks[interval]:
                    free_blocks.pop(interval)

    def clear_planned_blocks(self, target=None):
        if isinstance(target, Job):
            job = target

            if job.planned_block is None:
                return

            free_blocks = self.free_blocks[job.reservation]

            for node in job.planned_block[1]:
                i_start = node.interval_times.index(job.planned_block[0][0])
                if node.interval_times[i_start] != node.interval_times[i_start + 1]:
                    i_start -= 1
                node.interval_times.pop(i_start + 2)
                node.interval_times.pop(i_start + 1)
                node.jobs_plnd.remove(job)

            job.planned_block = None

        # Clears all free blocks related to this node, will need to be readded to free_blocks
        # afrterwards if node not doing down
        elif isinstance(target, Node):
            node = target

            while node.jobs_plnd:
                job = node.jobs_plnd.pop()
                job.planned_block[1].remove(node)
                if not job.planned_block[1]:
                    job.planned_block = None

            node.interval_times = [node.interval_times[0], node.interval_times[-1]]

        else:
            raise NotImplementedError("Something's gone wrong")

    def get_partition_by_name(self, name):
        return self.partitions_by_name[name]


class Partition:
    def __init__(self, name, priority_tier, priority_weight):
        self.name = name
        self.priority_tier = priority_tier
        self.priority_weight = priority_weight # Normalised s.t. partition with greatest has 1

        self.nodes = []

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        if isinstance(other, Partition):
            return self.name == other.name
        return False

    def add_node(self, node):
        node.partitions.append(self)
        self.nodes.append(node)
        # TODO does this still need to be sorted? can it just be a set
        self.nodes.sort(key=lambda node: (node.weight, node.id)) # Small weights get priority


class Node:
    def __init__(self, num, weight, down_schedule, reservation_schedule):
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

        self.partitions = []

        self.interval_times = [
            datetime.datetime.min,
            datetime.datetime.max if not reservation_schedule else reservation_schedule[-1][0]
        ]
        self.jobs_plnd = set()

        self.bf_free_blocks_start = None

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if isinstance(other, Node):
            return self.id == other.id
        return False

    def set_reserved(self, reservation_name, end_time):
        self.reservation = reservation_name
        self.unreserved_time = end_time
        if self.down or not self.free:
            return
        self.free = False

    def set_unreserved(self):
        self.reservation = ""
        self.unreserved_time = None
        if self.down or self.running_job:
            return
        self.free = True

    def set_down(self, up_time):
        self.down = True
        self.up_time = up_time
        # If job is already running it is allowed to finish
        if not self.free:
            return
        self.free = False

    def set_up(self):
        self.down = False
        self.up_time = None
        if self.reservation:
            return
        self.free = True

    def set_free(self):
        if self.down or self.reservation:
            return
        self.free = True

    def set_busy(self):
        if self.reservation:
            return
        self.free = False


from collections import defaultdict
import datetime; from datetime import timedelta

import pandas as pd

from helpers import convert_nodelist_to_node_nums


# NOTE Do I really need this partitions class, not doing anything but hiding the data prep atm
class Partitions:
    def __init__(self, node_events_dump, res_dump):
        ret_tuple = self._get_nodes_partitions(node_events_dump, res_dump)
        self.nodes, self.partitions, self.valid_reservations, self.hpe_restrictlong_nodes = (
            ret_tuple
        )

        self.reservations = defaultdict(list)

        self.partitions_by_name = { partition.name : partition for partition in self.partitions }

        # { reservation : { interval : nodes, ... }, ... }
        self.free_blocks = defaultdict(lambda: defaultdict(set))
        for node in self.nodes:
            self.free_blocks[node.reservation][node.free_block_interval].add(node)

    def remove_free_block(self, node, from_back=False):
        if not from_back:
            interval = (node.interval_times[0], node.interval_times[1])
        else:
            interval = (node.interval_times[-2], node.interval_times[-1])
        self.free_blocks[node.reservation][interval].remove(node)
        if not self.free_blocks[node.reservation][interval]:
            self.free_blocks[node.reservation].pop(interval)

    def add_free_block(self, node, from_back=False):
        if not from_back:
            interval = (node.interval_times[0], node.interval_times[1])
        else:
            interval = (node.interval_times[-2], node.interval_times[-1])
        self.free_blocks[node.reservation][interval].add(node)

    def clean_free_blocks(self, time, bf_resolution):
        for res, free_blocks in self.free_blocks.items():
            for interval in list(free_blocks):
                if interval[0] > time: # TODO use longest over-run of reqtime for these checks
                    continue

                for node in list(free_blocks[interval]):
                    # If free start time is in the past or now and there is job running want to
                    # shift it forward
                    if node.running_job:
                        # print("overrun {}, will overrun for another {}".format(time - node.running_job.endlimit, node.running_job.end - time))
                        node.free_block_interval = (
                            time + bf_resolution, node.free_block_interval[1]
                        )
                        node.interval_times[0] = node.free_block_interval[0]

                    # If free start time is in the past we want to move to current time
                    elif interval[0] != time:
                        node.free_block_interval = (time, node.free_block_interval[1])
                        node.interval_times[0] = node.free_block_interval[0]

                    else:
                        continue

                    free_blocks[interval].remove(node)

                    # Guard against reservations overlapping with eachother. Remove if this shift
                    # causes a reservation to fall entirely behind current time.
                    if (
                        len(node.interval_times) > 2 and
                        node.interval_times[0] >= node.interval_times[2]
                    ):
                        plnd_job = None
                        for job in node.jobs_with_plans:
                            for plnd_interval, nodes in job.planned_blocks.items():
                                if node not in nodes:
                                    continue
                                plnd_job = job
                                break
                        plnd_job.planned_blocks[plnd_interval].remove(node)
                        if not plnd_job.planned_blocks[plnd_interval]:
                            plnd_job.planned_blocks.pop(plnd_interval)

                        free_blocks[(node.interval_times[2], node.interval_times[3])].remove(node)
                        node.jobs_with_plans.remove(plnd_job)
                        node.interval_times.pop(2)
                        node.interval_times.pop(1)

                    free_blocks[(node.interval_times[0], node.interval_times[1])].add(node)

                if not free_blocks[interval]:
                    free_blocks.pop(interval)
                    continue


    def get_partition_by_name(self, name):
        return self.partitions_by_name[name]

    def _get_nodes_partitions(self, node_events_dump, reservations_dump):
        df_events = pd.read_csv(
            node_events_dump, delimiter='|', lineterminator='\n', header=0,
            usecols=["NodeName", "TimeStart", "TimeEnd", "State"]
        )
        df_events = df_events.loc[
            (
                (df_events.NodeName.notna()) & (df_events.NodeName.str.contains("nid")) &
                (df_events.TimeEnd != "Unknown") & (df_events.TimeStart != "Unknown")
            )
        ]

        df_events.TimeStart = pd.to_datetime(df_events.TimeStart, format="%Y-%m-%dT%H:%M:%S")
        df_events.TimeEnd = pd.to_datetime(df_events.TimeEnd, format="%Y-%m-%dT%H:%M:%S")
        df_events["Duration"] = df_events.apply(lambda row: (row.TimeEnd - row.TimeStart), axis=1)
        df_events.State = df_events.State.apply(lambda row: "DRAIN" if "DRAIN" in row else "DOWN")
        df_events["Id"] = df_events.NodeName.apply(lambda row: int(row.split("nid")[1]))

        # Basic reservations implementation, only consider reservations that are still in the database
        # (this looks sufficient for ARCHER2) and ignore any flags such as REPLACE_DOWN
        df_reservations = pd.read_csv(
            reservations_dump, delimiter='|', lineterminator='\n', header=0,
            usecols=["RESV_NAME", "STATE", "START_TIME", "END_TIME", "NODELIST"]
        )

        df_reservations.START_TIME = pd.to_datetime(
            df_reservations.START_TIME, format="%Y-%m-%dT%H:%M:%S"
        )
        df_reservations.END_TIME = pd.to_datetime(df_reservations.END_TIME, format="%Y-%m-%dT%H:%M:%S")
        df_reservations.NODELIST = df_reservations.NODELIST.apply(convert_nodelist_to_node_nums)

        valid_reservations = [ row.RESV_NAME for _, row in df_reservations.iterrows() ]

        df_reservations = df_reservations.explode("NODELIST")

        partitions = {
            "standard" : Partition("standard", 1, 1.0), "highmem" : Partition("highmem", 1, 1.0)
        }
        nodes, hpe_restrictlong_nids = [], []
        for nid in range(1000, 6860):
            down_schedule = []
            for _, row in df_events.loc[(df_events.Id == nid)].iterrows():
                down_schedule.append([row.TimeStart, row.Duration, row.State])
            down_schedule.sort(key=lambda schedule: schedule[0])

            reservation_schedule = []
            for _, row in df_reservations.loc[(df_reservations.NODELIST == nid)].iterrows():
                # Think this behaviour is being controlled by a maintenance script running in a
                # screen session
                if row.RESV_NAME == "HPE_RestrictLongJobs":
                    hpe_restrictlong_nids.append(nid)
                    continue
                reservation_schedule.append((row.START_TIME, row.END_TIME, row.RESV_NAME))
            reservation_schedule.sort(key=lambda schedule: schedule[0], reverse=True)

            # Merge any adjacent events
            i_event = 0
            while i_event < len(down_schedule) - 1:
                event, next_event = down_schedule[i_event], down_schedule[i_event + 1]
                if event[0] + event[1] == next_event[0] and event[2] == next_event[2]:
                    down_schedule[i_event][1] += next_event[1]
                    down_schedule.pop(i_event + 1)
                    continue
                i_event += 1
            # Now reverse so that we can pop from the end
            down_schedule.sort(key=lambda schedule: schedule[0], reverse=True)

            if (2756 <= nid <= 3047) or (6376 <= nid <= 6667):
                nodes.append(
                    Node(
                        nid, 1000, down_schedule=down_schedule,
                        reservation_schedule=reservation_schedule,
                    )
                )
                partitions["highmem"].add_node(nodes[-1])
            else:
                nodes.append(
                    Node(
                        nid, 0, down_schedule=down_schedule,
                        reservation_schedule=reservation_schedule,
                    )
                )
            partitions["standard"].add_node(nodes[-1])

        return (
            nodes, list(partitions.values()), valid_reservations,
            [ node for node in nodes if node.id in hpe_restrictlong_nids ]
        )


class Partition:
    def __init__(self, name, priority_tier, priority_weight):
        self.name = name
        self.priority_tier = priority_tier
        self.priority_weight = priority_weight # Normalised s.t. partition with greatest has 1

        self.nodes = []

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name

    def add_node(self, node):
        node.partitions.append(self)
        self.nodes.append(node)
        # TODO does this still need to be sorted? can it just be a set
        self.nodes.sort(key=lambda node: (node.weight, node.id)) # Small weights get priority


class Node:
    def __init__(self, num, weight=0, down_schedule=[], reservation_schedule=[]):
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

        self.free_block_interval = (
            datetime.datetime.min,
            datetime.datetime.max if not reservation_schedule else reservation_schedule[-1][0]
        )
        self.interval_times = [self.free_block_interval[0], self.free_block_interval[1]]
        self.jobs_with_plans = set()

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

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


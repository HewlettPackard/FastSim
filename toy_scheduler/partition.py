from collections import defaultdict
import datetime; from datetime import timedelta

import pandas as pd

from helpers import convert_nodelist_to_node_nums

from job_queue import Job


# NOTE Do I really need this partitions class, not doing anything but hiding the data prep atm
class Partitions:
    def __init__(self, slurm_conf, considered_partitions, node_events_dump, res_dump):
        ret_tuple = self._get_nodes_partitions(
            slurm_conf, considered_partitions, node_events_dump, res_dump
        )
        self.nodes, self.partitions, self.valid_reservations, self.hpe_restrictlong_nodes = (
            ret_tuple
        )

        self.reservations = defaultdict(list)

        self.partitions_by_name = { partition.name : partition for partition in self.partitions }

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

    def _get_nodes_partitions(
        self, slurm_conf, considered_partitions, node_events_dump, reservations_dump
    ):
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

        partitions = {}
        nid_features, nodesets, nid_partitions, nid_weight = {}, {}, defaultdict(set), {}

        with open(slurm_conf, "r") as f:
            for line in f:
                if not line.startswith("NodeName"):
                    continue
                line = line.strip("\n")

                # NOTE Just assuming all node names start with nid since I don't have time to do
                # this properly. For archer and lumi this is fine.
                if "nid" not in line.split("NodeName=")[1].split(" ")[0]:
                    continue

                nids = convert_nodelist_to_node_nums(line.split("NodeName=")[1].split(" ")[0])

                features = set(line.split("Feature=")[1].split(" ")[0].split(","))

                if "Weight" not in line:
                    weight = 1
                else:
                    weight = int(line.split("Weight=")[1].split(" ")[0])

                for nid in nids:
                    nid_features[nid] = features
                    nid_weight[nid] = weight

            f.seek(0)
            for line in f:
                if not line.startswith("NodeSet"):
                    continue
                line = line.strip("\n")

                name = line.split("NodeSet=")[1].split(" ")[0]
                nodeset_features = set(line.split("Feature=")[1].split(" ")[0].split(","))

                nodesets[name] = [
                    nid
                    for nid, features in nid_features.items()
                        if features.intersection(nodeset_features)
                ]

            f.seek(0)
            for line in f:
                if not line.startswith("PartitionName"):
                    continue
                line = line.strip("\n")

                name = line.split("PartitionName=")[1].split(" ")[0]

                if name not in considered_partitions:
                    continue

                if "PriorityTier" not in line:
                    prio_tier = 1
                else:
                    prio_tier = int(line.split("PriorityTier=")[1].split(" ")[0])

                if "PriorityJobFactor" not in line:
                    prio_jobfactor = 1
                else:
                    prio_jobfactor = int(line.split("PriorityJobFactor=")[1].split(" ")[0])

                partition = Partition(name, prio_tier, prio_jobfactor)
                partitions[name] = partition

                nodes = line.split(" Nodes=")[1].split(" ")[0]

                if nodes in nodesets:
                    for nid in nodesets[nodes]:
                        nid_partitions[nid].add(partition)
                else:
                    for nid in convert_nodelist_to_node_nums(nodes):
                        nid_partitions[nid].add(partition)

        nodes, hpe_restrictlong_nids = [], []
        for nid in nid_partitions:
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

            nodes.append(
                Node(
                    nid, nid_weight[nid], down_schedule=down_schedule,
                    reservation_schedule=reservation_schedule,
                )
            )
            for partition in nid_partitions[nid]:
                partition.add_node(nodes[-1])

        for name, partition in partitions.items():
            print(name, len(set(partition.nodes)))

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
        if isinstance(other, Partition):
            return self.name == other.name
        return False

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

        self.interval_times = [
            datetime.datetime.min,
            datetime.datetime.max if not reservation_schedule else reservation_schedule[-1][0]
        ]
        self.jobs_plnd = set()

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


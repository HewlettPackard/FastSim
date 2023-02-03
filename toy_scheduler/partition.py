from datetime import timedelta

import pandas as pd

from helpers import convert_nodelist_to_node_nums


class Partitions:
    def __init__(self, node_events_dump, res_dump):
        self.nodes, self.partitions, self.valid_reservations = self._get_nodes_partitions(
            node_events_dump, res_dump
        )

        self.partitions_by_name = { partition.name : partition for partition in self.partitions }

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
        nodes = []
        for nid in range(1000, 6860):
            down_schedule = []
            for _, row in df_events.loc[(df_events.Id == nid)].iterrows():
                down_schedule.append([row.TimeStart, row.Duration, row.State])
            down_schedule.sort(key=lambda schedule: schedule[0])

            reservation_schedule, job_end_restriction = [], None
            for _, row in df_reservations.loc[(df_reservations.NODELIST == nid)].iterrows():
                # Think this behaviour is being controlled by a maintenance script running in a
                # screen session
                if row.RESV_NAME == "HPE_RestrictLongJobs":
                    job_end_restriction = (
                        lambda time: (
                            (time + timedelta(hours=1, minutes=5)).replace(minute=0, second=0)
                        )
                    )
                    continue
                reservation_schedule.append((row.START_TIME, row.END_TIME, row.RESV_NAME))
            reservation_schedule.sort(key=lambda schedule: schedule[0])

            # Merge any adjacent events
            i_event = 0
            while i_event < len(down_schedule) - 1:
                event, next_event = down_schedule[i_event], down_schedule[i_event + 1]
                if event[0] + event[1] == next_event[0] and event[2] == next_event[2]:
                    down_schedule[i_event][1] += next_event[1]
                    down_schedule.pop(i_event + 1)
                    continue
                i_event += 1

            if (2756 <= nid <= 3047) or (6376 <= nid <= 6667):
                nodes.append(
                    Node(
                        nid, 1000, down_schedule=down_schedule,
                        reservation_schedule=reservation_schedule,
                        job_end_restriction=job_end_restriction
                    )
                )
                partitions["highmem"].add_node(nodes[-1])
            else:
                nodes.append(
                    Node(
                        nid, 0, down_schedule=down_schedule,
                        reservation_schedule=reservation_schedule,
                        job_end_restriction=job_end_restriction
                    )
                )
            partitions["standard"].add_node(nodes[-1])

        return nodes, list(partitions.values()), valid_reservations


class Partition:
    def __init__(self, name, priority_tier, priority_weight):
        self.name = name
        self.priority_tier = priority_tier
        self.priority_weight = priority_weight # Normalised s.t. partition with greatest has 1

        self.nodes = []
        self.planned_nodes = []
        self.always_available_nodes = []
        self.backfill_only_nodes = []

        self.num_available = 0
        self.num_available_only_backfill = 0

    def add_node(self, node):
        node.partitions.append(self)
        self.nodes.append(node)
        self.nodes.sort(key=lambda node: (node.weight, node.id)) # Small weights get priority
        if not node.job_end_restriction:
            self.always_available_nodes.append(node)
            self.num_available += node.free
        else:
            self.backfill_only_nodes.append(node)
            self.num_available_only_backfill += node.free

    def available_nodes(self, backfill=False):
        if backfill:
            return self.num_available + self.num_available_only_backfill

        return self.num_available

    def set_planned(self):
        for node in self.nodes:
            if node.free:
                node.set_planned()
                self.planned_nodes.append(node)

    def set_unplanned(self):
        while self.planned_nodes:
            self.planned_nodes.pop(0).set_unplanned()


class Node():
    def __init__(
        self, num, weight=0, down_schedule=[], reservation_schedule=[], job_end_restriction=None
    ):
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
        self.job_end_restriction = job_end_restriction

        self.planned = False

        self.partitions = []

    def set_planned(self):
        self.planned = True
        self.free = False
        for partition in self.partitions:
            if self.job_end_restriction: # NOTE This is so janky but Im desparate, fix in refactor
                partition.num_available_only_backfill -= 1
                continue
            partition.num_available -= 1

    def set_unplanned(self):
        self.planned = False
        self.free = True
        for partition in self.partitions:
            if self.job_end_restriction:
                partition.num_available_only_backfill += 1
                continue
            partition.num_available += 1

    def set_reserved(self, reservation_name, end_time):
        self.reservation = reservation_name
        self.unreserved_time = end_time
        if self.down or not self.free:
            return
        self.free = False
        for partition in self.partitions:
            if self.job_end_restriction:
                partition.num_available_only_backfill -= 1
                continue
            partition.num_available -= 1

    def set_unreserved(self):
        self.reservation = ""
        self.unreserved_time = None
        if self.down or self.running_job:
            return
        self.free = True
        for partition in self.partitions:
            if self.job_end_restriction:
                partition.num_available_only_backfill += 1
                continue
            partition.num_available += 1

    def set_down(self, up_time):
        self.down = True
        self.up_time = up_time
        # If job is already running it is allowed to finish
        if not self.free:
            return
        self.free = False
        for partition in self.partitions:
            if self.job_end_restriction:
                partition.num_available_only_backfill -= 1
                continue
            partition.num_available -= 1

    def set_up(self):
        self.down = False
        self.up_time = None
        if self.reservation:
            return
        self.free = True
        for partition in self.partitions:
            if self.job_end_restriction:
                partition.num_available_only_backfill += 1
                continue
            partition.num_available += 1

    def set_free(self):
        if self.down or self.reservation:
            return
        self.free = True
        for partition in self.partitions:
            if self.job_end_restriction:
                partition.num_available_only_backfill += 1
                continue
            partition.num_available += 1

    def set_busy(self):
        self.free = False
        for partition in self.partitions:
            if self.job_end_restriction:
                partition.num_available_only_backfill -= 1
                continue
            partition.num_available -= 1


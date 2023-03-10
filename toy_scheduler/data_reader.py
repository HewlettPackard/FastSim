from collections import defaultdict
import datetime; from datetime import timedelta
from copy import deepcopy

import pandas as pd

from helpers import (
    convert_nodelist_to_node_nums, timelimit_str_to_timedelta, convert_to_raw, get_sbatch_cli_arg
)


class SlurmDataReader:
    def __init__(self, slurm_conf, node_events_dump, resv_dump, job_dump, qos_dump):
        self.slurm_conf = slurm_conf
        self.node_events_dump = node_events_dump
        self.resv_dump = resv_dump
        self.job_dump = job_dump
        self.qos_dump = qos_dump

    def get_nodes_partitions(
        self, considered_partitions, hpe_restrictlong_sliding_res, max_sim_t, nodes_down_in_blades
    ):
        df_events = pd.read_csv(
            self.node_events_dump, delimiter='|', lineterminator='\n', header=0,
            usecols=["NodeName", "TimeStart", "TimeEnd", "State", "Reason"]
        )
        df_events = df_events.loc[
            (
                (df_events.NodeName.notna()) & (df_events.NodeName.str.contains("nid")) &
                (df_events.TimeStart != "Unknown")
            )
        ]

        df_events.TimeStart = pd.to_datetime(df_events.TimeStart, format="%Y-%m-%dT%H:%M:%S")
        df_events.loc[(df_events.TimeEnd == "Unknown"), "TimeEnd"] = max_sim_t
        df_events.TimeEnd = pd.to_datetime(df_events.TimeEnd, format="%Y-%m-%dT%H:%M:%S")
        df_events["Duration"] = df_events.apply(lambda row: (row.TimeEnd - row.TimeStart), axis=1)
        df_events.State = df_events.State.apply(lambda row: "DRAIN" if "DRAIN" in row else "DOWN")
        df_events["Id"] = df_events.NodeName.apply(lambda row: int(row.split("nid")[1]))

        # NOTE Not considering any reservation flags
        df_resv = pd.read_csv(
            self.resv_dump, delimiter='|', lineterminator='\n', header=0,
            usecols=["RESV_NAME", "STATE", "START_TIME", "END_TIME", "NODELIST"]
        )

        df_resv.START_TIME = pd.to_datetime(df_resv.START_TIME, format="%Y-%m-%dT%H:%M:%S")
        df_resv.END_TIME = pd.to_datetime(df_resv.END_TIME, format="%Y-%m-%dT%H:%M:%S")
        df_resv.NODELIST = df_resv.NODELIST.apply(convert_nodelist_to_node_nums)

        valid_resv = [ row.RESV_NAME for _, row in df_resv.iterrows() ]

        df_resv = df_resv.explode("NODELIST")

        partition_data = {}
        nid_features, nodesets, nid_partitions, nid_weight = {}, {}, defaultdict(set), {}

        with open(self.slurm_conf, "r") as f:
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

                if "Weight=" not in line:
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

                partition_data[name] = {
                    "prio_tier" : prio_tier, "prio_jobfactor" : prio_jobfactor
                }

                nodes = line.split(" Nodes=")[1].split(" ")[0]

                if nodes in nodesets:
                    for nid in nodesets[nodes]:
                        nid_partitions[nid].add(name)
                else:
                    for nid in convert_nodelist_to_node_nums(nodes):
                        nid_partitions[nid].add(name)

        max_partition_prio = max(data["prio_jobfactor"] for data in partition_data.values())
        if max_partition_prio:
            for data in partition_data.values():
                data["prio_jobfactor"] /= max_partition_prio

        nid_data, hpe_restrictlong_nids = {}, []

        for nid in nid_partitions:
            down_schedule = []
            for _, row in df_events.loc[(df_events.Id == nid)].iterrows():
                down_schedule.append([row.TimeStart, row.Duration, row.State, row.Reason])
            down_schedule.sort(key=lambda schedule: schedule[0])

            resv_schedule = []
            for _, row in df_resv.loc[(df_resv.NODELIST == nid)].iterrows():
                # Think this behaviour is being controlled by a maintenance script running in a
                # screen session
                if row.RESV_NAME == "HPE_RestrictLongJobs":
                    hpe_restrictlong_nids.append(nid)
                    continue
                resv_schedule.append((row.START_TIME, row.END_TIME, row.RESV_NAME))
            resv_schedule.sort(key=lambda schedule: schedule[0], reverse=True)

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

            nid_data[nid] = {
                "weight" : nid_weight[nid], "down_schedule" : down_schedule,
                "resv_schedule" : resv_schedule, "partitions" : nid_partitions[nid]
            }

        # It looks like LUMI puts blades with a down node into a maintenance reservation that
        # blocks all jobs. Recreate this by putting all the nodes in blade down when one of them
        # goes down
        if nodes_down_in_blades:
            # Give all nodes in blade same down schedule
            for first_blade_nid in list(nid_data)[::4]:
                shared_drain_schedule = [
                    down_block
                    for nid in range(first_blade_nid, first_blade_nid + 4)
                        if nid in nid_data
                        for down_block in nid_data[nid]["down_schedule"]
                            if down_block[2] == "DRAIN"
                ]

                for nid in range(first_blade_nid, first_blade_nid + 4):
                    if nid not in nid_data:
                        continue

                    for drain_block in shared_drain_schedule:
                        # If any overlap with existing DRAINs on the node, assume this node was
                        # in a maintenance reservation
                        if any(
                            max(
                                0,
                                (
                                    min(block[0] + block[1], drain_block[0] + drain_block[1]) -
                                    max(block[0], drain_block[0])
                                ).total_seconds()
                            )
                            for block in nid_data[nid]["down_schedule"]
                                if block[2] == "DRAIN"
                        ):
                            nid_data[nid]["down_schedule"].append(list(drain_block))

                    nid_data[nid]["down_schedule"].sort(key=lambda schedule: schedule[0])

                    i_event = 0
                    while i_event < len(nid_data[nid]["down_schedule"]) - 1:
                        event = nid_data[nid]["down_schedule"][i_event]
                        next_event = nid_data[nid]["down_schedule"][i_event + 1]

                        if event[0] + event[1] <= next_event[0]:
                            i_event += 1
                            continue

                        else:
                            nid_data[nid]["down_schedule"][i_event][1] = max(
                                event[1], next_event[0] + next_event[1] - event[0]
                            )
                            nid_data[nid]["down_schedule"][i_event][2] = "DRAIN"
                            nid_data[nid]["down_schedule"][i_event][3] = "blade down maintenance"
                            nid_data[nid]["down_schedule"].pop(i_event + 1)
                            continue

                    nid_data[nid]["down_schedule"].sort(
                        key=lambda schedule: schedule[0], reverse=True
                    )

        # TODO Hide all the below away, loads of mess just to do one fairly simple thing

        # For ARCHER2 it looks like nodes that go down with a reason like
        # "LFP: ..." "RML: ..." "KT: ..." are nodes that were in the maintenance reservation while
        # something like "Not responding" are unplanned down states. Fill hpe_restriclong resv
        # with nodes that go down for these reasons, extend the time they are in the resv such that
        # the resv has at least the same number of nodes as the in the current resv dump

        # Submit some hrs before first down time, add a -num to the end to overrride 8 hrs
        if "-" in hpe_restrictlong_sliding_res:
            hpe_restrictlong_sliding_res, submit_hrs_before = (
                hpe_restrictlong_sliding_res.split("-")
            )
            submit_hrs_before = int(submit_hrs_before)
        else:
            submit_hrs_before = 0

        if (
            hpe_restrictlong_sliding_res == "dynamic" or
            hpe_restrictlong_sliding_res == "dynamic+const" or
            hpe_restrictlong_sliding_res == "dynamic+%extra"
        ):
            target_num_hpe_restrictlong = len(hpe_restrictlong_nids)
            if hpe_restrictlong_sliding_res == "dynamic+const":
                hpe_restrictlong_nids_cpy = set(hpe_restrictlong_nids)
                hpe_restrictlong_nids = defaultdict(lambda: hpe_restrictlong_nids_cpy.copy())
            else:
                hpe_restrictlong_nids = defaultdict(set)

            hpe_restrictlong_nids_nosubmitearly = defaultdict(set)

            for nid, data in nid_data.items():
                if not data["down_schedule"] or nid in hpe_restrictlong_nids:
                    continue

                for down_schedule in data["down_schedule"]:
                    if down_schedule[2] == "DOWN":
                        continue

                    reason_prefix = down_schedule[3].split(" ")[0]

                    if not reason_prefix.isupper():
                        continue

                    # Nodes go down in sets of 4 like this
                    nids = { nid for nid in range(nid - nid % 4, nid - nid % 4 + 4) }

                    first_submit = (
                        down_schedule[0].replace(minute=0, second=0) -
                        timedelta(hours=submit_hrs_before, minutes=5)
                    )
                    for submit_hr in range(
                        int(down_schedule[1] / timedelta(hours=1)) + submit_hrs_before + 2
                    ):
                        hpe_restrictlong_nids[
                            first_submit + timedelta(hours=submit_hr)
                        ].update(nids)
                    for submit_hr in range(int(down_schedule[1] / timedelta(hours=1)) + 1):
                        hpe_restrictlong_nids_nosubmitearly[
                            first_submit + timedelta(hours=submit_hr)
                        ].update(nids)


            if hpe_restrictlong_sliding_res == "dynamic":
                rev_submit_hrs = sorted(hpe_restrictlong_nids, reverse=True)
                for prev_submit_hr, submit_hr in zip(rev_submit_hrs[1:], rev_submit_hrs[:-1]):
                    new_nids = list(
                        hpe_restrictlong_nids[submit_hr] - hpe_restrictlong_nids[prev_submit_hr]
                    )
                    for new_nid in new_nids[
                        :max(
                            (
                                target_num_hpe_restrictlong -
                                len(hpe_restrictlong_nids[prev_submit_hr])
                            ),
                            0
                        )
                    ]:
                        hpe_restrictlong_nids[prev_submit_hr].add(new_nid)

            # Assume that at any given time there are some % extra compute blades in the hpelong
            # reservation than the ones that are actually down for doing work on
            if hpe_restrictlong_sliding_res == "dynamic+%extra":
                rev_submit_hrs = sorted(hpe_restrictlong_nids, reverse=True)
                for prev_submit_hr, submit_hr in zip(rev_submit_hrs[1:], rev_submit_hrs[:-1]):
                    prev_blade_nids = {
                        tuple( nid for nid in range(first_nid, first_nid + 4) )
                        for first_nid in sorted(hpe_restrictlong_nids[prev_submit_hr])[::4]
                    }
                    blade_nids = {
                        tuple( nid for nid in range(first_nid, first_nid + 4) )
                        for first_nid in sorted(hpe_restrictlong_nids[submit_hr])[::4]
                    }
                    new_blade_nids = list(blade_nids - prev_blade_nids)
                    # XXX Currently set to 30% XXX
                    target_blade_nids = int(
                        (len(hpe_restrictlong_nids_nosubmitearly[prev_submit_hr]) / 5) * 1.3 + 1
                    )
                    for blade_nids in new_blade_nids[
                        :max(target_blade_nids - len(prev_blade_nids), 0)
                    ]:
                        hpe_restrictlong_nids[prev_submit_hr].update(blade_nids)

        elif hpe_restrictlong_sliding_res != "": # file path to time - num nodes file
            # Fill with nodes that go down with certain reasons
            hpe_restrictlong_nids = defaultdict(set)
            for nid, data in nid_data.items():
                if not data["down_schedule"] or nid in hpe_restrictlong_nids:
                    continue

                for down_schedule in data["down_schedule"]:
                    if down_schedule[2] == "DOWN":
                        continue

                    reason_prefix = down_schedule[3].split(" ")[0]

                    if not reason_prefix.isupper():
                        continue

                    first_submit = (
                        down_schedule[0].replace(minute=0, second=0) - timedelta(minutes=5)
                    )
                    for submit_hr in range(int(down_schedule[1] / timedelta(hours=1)) + 2):
                        hpe_restrictlong_nids[
                            first_submit + timedelta(hours=submit_hr)
                        ].add(nid)

            # Load and clean actual hpe long num nodes data
            df_num_hpelong = pd.read_csv(
                hpe_restrictlong_sliding_res,  delimiter=' ', lineterminator='\n',
                names=["Time", "NNodes"], encoding="ISO-8859-1"
            )

            time_nnodes = defaultdict(int)
            for _, row in df_num_hpelong.iterrows():
                t = datetime.datetime.strptime(row.Time, "%Y-%m-%dT%H:%M:%S").replace(
                    minute=0, second=0
                )
                time_nnodes[t] = max(int(row.NNodes), time_nnodes[t])

            t_i = min(hpe_restrictlong_nids) + timedelta(minutes=5)
            t_f = max(hpe_restrictlong_nids) + timedelta(minutes=5)
            time_nnodes = {
                time : nnodes
                for time, nnodes in time_nnodes.items()
                    if time <= t_f
            }

            for time in list(time_nnodes):
                later_time = time + timedelta(hours=1)
                while later_time not in time_nnodes and later_time <= t_f:
                    time_nnodes[later_time] = time_nnodes[time]
                    later_time += timedelta(hours=1)

            time_nnodes = {
                time : nnodes
                for time, nnodes in time_nnodes.items()
                    if time >= t_i
            }

            for submit_hr, nids in hpe_restrictlong_nids.items():
                nid_num_from_blade = []

                for nid in nids:
                    blade_nids = { nid for nid in range(nid - nid % 4, nid - nid % 4 + 4) }
                    num_from_blade = len(blade_nids.intersection(nids))

                    if any((nid, num_from_blade) in nid_num_from_blade for nid in blade_nids):
                        continue

                    nid_num_from_blade.append((nid, num_from_blade))

                nid_num_from_blade.sort(key=lambda nid_num: nid_num[1])

                # If needed, add extra nodes to reservation from blades with the most nodes down at
                # this hour
                while (
                    nid_num_from_blade and
                    len(nids) < time_nnodes[submit_hr + timedelta(minutes=5)]
                ):
                    nid, _ = nid_num_from_blade.pop()

                    nids.update({ nid for nid in range(nid - nid % 4, nid - nid % 4 + 4) })

                nid_num_from_blade.sort(key=lambda nid_num: nid_num[1], reverse=True)

                # If needed, remove extra nodes from reservation from blades with the leasts nodes
                # down at this hour
                while (
                    nid_num_from_blade and
                    len(nids) > time_nnodes[submit_hr + timedelta(minutes=5)]
                ):
                    nid, _ = nid_num_from_blade.pop()
                    nids -= {  nid for nid in range(nid - nid % 4, nid - nid % 4 + 4) }

            # Now put nodes into the reservation at earlier times to make up any remaining
            # difference
            rev_submit_hrs = sorted(hpe_restrictlong_nids, reverse=True)
            for prev_submit_hr, submit_hr in zip(rev_submit_hrs[1:], rev_submit_hrs[:-1]):
                new_nids = list(
                    hpe_restrictlong_nids[submit_hr] - hpe_restrictlong_nids[prev_submit_hr]
                )
                for new_nid in new_nids[
                    :max(
                        (
                            time_nnodes[prev_submit_hr + timedelta(minutes=5)] -
                            len(hpe_restrictlong_nids[prev_submit_hr])
                        ),
                        0
                    )
                ]:
                    hpe_restrictlong_nids[prev_submit_hr].add(new_nid)

            # Now keep nodes into the reservation until later times to make up any remaining
            # difference
            submit_hrs = sorted(hpe_restrictlong_nids)
            for next_submit_hr, submit_hr in zip(submit_hrs[1:], submit_hrs[:-1]):
                new_nids = list(
                    hpe_restrictlong_nids[submit_hr] - hpe_restrictlong_nids[next_submit_hr]
                )
                for new_nid in new_nids[
                    :max(
                        (
                            time_nnodes[next_submit_hr + timedelta(minutes=5)] -
                            len(hpe_restrictlong_nids[next_submit_hr])
                        ),
                        0
                    )
                ]:
                    hpe_restrictlong_nids[next_submit_hr].add(new_nid)

            # Fill any remaining differences with nodes from the first hours
            for submit_hr in submit_hrs:
                target_nids = time_nnodes[submit_hr + timedelta(minutes=5)]
                for submit_hr_start in submit_hrs:
                    if len(hpe_restrictlong_nids[submit_hr]) == target_nids:
                        break

                    for nid in hpe_restrictlong_nids[submit_hr_start]:
                        if len(hpe_restrictlong_nids[submit_hr]) == target_nids:
                            break

                        hpe_restrictlong_nids[submit_hr].add(nid)

        # XXX ARCHER2 specific - can't be bothered to implement REPLACE_DOWN on reservations so
        # just fill with nodes that don't go down at any point
        if len(df_resv.loc[(df_resv.RESV_NAME == "shortqos")]):
            shortqos_nids_to_replace = {
                nid
                for nid, data in nid_data.items()
                    if (
                        any(resv[2] == "shortqos" for resv in data["resv_schedule"]) and
                        not data["down_schedule"]
                    )
            }
            never_down_nids = {
                nid
                for nid, data in nid_data.items()
                    if (
                        not data["down_schedule"] and
                        not data["resv_schedule"] and
                        not any(partition == "highmem" for partition in data["partitions"])
                    )
            }
            for i_shortqos_nid, shortqos_nid in enumerate(shortqos_nids_to_replace):
                if not never_down_nids:
                    break
                never_down_nid = never_down_nids.pop()
                nid_data[never_down_nid]["resv_schedule"] = nid_data[shortqos_nid]["resv_schedule"]
                nid_data[shortqos_nid]["resv_schedule"] = []
            print(
                "Replaced {} / {} shortqos nodes with nodes that never go down".format(
                    i_shortqos_nid + 1, len(shortqos_nids_to_replace)
                )
            )

        return nid_data, partition_data, valid_resv, hpe_restrictlong_nids

    def get_qos(self):
        df_qos = pd.read_csv(
            self.qos_dump,  delimiter='|', lineterminator='\n', header=0, encoding="ISO-8859-1"
        )

        qos_data = {}

        for _, row in df_qos.iterrows():
            qos_data[row.Name] = {
                "name" : row.Name,
                "prio" : int(row.Priority),
                "GrpTRES" : (
                    None
                    if pd.isna(row.GrpTRES) or "node=" not in row.GrpTRES
                    else int(row.GrpTRES.split("node=")[1].split(",")[0])
                ),
                "GrpJobs" : None if pd.isna(row.GrpJobs) else int(row.GrpJobs),
                "GrpSubmit" : None if pd.isna(row.GrpSubmit) else int(row.GrpSubmit),
                "MaxTRESPU" : (
                    None
                    if pd.isna(row.MaxTRESPU) or "node=" not in row.MaxTRESPU
                    else int(row.MaxTRESPU.split("node=")[1].split(",")[0])
                ),
                "MaxJobsPU" : None if pd.isna(row.MaxJobsPU) else int(row.MaxJobsPU),
                "MaxJobs" : None if pd.isna(row.MaxJobs) else int(row.MaxJobs),
                "MaxSubmitPU" : None if pd.isna(row.MaxSubmitPU) else int(row.MaxSubmitPU),
                "MaxSubmit" : None if pd.isna(row.MaxSubmit) else int(row.MaxSubmit)
            }

        max_qos_prio = max(data["prio"] for data in qos_data.values())
        if max_qos_prio != 0:
            for data in qos_data.values():
                data["prio"] /= max_qos_prio

        return qos_data

    def get_cleaned_job_df(self, considered_partitions, def_power_per_node):
        df_jobs = pd.read_csv(
            self.job_dump, delimiter='|', lineterminator='\n', header=0, encoding="ISO-8859-1",
            usecols=[
                "JobID", "Start", "End", "Submit", "Elapsed", "ConsumedEnergyRaw", "AllocNodes",
                "Timelimit", "ReqCPUS", "ReqNodes", "Group", "QOS", "ReqMem", "User", "Account",
                "Partition", "SubmitLine", "JobName", "Reason", "State"
            ],
        )
        df_jobs = df_jobs.loc[
            (df_jobs.Start != "Unknown") & (df_jobs.Start.notna()) & (df_jobs.End != "Unknown") &
            (df_jobs.End.notna()) & (df_jobs.Partition.isin(considered_partitions)) &
            (df_jobs.Timelimit.notna()) & (df_jobs.ReqNodes != "0") & (df_jobs.ReqNodes != 0) &
            (
                ((df_jobs.AllocNodes != "0") & (df_jobs.AllocNodes != 0)) |
                ((df_jobs.State.str.contains("CANCELLED")) & (df_jobs.Start == df_jobs.End))
            )
        ]

        df_jobs.Submit = pd.to_datetime(df_jobs.Submit, format="%Y-%m-%dT%H:%M:%S")
        df_jobs.Start = pd.to_datetime(df_jobs.Start, format="%Y-%m-%dT%H:%M:%S")
        df_jobs.End = pd.to_datetime(df_jobs.End, format="%Y-%m-%dT%H:%M:%S")
        df_jobs.Elapsed = df_jobs.End - df_jobs.Start
        df_jobs.Timelimit = df_jobs.Timelimit.apply(lambda row: timelimit_str_to_timedelta(row))

        convert_to_raw(df_jobs, "AllocNodes")
        convert_to_raw(df_jobs, "ReqNodes")

        df_jobs["Nodes"] = df_jobs.apply(
            lambda row: row.ReqNodes if row.AllocNodes == 0 else row.AllocNodes, axis=1
        )

        num_bad = len(
            df_jobs.loc[
                (~df_jobs.State.str.contains("CANCELLED")) &
                (
                    (df_jobs.ConsumedEnergyRaw.isna()) | (df_jobs.ConsumedEnergyRaw == 0.0) |
                    (df_jobs.ConsumedEnergyRaw == "")
                )
            ]
        )
        if num_bad / len(df_jobs) < 0.25:
            df_jobs.ConsumedEnergyRaw = df_jobs.apply(
                lambda row: (
                    float(row.ConsumedEnergyRaw)
                    if (
                        row.ConsumedEnergyRaw == row.ConsumedEnergyRaw and
                        row.ConsumedEnergyRaw != 0.0 and
                        row.ConsumedEnergyRaw != ""
                    )
                    else float(def_power_per_node * row.AllocNodes * row.Elapsed.total_seconds())
                ),
                axis=1
            )
        else:
            print(
                "!!!More than 25% of jobs do not have a valid ConsumedEnergy,"
                "setting all ConsumedEnergies to zero!!!"
            )
            df_jobs = df_jobs.assign(ConsumedEnergyRaw=0.0)
            def_power_per_node = 0.0

        df_jobs["Power"] = df_jobs.apply(
            lambda row: (
                float(row.ConsumedEnergyRaw) / row.Elapsed.total_seconds()
                if row.Elapsed.total_seconds() != 0
                else 0.0
            ),
            axis=1
        )
        num_bad += len(df_jobs.loc[(df_jobs.Power >= 10000000)])
        for i, anomalous_row in df_jobs.loc[(df_jobs.Power >= 10000000)].iterrows():
            df_jobs.at[i, "Power"] = def_power_per_node * df_jobs.at[i, "AllocNodes"]
        if def_power_per_node:
            print(
                "Set {} jobs with bad or missing ConsumedEnergyRaw to mean power per node {}W" \
                "".format(num_bad, def_power_per_node)
            )

        df_jobs["TruePowerPerNode"] = df_jobs.apply(
            lambda row: float(row.Power) / float(row.AllocNodes) if row.AllocNodes != 0 else 0.0,
            axis=1
        )

        df_jobs["DependencyArg"] = df_jobs.SubmitLine.apply(
            lambda row: get_sbatch_cli_arg(row, long="--dependency", short="-d")
        )
        df_jobs["ReservationArg"] = df_jobs.SubmitLine.apply(
            lambda row: get_sbatch_cli_arg(row, long="--reservation")
        )
        df_jobs["BeginArg"] = df_jobs.SubmitLine.apply(
            lambda row: get_sbatch_cli_arg(row, long="--begin", short="-b")
        )
        df_jobs["NodelistArg"] = df_jobs.SubmitLine.apply(
            lambda row: get_sbatch_cli_arg(row, long="--nodelist", short="-w")
        )
        df_jobs["ExcludeArg"] = df_jobs.SubmitLine.apply(
            lambda row: get_sbatch_cli_arg(row, long="--exclude", short="-x")
        )

        # Some error in slurm accounting, can correct for case of one other user in account
        num_broken, num_fixed = len(df_jobs.loc[(df_jobs.User == "00:00:00")]), 0
        for i, anomalous_row in df_jobs.loc[(df_jobs.User == "00:00:00")].iterrows():
            acc_users = df_jobs.loc[(df_jobs.Account == anomalous_row.Account)].User.unique()
            if len(acc_users) == 2:
                num_fixed += 1
                df_jobs.at[i, "User"] = (
                    acc_users[1] if acc_users[0] == "00:00:00" else acc_users[0]
                )
        print("Corrected {} of {} users with name 00:00:00".format(num_fixed, num_broken))

        df_jobs["Cancelled"] = df_jobs.apply(
            lambda row: None if row.AllocNodes != 0 else row.End - row.Submit, axis=1
        )

        df_jobs.JobID = df_jobs.JobID.apply(lambda row: str(row))
        print("{} heterogeneous JobIDs converted to regular JobIDs".format(
            len(df_jobs.loc[(df_jobs.JobID.str.contains("+", regex=False))])
        ))
        df_jobs.JobID = df_jobs.JobID.apply(
            lambda row: str(int(row.split("+")[0]) + int(row.split("+")[1])) if "+" in row else row
        )

        # Function time :(
        df_jobs.JobID = df_jobs.JobID.apply(
            lambda row: (
                [row.replace("[", "").replace("]", "")]
                if( "-" not in row and "," not in row) or ":" in row # Get out im not doing array steps
                else [
                    row.split("[")[0]  + str(num)#]
                    for num in [
                        index
                        for entry in row.split("_[")[1].strip("]").split("%")[0].split(",")
                            for index in range(
                                int(entry.split("-")[0]),
                                (
                                    int(entry.split("-")[1])
                                    if len(entry.split("-")) == 2
                                    else int(entry.split("-")[0]) + 1
                                )
                            )
                    ]
                ]
            )
        )
        num_jobs_with_arrs = len(df_jobs)
        df_jobs = df_jobs.explode("JobID")
        print(
            "{} cancelled job arrays converted to individual job entries".format(
                len(df_jobs) - num_jobs_with_arrs
            )
        )

        df_jobs_orig_len = len(df_jobs)
        df_jobs = df_jobs[~df_jobs.duplicated(subset=["JobID", "Submit"], keep="first")]
        print(
            "{} duplicate (JobID,Submit) present, deleting".format(df_jobs_orig_len - len(df_jobs))
        )

        print("{} Jobs in workload trace".format(len(df_jobs)))
        print(
            "{} Jobs in workload trace cancelled before running".format(
                len(df_jobs.loc[(df_jobs.AllocNodes == 0)])
            )
        )

        return df_jobs


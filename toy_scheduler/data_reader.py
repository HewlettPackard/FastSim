from collections import defaultdict
from datetime import timedelta

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

    def get_nodes_partitions(self, considered_partitions, hpe_restrictlong_sliding_res, max_sim_t):
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

        # For ARCHER2 it looks like nodes that go down with a reason like
        # "LFP: ..." "RML: ..." "KT: ..." are nodes that were in the maintenance reservation while
        # something like "Not responding" are unplanned down states. Fill hpe_restriclong resv
        # with nodes that go down for these reasons, extend the time they are in the resv such that
        # the resv has at least the same number of nodes as the in the current resv dump
        if hpe_restrictlong_sliding_res == "dynamic":
            target_num_hpe_restrictlong = len(hpe_restrictlong_nids)
            hpe_restrictlong_nids = defaultdict(set)

            for nid, data in nid_data.items():
                if not data["down_schedule"] or nid in hpe_restrictlong_nids:
                    continue

                for down_schedule in data["down_schedule"]:
                    if ": " not in down_schedule[3]:
                        continue

                    reason_prefix = down_schedule[3].split(": ")[0]

                    if not reason_prefix.isupper():
                        continue
                    
                    # Nodes go down in sets of 4 like this
                    nids = { nid for nid in range(nid - nid % 4, nid - nid % 4 + 4) }

                    first_submit = (
                        down_schedule[0].replace(minute=0, second=0) -
                        timedelta(hours=48, minutes=5)
                    )
                    # Submit 8hrs before first down time
                    for submit_hr in range(int(down_schedule[1] / timedelta(hours=1)) + 10):
                        hpe_restrictlong_nids[first_submit + timedelta(hours=submit_hr)].update(nids)

            rev_submit_hrs = sorted(hpe_restrictlong_nids, reverse=True)
            for prev_submit_hr, submit_hr in zip(rev_submit_hrs[1:], rev_submit_hrs[:-1]):
                new_nids = list(hpe_restrictlong_nids[submit_hr] - hpe_restrictlong_nids[prev_submit_hr])
                for new_nid in new_nids[
                    :max(
                        target_num_hpe_restrictlong - len(hpe_restrictlong_nids[prev_submit_hr]),
                        0
                    )
                ]:
                    hpe_restrictlong_nids[prev_submit_hr].add(new_nid)

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
            (df_jobs.End.notna()) & (df_jobs.AllocNodes != "0") & (df_jobs.AllocNodes != 0) &
            (df_jobs.Partition.isin(considered_partitions)) & (df_jobs.Timelimit.notna())
        ]

        df_jobs.Submit = pd.to_datetime(df_jobs.Submit, format="%Y-%m-%dT%H:%M:%S")
        df_jobs.Start = pd.to_datetime(df_jobs.Start, format="%Y-%m-%dT%H:%M:%S")
        df_jobs.End = pd.to_datetime(df_jobs.End, format="%Y-%m-%dT%H:%M:%S")
        df_jobs.Elapsed = df_jobs.End - df_jobs.Start
        df_jobs.Timelimit = df_jobs.Timelimit.apply(lambda row: timelimit_str_to_timedelta(row))

        print(
            "{} duplicate ids present".format(
                len(df_jobs) -  len(df_jobs[~df_jobs.duplicated(subset="JobID", keep="first")])
            )
        )

        convert_to_raw(df_jobs, "AllocNodes")

        num_bad = len(
            df_jobs.loc[
                (df_jobs.ConsumedEnergyRaw.isna()) | (df_jobs.ConsumedEnergyRaw == 0.0) |
                (df_jobs.ConsumedEnergyRaw == "")
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
                (
                    float(row.ConsumedEnergyRaw) / row.Elapsed.total_seconds()
                ) if row.Elapsed.total_seconds() != 0 else 0.0
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
            lambda row: float(row.Power) / float(row.AllocNodes), axis=1
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

        df_jobs.JobID = df_jobs.JobID.apply(lambda row: str(row))
        print("{} heterogeneous JobIDs converted to regular JobIDs".format(
            len(df_jobs.loc[(df_jobs.JobID.str.contains("+", regex=False))])
        ))
        df_jobs.JobID = df_jobs.JobID.apply(
            lambda row: str(int(row.split("+")[0]) + int(row.split("+")[1])) if "+" in row else row
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

        print("{} Jobs in workload trace".format(len(df_jobs)))

        return df_jobs


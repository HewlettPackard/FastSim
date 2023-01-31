import datetime
from datetime import timedelta

import pandas as pd


class Queue():
    def __init__(self, job_data, priority_sorter, valid_reservations=[]):
        self.priority_sorter = priority_sorter

        # hardcoded for now
        # Don't have a proper way to simulate reservations as I can't see a history of deleted ones
        self.qoss = {
            "normal" : QOS("normal", 0, -1, -1, -1, -1, -1),
            "reservation" : QOS("reservation", 0.1, -1, -1, -1, -1, -1),
            "taskfarm" : QOS("taskfarm", 0.1, -1, 128, 512, 32, 128),
            "standard" : QOS("standard", 0.1, -1, -1, 1024, 16, 64),
            "short" : QOS("short", 0.1, -1, -1, 32, 4, 16),
            "long" : QOS("long", 0.1, 2048, -1, 512, 16, 16),
            "largescale" : QOS("largescale", 0.1, -1, -1, -1, 1, 8),
            "highmem" : QOS("highmem", 0.1, -1, -1, 256, 16, 16),
            "lowpriority" : QOS("lowpriority", 0.0002, -1, -1, 2048, 16, 16)
        }

        df_jobs = _prep_job_data(df_jobs)
        self.time = df_jobs.Start.min()
        self.all_jobs = [
            Job(
                job_row.JobID, job_row.Submit, job_row.AllocNodes, job_row.Elapsed,
                job_row.Timelimit, job_row.PowerPerNode, job_row.TruePowerPerNode, job_row.Start,
                job_row.User, job_row.Account, self.qoss[job_row.QOS], job_row.Partition,
                job_row.DependencyArg, job_row.JobName, job_row.Reason, job_row.ReservationArg,
                job_row.BeginArg
            ) for _, job_row in df_jobs.sort_values("Submit").iterrows()
        ]
        self._verify_dependencies()
        self.queue = []

        self.waiting_dependency = []

        self.qos_held = { qos : [] for qos in self.qoss.values() }
        self.qos_submit_held = { qos : [] for qos in self.qoss.values() }

        self._verify_reservations(valid_reservations)
        self.reservations = defaultdict(list)

    def _get_sbatch_cli_arg(submit_line, long="", short=""):
        words = submit_line.split(" ")
        dep_arg = None
        for i_last_word, word in enumerate(words[1:]):
            # Batch script or executable marks end of options
            if word[0] != "-" and (words[i_last_word][0] != "-" or "=" in words[i_last_word]):
                break
            if long:
                if long + "=" in word:
                    dep_arg = word.split("--dependency=")[1]
                    break
                if word == long:
                    dep_arg = words[i_last_word + 2]
                    break
            if short:
                if word == short:
                    dep_arg = words[i_last_word + 2]
                    break

        return dep_arg

    def _timelimit_str_to_timedelta(t_str):
        days, hrs = 0, 0
        try:
            if "-" in t_str:
                days = int(t_str.split("-")[0])
                t_str = t_str.split("-")[1]
        except:
            print(t_str)

        if t_str.count(":") == 1 and t_str.count("."): # MM:SS.SS
            mins, secs = t_str.split(":")
            mins = int(mins)
            secs = float(secs)
        elif t_str.count(":") == 2: ## HH:MM:SS (SS has no decimal place for these ones)
            hrs, mins, secs = map(int, t_str.split(":"))
        else:
            raise NotImplementedError("Bruh")

        return datetime.timedelta(days=days, hours=hrs, minutes=mins, seconds=secs)

    def _convert_to_raw(df, cols):
        df[cols] = df[cols].astype(str)
        df[cols] = df[cols].replace(
            { "K" : "e+03", "M" : "e+06", "G" : "e+09", "T" : "e+12" }, regex=True
        ).astype(float).astype(int)
        return df

    def _prep_job_data(data_path):
        df_jobs = pd.read_csv(
            data_path, delimiter='|', lineterminator='\n', header=0,
            usecols=[
                "JobID", "Start", "End", "Submit", "Elapsed", "ConsumedEnergyRaw", "AllocNodes",
                "Timelimit", "ReqCPUS", "ReqNodes", "Group", "QOS", "ReqMem", "User", "Account",
                "Partition", "SubmitLine", "JobName", "Reason", "State"
            ]
        )
        df_jobs = df_jobs.loc[
            (df.Start != "Unknown") & (df.Start.notna()) & (df.End != "Unknown") &
            (df.End.notna()) & (df.AllocNodes != "0") & (df.AllocNodes != 0) & # Just to be safe
            (df.Partition != "serial")
        ]

        df_jobs.Submit = pd.to_datetime(df_jobs.Submit, format="%Y-%m-%dT%H:%M:%S")
        df_jobs.Start = pd.to_datetime(df_power.Start, format="%Y-%m-%dT%H:%M:%S")
        df_jobs.End = pd.to_datetime(df_power.End, format="%Y-%m-%dT%H:%M:%S")
        df_jobs.Elapsed = df_jobs.End - df_jobs.Start
        df_jobs.Timelimit = df_jobs.Timelimit.apply(lambda row: _timelimit_str_to_timedelta(row))

        df_jobs = df_jobs[~df_jobs.duplicated(subset="JobID", keep="first")]

        _convert_to_raw(df_jobs, "AllocNodes")

        num_bad = len(
            df_jobs.loc[
                (df_jobs.ConsumedEnergyRaw.isna()) | (df_jobs.ConsumedEnergyRaw == 0.0) |
                (df_jobs.ConsumedEnergyRaw == "")
            ]
        )
        df_jobs.ConsumedEnergyRaw = df_jobs.apply(
            lambda row: (
                float(row.ConsumedEnergyRaw) if (
                    row.ConsumedEnergyRaw == row.ConsumedEnergyRaw and # ie. not nan
                    row.ConsumedEnergyRaw != 0.0 and row.ConsumedEnergyRaw != ""
                ) else float(550 * row.AllocNodes * row.DeltaT)
            ),
            axis=1
        )
        df_jobs["Power"] = df_jobs.apply(
            lambda row: (
                float(row.ConsumedEnergyRaw) / float(row.DeltaT) if row.DeltaT != 0 else 0.0
            ),
            axis=1
        )
        num_bad += len(df_jobs.loc[(df_jobs.Power >= 10000000)])
        for i, anomalous_row in df_jobs.loc[(df_jobs.Power >= 10000000)].iterrows():
            df_jobs.at[i, "Power"] = 550 * df_jobs.at[i, "AllocNodes"]
        print("Salvaged {} jobs with bad ConsumedEnergyRaw".format(num_bad))

        df_jobs["TruePowerPerNode"] = df_jobs.apply(
            lambda row: float(row.Power) / float(row.AllocNodes), axis=1
        )

        df_jobs["DependencyArg"] = df_jobs.SubmitLine.apply(
            lambda row: _get_sbatch_cli_arg(row, long="--dependency", short="-d")
        )
        df_jobs["ReservationArg"] = df_jobs.SubmitLine.apply(
            lambda row: _get_sbatch_cli_arg(row, long="--reservation")
        )
        df_jobs["BeginArg"] = df_jobs.SubmitLine.apply(
            lambda row: _get_sbatch_cli_arg(row, long="--begin", short="-b")
        )

        print("{} heterogeneous JobIDs converted to regular JobIDs".format(
            len(df_jobs.loc[(df_jobs.JobID.str.contains("+", regex=False))])
        ))
        df_jobs.JobID = df_jobs.JobID.apply(
            lambda row: str(int(row.split("+")[0]) + int(row.split("+")[1])) if "+" in row else row
        )

        # Some error in slurm accounting, can correct for case of one other user in account
        num_fixed = 0
        for i, anomalous_row in df_jobs.loc[(df_jobs.User == "00:00:00")].iterrows():
            acc_users = df_jobs.loc[(df_jobs.Account == anomalous_row.Account)].User.unique()
            if len(acc_users) == 2:
                num_fixed += 1
                df_jobs.at[i, "User"] = (
                    acc_users[1] if acc_users[0] == "00:00:00" else acc_users[0]
                )
        print("Corrected {} of {} users with name 00:00:00".format(
            num_fixed, len(df_jobs.loc[(df_jobs.User == "00:00:00")])
        ))

        return df_jobs

    def _verify_reservations(self, valid_reservations):
        removed_res, removed_res_cnt = set(), 0
        for job in self.all_jobs:
            if not job.reservation:
                # short qos is reservationrequired
                if job.qos.name == "short":
                    job.reservation = "shortqos"
                continue

            if job.reservation not in valid_reservations:
                removed_res.add(job.reservation)
                removed_res_cnt += 1
                job.reservation = ""
                job.ignore_in_eval = True

        print(
            "Missing reservation records for {} resulting in ignoring reservations for" \
            "{} jobs".format(removed_res, removed_res_cnt)
        )

    def _verify_dependencies(self):
        removed_dep_cnt, ignored_dep_cnt = 0, 0
        all_ids = { job.id for job in self.all_jobs }
        for job in self.all_jobs:
            if not job.dependency:
                # Dependency hidden in batch file or just removed
                if job.reason == "Dependency":
                    job.ignore_in_eval = True
                    ignored_dep_cnt += 1
                continue

            for dep_type, ids in job.dependency.conditions.items():
                intersection = ids.intersection(all_ids)
                job.dependency.conditions[dep_type] = intersection

                # Job arrays need to be expanded into individual JobIDs
                for unmatched_id in ids.difference(intersection):
                    array_ids = {
                        id for id in all_ids if re.match("{}_[0-9]".format(unmatched_id), id)
                    }
                    if array_ids:
                        if job.dependency.delimiter == "?":
                            raise NotImplemetedError(
                                "Not implemented expanding job array ids for OR dependencies"
                            )
                        job.dependency.conditions[dep_type].update(array_ids)

            job.dependency.conditions_met = not any(job.dependency.conditions.values())
            if job.dependency.conditions_met and not job.dependency.singleton:
                removed_dep_cnt += 1
                job.dependency = None
                if job.reason == "Dependency":
                    job.ignore_in_eval = True
                    ignored_dep_cnt += 1

        print(
            (
                "Removed {} dependencies that cannot be satisfied from workload trace\n".format(
                    removed_dep_cnt
                )
            ) +
            (
                "Ignored {} in evaulation due to dependency not being in SubmitLine or missing " \
                "from workload trace".format(
                    ignored_dep_cnt
                )
            )
        )

    def _check_dependencies(self, finished_jobs_step, submitted_jobs_step, running_jobs):
        released = []
        for i_job, job in enumerate(self.waiting_dependency):
            job.dependency.update_finished(finished_jobs_step)
            job.dependency.update_submitted(submitted_jobs_step)

            if not job.dependency.can_release(self.queue, running_jobs):
                continue

            released_job = self.waiting_dependency[i_job]
            if released_job.qos.hold_job(released_job):
                self.qos_held[released_job.qos].append(released_job.launch(self.time))
            elif released_job.reservation:
                self.reservations[released_job.reservation].append(
                    released_job.launch(self.time).launch_into_priority()
                )
            else:
                self.queue.append(released_job.launch(self.time).launch_into_priority())
            released.append(i_job)

        for i_job in sorted(released, reverse=True):
            self.waiting_dependency.pop(i_job)

    def _check_qos_holds(self, running_jobs, job_history):
        for qos in self.qos_submit_held.keys():
            if not self.qos_submit_held[qos]:
                continue

            released = []
            for i_job, job in enumerate(self.qos_submit_held[qos]):
                if job.qos.hold_job_submit_usr(job):
                    continue

                # Pretend the job is now resubmitted
                job.submit_job(self.time)
                released.append(i_job)

                if job.dependency:
                    job.dependency.update_finished(job_history)
                    job.dependency.update_submitted(running_jobs + job_history)
                    if not job.dependency.can_release(self.queue, running_jobs):
                        self.waiting_dependency.append(job)
                        continue

                if job.qos.hold_job(job):
                    self.qos_held[job.qos].append(job.launch(self.time))
                    continue

                if job.reservation:
                    self.reservations[job.reservation].append(job.launch_into_priority())
                else:
                    self.queue.append(job.launch_into_priority())

            for i_job in sorted(released, reverse=True):
                self.qos_submit_held[qos].pop(i_job)

        for qos in self.qos_held.keys():
            # Dont need to check if we know nothing will be released
            if (
                not self.qos_held[qos] or
                not qos.job_quota_remaining or
                not qos.node_quota_remaining
            ):
                continue
            self.qos_held[qos] = self.priority_sorter.sort(self.qos_held[qos], self.time)

            released = []
            for i_job, job in enumerate(self.qos_held[qos]):
                if job.qos.hold_job_grp(job):
                    break

                # Lower priority jobs can be released if the job infront is only held by a per user
                # limit
                if job.qos.hold_job_usr(job):
                    continue

                if job.reservation:
                    self.reservations[job.reservation].append(job.launch_into_priority())
                else:
                    self.queue.append(job.launch_into_priority())

                released.append(i_job)

            for i_job in sorted(released, reverse=True):
                self.qos_held[qos].pop(i_job)

    def step(self, time, finished_jobs_step, submitted_jobs_step, running_jobs, job_history):
        self.time = time

        self._check_dependencies(finished_jobs_step, submitted_jobs_step, running_jobs)
        self._check_qos_holds(running_jobs, job_history)

        if self.time < self.next_newjob():
            # Still need to sort as there may be new jobs from dependency and qos releases
            self.queue[retained:] = self.priority_sorter.sort(self.queue[retained:], self.time)
            return

        try:
            while self.all_jobs[0].submit <= self.time:
                new_job = self.all_jobs.pop(0)

                if new_job.qos.hold_job_submit_usr(new_job):
                    self.qos_submit_held[new_job.qos].append(new_job)
                    continue

                new_job.submit_job()

                if new_job.dependency:
                    new_job.dependency.update_finished(job_history)
                    new_job.dependency.update_submitted(running_jobs + job_history)
                    if not new_job.dependency.can_release(self.queue, running_jobs):
                        self.waiting_dependency.append(new_job)
                        continue

                if new_job.qos.hold_job(new_job):
                    self.qos_held[new_job.qos].append(new_job.launch(self.time))
                    continue

                if new_job.reservation:
                    self.reservations[new_job.reservation].append(new_job.launch(self.time))
                    continue

                self.queue.append(new_job.launch(self.time).launch_into_priority())

        except IndexError: # No more new jobs
            pass

        self.queue[retained:] = self.priority_sorter.sort(self.queue[retained:], self.time)
        for reservation in list(self.reservations.keys()):
            self.reservations[reservation] = (
                self.priority_sorter.sort(self.reservations[reservation], self.time)
            )

    def next_newjob(self):
        try:
            return self.all_jobs[0].submit
        except IndexError:
            return datetime.max


class QOS():
    def __init__(self, name, priority, grp_nodes, grp_jobs, usr_nodes, usr_jobs, usr_submit):
        self.name = name
        self.priority = priority

        # TODO This is dumb, implement in a way that means in these cases the quota is just not tracked
        grp_jobs = 1000000000000 if grp_jobs == -1 else grp_jobs
        grp_nodes = 1000000000000 if grp_nodes == -1 else grp_nodes
        usr_jobs = 1000000000000 if usr_jobs == -1 else usr_jobs
        usr_nodes = 1000000000000 if usr_nodes == -1 else usr_nodes
        usr_submit = 1000000000000 if usr_submit == -1 else usr_submit

        self.grp_quotas = []
        self.job_quota_remaining = grp_jobs
        self.node_quota_remaining = grp_nodes

        self.usr_job_quota_remaining = defaultdict(lambda: usr_jobs)
        self.usr_node_quota_remaining = defaultdict(lambda: usr_nodes)

        self.usr_submit_quota_remaining = defaultdict(lambda: usr_submit)

    def job_submitted(self, user):
        self.usr_submit_quota_remaining[user] -= 1

    def job_launched(self, user, nodes):
        self.job_quota_remaining -= 1
        self.node_quota_remaining -= nodes
        self.usr_job_quota_remaining[user] -= 1
        self.usr_node_quota_remaining[user] -= nodes

    def job_ended(self, user, nodes):
        self.job_quota_remaining += 1
        self.node_quota_remaining += nodes
        self.usr_job_quota_remaining[user] += 1
        self.usr_node_quota_remaining[user] += nodes
        self.usr_submit_quota_remaining[user] += 1
        self.possible_to_release = True

    def hold_job(self, job):
        return (
            not self.job_quota_remaining or
            job.nodes > self.node_quota_remaining or
            not self.usr_job_quota_remaining[job.user] or
            job.nodes > self.usr_node_quota_remaining[job.user]
        )

    def hold_job_grp(self, job):
        return not self.job_quota_remaining or job.nodes > self.node_quota_remaining

    def hold_job_usr(self, job):
        return (
            not self.usr_job_quota_remaining[job.user] or
            job.nodes > self.usr_node_quota_remaining[job.user]
        )

    def hold_job_submit_usr(self, job):
        return not self.usr_submit_quota_remaining[job.user]


class Job():
    def __init__(
        self, id, submit : datetime, nodes, runtime : timedelta, reqtime: timedelta, node_power,
        true_node_power, true_job_start, user, account, qos, partition, dependency_arg, name,
        reason, reservation_arg, begin_arg
    ):
        self.id = id
        self.nodes = nodes
        self.runtime = runtime
        self.reqtime = reqtime
        self.node_power = node_power
        self.true_node_power = true_node_power
        self.true_submit = submit
        self.submit = submit
        self.true_job_start = true_job_start
        self.user = user
        self.account = account
        self.qos = qos
        self.partition = partition
        self.name = name
        self.dependency = Dependency(dependency_arg, user, name) if dependency_arg else None
        self.reservation = reservation_arg

        # Some features are not relevant for scheduluing (AssocMaxCpuMinutesPerJobLimit means for
        # archer that the user hasnt been allocated time yet, reservations, jobs held by user, ...)
        # and some I cant implemented with available data (JobArrayTaskLimit is usually specified
        # in batch script). Want to have these jobs in simulation but don't want to include them in
        # evaluation stage

        self.reason = reason
        self.ignore_in_eval = (
            reason in [
                "AssocMaxCpuMinutesPerJobLimit", "ReqNodeNotAvail", "BeginTime", "JobHeldUser",
                "DependencyNeverSatisfied", "JobArrayTaskLimit"
            ] or
            begin_arg
        )

        self.launch_time = submit
        self.start = None
        self.end = None
        self.assigned_nodes = []

    def submit_job(self, time=None):
        if time:
            self.submit = time
        self.qos.job_submitted(self.user)
        return self

    def start_job(self, time : datetime):
        self.start = time
        self.end = time + self.runtime
        self.endlimit = time + self.reqtime
        return self

    def assign_node(self, node):
        node.set_busy()
        self.assigned_nodes.append(node)
        node.running_job = self
        if len(self.assigned_nodes) >= self.nodes:
            return True
        return False

    def end_job(self):
        for node in self.assigned_nodes:
            node.set_free()
            node.running_job = None
        self.qos.job_ended(self.user, self.nodes)
        return self

    # When job starts accruing age
    def launch(self, time):
        self.launch_time = time
        return self

    # When the QOS recognises the job as 'running' for resource limits accounting
    def launch_into_priority(self):
        self.qos.job_launched(self.user, self.nodes)
        return self


class Dependency():
    def __init__(self, dependency_args, user, name):
        self.job_user_name = (user, name)
        self.delimiter = "?" if "?" in dependency_args else ","

        self.conditions = {}
        self.singleton = False
        for condition in dependency_args.split(self.delimiter):
            if condition == "singleton":
                self.singleton = True
                continue

            dep_type = condition.split(":")[0]
            # NOTE after can can take a +time after job_id, just going to ignore these for now
            if "+" in condition:
                print(condition)
            job_ids = { job_id.split("+")[0] for job_id in condition.split(":")[1:] }
            # Jobs in trace all ran so can assume these conditions are met and treat all the same
            if dep_type == "afterok" or dep_type == "afternotok" or dep_type == "afterany":
                self.conditions["afterany"] = job_ids
                continue

            if dep_type == "after":
                self.conditions[dep_type] = job_ids
                continue

            raise NotImplementedError("Unrecognised dep_type {}".format(dep_type))

        self.submitted_relevant = "after" in self.conditions.keys()
        self.finished_relevant = "afterany" in self.conditions.keys()

        self.conditions_met = not any(self.conditions.values())

    def update_finished(self, jobs):
        # return
        if not self.finished_relevant or self.conditions_met:
            return

        for job in jobs:
            if job.id in self.conditions["afterany"]:
                self.conditions["afterany"].remove(job.id)
                if self.delimiter == "?":
                    self.conditions_met = True
                    return

    def update_submitted(self, jobs):
        # return
        if not self.submitted_relevant or self.conditions_met:
            return

        for job in jobs:
            if job.id in self.conditions["after"]:
                self.conditions["after"].remove(job.id)
                if self.delimiter == "?":
                    self.conditions_met = True
                    return

    def can_release(self, queued_jobs, running_jobs):
        if not self.conditions_met:
            self.conditions_met = not any(self.conditions.values())
            if not self.conditions_met:
                return False

        if self.conditions_met and not self.singleton:
            return True

        if self.conditions_met and self.singleton:
            launched_jobs = running_jobs + queued_jobs
            if self.job_user_name in { (job.user, job.name) for job in launched_jobs }:
                return False
            return True

        return False

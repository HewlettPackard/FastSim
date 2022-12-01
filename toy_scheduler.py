"""
"""

import argparse, os, pickle, sys, joblib
from glob import glob
from datetime import datetime, timedelta
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.dates
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from funcs import parse_cache, timelimit_str_to_timedelta, hour_to_timeofday

CABS_DIR = "/home/y02/shared/power"
PLOT_DIR = "/work/y02/y02/awilkins/archer2_jobdata/plots"

# Factors are from linear fit (see powerusage_cabs_against_slurm_shifted_grouped.pdf)
# node_down_mean is just from assuming todays sinfo -R is typical (there were also big partial
# shutdowns at the start of the slurm data I am not accounting for). I am failry sure that these
# depend on node occupancy so I am just going to ignore these for now
BASELINE_POWER = 0 # kW (previously 1789 then 1692)
SLURMTOCAB_FACTOR = 1.0 # (previously 0.517 then 0.578)
NODEDOWN_MEAN = 291
BD_THRESHOLD = timedelta(minutes=10)


class Queue():
    def __init__(self, df_jobs, priority, init_time, custom_low_or_high=None):
        self.priority = priority
        self.time = init_time
        self.custom_low_or_high = custom_low_or_high

        self.all_jobs = [
            Job(
                job_row.Submit, job_row.AllocNodes, job_row.Elapsed, job_row.Timelimit,
                job_row.PowerPerNode, job_row.TruePowerPerNode
            ) for _, job_row in df_jobs.sort_values("Submit").iterrows()
        ]
        self.queue = []

    def step(self, t_step, retained):
        self.time += t_step

        if self.time < self.next_newjob():
            return

        try:
            while self.all_jobs[0].submit <= self.time:
                self.queue.append(self.all_jobs.pop(0))
        except IndexError:
            pass

        if self.priority == "fcfs":
            self.queue.sort(key=lambda job: job.submit)
        elif self.priority == "low-high_power":
            # high power priority at off-peak, low power priority at peak
            # To ensure that reordering the queue doesn't cause the scheduler to dance around
            # large jobs, any job that the scheduler was waiting to submit will be finished
            # before reordering the queue
            if hour_to_timeofday(self.time.hour) in ["morning", "afternoon", "evening"]:
                self.queue[retained:] = sorted(
                    self.queue[retained:], key=lambda job: job.node_power
                )
            else:
                self.queue[retained:] = sorted(
                    self.queue[retained:], key=lambda job: job.node_power, reverse=True
                )
        elif self.priority == "custom_low_or_high":
            if custom_low_or_high(self.time.hour) == "low":
                self.queue[retained:] = sorted(
                    self.queue[retained:], key=lambda job: job.node_power
                )
            else:
                self.queue[retained:] = sorted(
                    self.queue[retained:], key=lambda job: job.node_power, reverse=True
                )


    def next_newjob(self):
        try:
            return self.all_jobs[0].submit
        except IndexError:
            return datetime.max


class Job():
    def __init__(
        self, submit : datetime, nodes, runtime : timedelta, reqtime: timedelta, node_power,
        true_node_power
    ):
        self.nodes = nodes
        self.runtime = runtime
        self.reqtime = reqtime
        self.node_power = node_power
        self.true_node_power = true_node_power
        self.submit = submit

        self.start = None
        self.end = None

    def start_job(self, time : datetime):
        self.start = time
        self.end = time + self.runtime
        self.endlimit = time + self.reqtime
        return self


class ARCHER2():
    def __init__(
        self, init_time : datetime, baseline_power=1500, slurmtocab_factor=1.0, node_down_mean=0,
        backfill_opts={}
    ):
        self.power_usage = baseline_power / 1e+3 # MW
        self.slurmtocab_factor = slurmtocab_factor
        self.node_down_mean = node_down_mean
        self.time = init_time
        self.backfill_opts = backfill_opts
        if "min_block_width" not in self.backfill_opts:
            self.backfill_opts["min_block_width"] = timedelta(minutes=5) # 1min for ARCHER2
        if "max_job_test" not in self.backfill_opts:
            self.backfill_opts["max_job_test"] = 1000 # 1000 for ARCHER2

        self.running_jobs = []

        self.power_history = [self.power_usage] # MW
        self.occupancy_history = [0] # %
        self.queue_size = 0
        self.queue_size_history = [0]
        self.times = [self.time]
        self.bd_slowdowns = []

        self.nodes_free = 5860
        self.nodes_drained = 0
        self.nodes_drained_carryover = 0
        self.sorted = False

    def has_space(self, job : Job):
        return True if self.available_nodes() >= job.nodes else False

    def available_nodes(self):
        return self.nodes_free - self.nodes_drained

    def next_event(self):
        if not self.running_jobs:
            return datetime.max

        if not self.sorted:
            self.running_jobs.sort(key=lambda job: job.end)
            self.sorted = True

        return self.running_jobs[0].end

    def get_backfill_jobs(self, queue : Queue):
        backfill_now = []

        free_blocks = defaultdict(int)
        free_blocks[(self.time, datetime.max)] = self.available_nodes()
        for job in self.running_jobs:
            # Some jobs exceeding timelimit and some with zero reqtime
            if job.endlimit <= self.time:
                job.endlimit = self.time + 0.1 * job.reqtime + timedelta(minutes=1)
            free_blocks[(job.endlimit, datetime.max)] += job.nodes

        min_required_block_time = min(
            self.time + self.backfill_opts["min_block_width"],
            self.time + min(queue.queue, key=lambda job: job.reqtime).reqtime
        )

        # max_block_time = datetime.max
        free_blocks_ready_intervals = (
            { (self.time, datetime.max) } if self.available_nodes() else set()
        )
        max_block_time = datetime.max
        for i_job, job in enumerate(
            list(queue.queue)[:max(len(queue.queue), self.backfill_opts["max_job_test"])]
        ):
            # Only need to plan nodes for jobs that may be relevant to immediate scheduling
            if self.time + job.reqtime > max_block_time:
                continue

            free_nodes = 0
            selected_intervals = {}

            # break if no blocks or only <= min blocks available for immediate backfill
            if not free_blocks_ready_intervals:
                break
            max_block_time = max(free_blocks_ready_intervals, key=lambda interval: interval[1])[1]
            if max_block_time < min_required_block_time:
                break

            for interval, nodes in sorted(free_blocks.items(), key=lambda entry: entry[0][0]):
                selected_intervals[interval] = nodes
                free_nodes += nodes

                if job.nodes <= free_nodes:
                    usage_block_end = min(selected_intervals.keys(), key=lambda key: key[1])[1]
                    usage_block_start = max(selected_intervals.keys(), key=lambda key: key[0])[0]

                    # Nodes do not remain available for this jobs runtime, find new block
                    if usage_block_start + job.reqtime > usage_block_end:
                        selected_intervals = {}
                        free_nodes = 0
                        continue

                    usage_block_end = usage_block_start + job.reqtime

                    if usage_block_start == self.time:
                        backfill_now.append(i_job)

                    for key in selected_intervals.keys():
                        if key[0] == self.time:
                            free_blocks_ready_intervals.remove((key[0], key[1]))
                            if key[0] != usage_block_start:
                                free_blocks_ready_intervals.add((key[0], usage_block_start))

                        if key[0] != usage_block_start:
                            free_blocks[(key[0], usage_block_start)] += free_blocks[key]
                        if key[1] != usage_block_end:
                            free_blocks[(usage_block_end, key[1])] += free_blocks[key]

                        free_blocks.pop(key)

                    if free_nodes - job.nodes:
                        free_blocks[(usage_block_start, usage_block_end)] += free_nodes - job.nodes
                        if usage_block_start == self.time:
                            free_blocks_ready_intervals.add((key[0], usage_block_end))

                    break

        # print(len(backfill_now), self.available_nodes(), len(queue.queue), max_block_time - self.time, min(queue.queue, key=lambda job: job.reqtime).reqtime, queue.queue[0].nodes, sep=" | ")
        return backfill_now

    def submit_jobs(self, queue : Queue):
        for job in list(queue.queue):
            if self.has_space(job):
                self.submit(queue.queue.pop(0).start_job(self.time))
            else:
                break

        if queue.queue and self.available_nodes():
            backfill_now = self.get_backfill_jobs(queue)
            # for i in sorted(self.get_backfill_jobs(queue), reverse=True):
            for i in sorted(backfill_now, reverse=True):
                if not self.submit(queue.queue.pop(i).start_job(self.time)):
                    print(self.available_nodes(), i, backfill_now)

        self.queue_size = len(queue.queue)

    def submit(self, job : Job):
        if self.has_space(job):
            self.running_jobs.append(job)
            self.nodes_free -= job.nodes
            self.power_usage += (job.true_node_power * job.nodes * self.slurmtocab_factor) / 1e+6
            self.bd_slowdowns.append(
                max((job.end - job.submit)/max(job.runtime, BD_THRESHOLD), 1)
            )
            self.sorted = False
            return True
        else:
            print("No free nodes, job not submitted")
            return False

    def step(self, t_step : timedelta):
        self.time += t_step

        if not self.sorted:
            self.running_jobs.sort(key=lambda job: job.end)
            self.sorted = True

        while self.running_jobs and self.running_jobs[0].end <= self.time:
            job = self.running_jobs.pop(0)
            self.nodes_free += job.nodes
            self.power_usage -= (job.true_node_power * job.nodes * self.slurmtocab_factor) / 1e+6

        # Resample drained nodes every 12 hour at most
        if self.time.hour != (self.time - t_step).hour and not self.time.hour % 12:
            num_drain = max(
                (
                    round(np.random.normal(
                        loc=self.node_down_mean, scale=self.node_down_mean / 2
                    )) +
                    self.nodes_drained_carryover
                ),
                0
            )
            if num_drain <= self.nodes_free:
                self.nodes_drained = num_drain
                self.nodes_drained_carryover = 0
            else:
                self.nodes_drained = self.nodes_free
                self.nodes_drained_carryover = num_drain - self.nodes_free

        self.occupancy_history.append(1 - (self.available_nodes()/(5860 - self.nodes_drained)))
        self.power_history.append(self.power_usage)
        self.queue_size_history.append(self.queue_size)
        self.times.append(self.time)


def prep_job_data(data, cache, df_name, cols, model=None, rows=None):
    df_jobs = parse_cache(
        data, cache, ".".join(os.path.basename(data).split(".")[:-1]), df_name, cols, nrows=rows
    )

    df_jobs.AllocNodes = df_jobs.AllocNodes.astype(str)
    df_jobs.AllocNodes = df_jobs.AllocNodes.replace(
        { "K" : "e+03", "M" : "e+06", "G" : "e+09", "T" : "e+12" }, regex=True
    ).astype(float).astype(int)

    df_jobs["TruePowerPerNode"] = df_jobs.apply(
        lambda row: float(row.Power) / float(row.AllocNodes), axis=1
    )

    if model:
        df_jobs_cpy = df_jobs.copy()
        cols = ["ReqCPUS", "ReqNodes", "ReqMem"]
        df_jobs_cpy[cols] = df_jobs_cpy[cols].astype(str)
        df_jobs_cpy[cols] = df_jobs_cpy[cols].replace(
            { "K" : "e+03", "M" : "e+06", "G" : "e+09", "T" : "e+12" }, regex=True
        ).astype(float).astype(int)
        df_jobs_cpy.Submit = pd.to_datetime(df_jobs.Submit, format="%Y-%m-%dT%H:%M:%S")
        df_jobs_cpy.Submit = df_jobs_cpy.Submit.apply(lambda row: hour_to_timeofday(row.hour))
        df_jobs_cpy.Timelimit = df_jobs_cpy.Timelimit.apply(
            lambda row: round(timelimit_str_to_timedelta(row).total_seconds() / 60)
        )
        df_jobs["PowerPerNode"] = model.predict(df_jobs_cpy[[
            "JobID", "Start", "End", "ConsumedEnergyRaw", "AllocNodes", "ReqCPUS", "ReqNodes",
            "Group", "QOS", "ReqMem", "Timelimit", "Submit"
        ]])

    else:
        df_jobs["PowerPerNode"] = df_jobs.apply(
            lambda row: float(row.Power) / float(row.AllocNodes), axis=1
        )

    df_jobs = df_jobs.drop(["ReqCPUS", "ReqNodes", "Group", "QOS", "ReqMem"], axis=1)

    df_jobs.Elapsed = df_jobs.Elapsed.apply(lambda row: timelimit_str_to_timedelta(row))

    df_jobs.Timelimit = df_jobs.Timelimit.apply(lambda row: timelimit_str_to_timedelta(row))

    df_jobs.Submit = pd.to_datetime(df_jobs.Submit, format="%Y-%m-%dT%H:%M:%S")

    return df_jobs


def run_sim(
    df_jobs, system, scheduler, t0, seed=None, verbose=False, min_step=timedelta(seconds=10),
    custom_low_or_high=None
):
    queue = Queue(df_jobs, scheduler, t0)

    np.random.seed(seed)

    cnt = 0
    time = t0

    while queue.all_jobs or queue.queue or system.running_jobs:
        # Not enough precision to compute timedeltas with datetime.max
        try:
            t_step = max(min(queue.next_newjob() - time, system.next_event() - time), min_step)
        except pd._libs.tslibs.np_datetime.OutOfBoundsTimedelta:
            if queue.next_newjob() == datetime.max:
                t_step = system.next_event() - time
            else:
                t_step = queue.next_newjob() - time
        # Enforce minimum step to reduce simulation time
        if t_step < min_step:
            t_step = min_step
        time += t_step

        system.step(t_step)
        queue.step(t_step, 1 if queue.queue else None)

        system.submit_jobs(queue)

        # Print every 3 hours at most
        if verbose and (time.hour != (time - t_step).hour and not time.hour % 3):
            print(
                "{} (step {}):\n".format(time, cnt) +
                "Utilisation = {:.2f}%\tNodesDrained = {}({})\tPower = {:.4f} MW\tQueueSize = {}\t\
                 RunningJobs = {}".format(
                    system.occupancy_history[-1] * 100, system.nodes_drained,
                    system.nodes_drained_carryover, system.power_usage, len(queue.queue),
                    len(system.running_jobs)
                )
            )

        cnt += 1

    return system


# XXX implement save_prefix
def plot_blob(
    plots, archer, start, end, times, dates, archer_fcfs=None, times_fcfs=None, dates_fcfs=None,
    save_suffix="", batch=False
):
    def day_night_shade(ax, start, end):
        for day_num in range(round((end - start).days + 0.5) + 1):
            day = (start + timedelta(days=day_num)).replace(hour=0, minute=0, second=0)
            ax.axvspan(
                matplotlib.dates.date2num(day - timedelta(hours=4)),
                matplotlib.dates.date2num(day + timedelta(hours=8)),
                label="8pm - 8am" if not day_num else "_", color="gray", alpha=0.3
            )
            ax.axvspan(
                matplotlib.dates.date2num(day + timedelta(hours=8)),
                matplotlib.dates.date2num(day + timedelta(hours=20)),
                label="8am - 8pm" if not day_num else "_", color="lightgray", alpha=0.3
            )

    # TODO might want to shift the slurm info back as done in plot_sacct.py,
    # need to check the size of the shift in time rather than ticks then just hardcode it
    if "cab_power_plot" in plots:
        cabs = {
            datetime.strptime(os.path.basename(path), "system_%y%m%d") : path
            for path in glob(os.path.join(CABS_DIR, "system_[2]*"))
        }
        for date, cab in cabs.items():
            if date > start and (date + timedelta(days=1)) < end:
                df_cab = pd.read_csv(cab, delimiter=" ", names=["Time", "Power"])
                df_cab.Time = pd.to_datetime(df_cab.Time, format="%H:%M:%S")
                df_cab.Time = df_cab.Time.apply(
                    lambda row:
                    timedelta(hours=row.hour, minutes=row.minute, seconds=row.second) + date
                )

                try:
                    df_cabs = pd.concat([df_cabs, df_cab])
                except NameError:
                    df_cabs = df_cab
        t_cabs = pd.DatetimeIndex(df_cabs.sort_values("Time").Time.values)
        power_usage_cabs = np.zeros(t_cabs.values.size - 1)
        for i in range(len(power_usage_cabs)):
            slice = df_cabs.loc[(df_cabs.Time == t_cabs[i])]
            power_usage_cabs[i] = slice.Power.iloc[0] / 1000
        dates_cabs = matplotlib.dates.date2num(t_cabs[:-1])

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot_date(
            dates, np.array(archer.power_history), 'g', label="Toy scheduler - Slurm power",
            linewidth=0.6
        )
        ax.plot_date(dates_cabs, power_usage_cabs, 'r', label="Cabinet power", linewidth=0.6)
        ax.set_ylabel("Power (MW)")
        plt.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(
            PLOT_DIR, "toyscheduler_low-high_power_cabs{}.pdf".format(save_suffix)
        ))
        if batch:
            plt.close()
        else:
            plt.show()

    if "plot_v_fcfs" in plots:
        # Sanity check
        if sum(archer.power_history) != sum(archer_fcfs.power_history):
            print("{} - {}".format(sum(archer.power_history), sum(archer_fcfs.power_history)))

        fig = plt.figure(1, figsize=(12, 8))
        ax = fig.add_axes((.1, .3, .8, .6))
        ax.plot_date(
            dates_fcfs, np.array(archer_fcfs.power_history), 'r',
            label="Toy scheduler fcfs - Slurm power", linewidth=0.6
        )
        ax.plot_date(
            dates, np.array(archer.power_history), 'g',
            label="Toy scheduler low-high_power - Slurm power", linewidth=0.6
        )
        day_night_shade(ax, start, end)
        ax.set_ylabel("Power (MW)")
        ax.set_xticklabels([])
        plt.legend()
        ax2 = fig.add_axes((.1, .1, .8, .2))
        ax2.plot_date(
            dates_fcfs, np.array(archer_fcfs.occupancy_history) * 100, 'r',
            label="Toy scheduler fcfs - system occupancy", linewidth=0.6
        )
        ax2.plot_date(
            dates, np.array(archer.occupancy_history) * 100, 'g',
            label="Toy scheduler low-high_power - system occupancy", linewidth=0.6
        )
        day_night_shade(ax2, start, end)
        ax2.set_ylabel("System Occupancy (%)")
        plt.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(
            PLOT_DIR, "toyscheduler_low-high_power_fcfs_power_occupancy{}.pdf".format(save_suffix)
        ))
        if batch:
            plt.close()
        else:
            plt.show()

        fig = plt.figure(1, figsize=(12, 8))
        ax = fig.add_axes((.1, .3, .8, .6))
        ax.plot_date(
            dates, np.array(archer.power_history), 'g',
            label="Toy scheduler low-high_power - Slurm power", linewidth=0.6
        )
        day_night_shade(ax, start, end)
        ax.set_ylabel("Power (MW)")
        ax.set_xticklabels([])
        ax.set_title("Power Usage of Using low-high Power Toy Scheduler")
        ax2 = fig.add_axes((.1, .1, .8, .2))
        ax2.plot_date(
            dates, np.array(archer.queue_size_history), 'k',
            label="Toy scheduler low-high_power - queue size", linewidth=0.6
        )
        day_night_shade(ax2, start, end)
        ax2.set_ylabel("# Jobs")
        ax2.set_ylim(bottom=-0.1 * max(archer.queue_size_history))
        ax2.axhline(0, linestyle="dashed", c="k", linewidth=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(
            PLOT_DIR, "toyscheduler_low-high_power_power_queue{}.pdf".format(save_suffix)
        ))
        if batch:
            plt.close()
        else:
            plt.show()

        # The same plot but in a ranges of interest
        for start_month, start_day, end_month, end_day in (
            [(10, 24, 11, 7), (10, 24, 11, 20), (11, 8, 11, 26)]
        ):
            for tick, time in enumerate(times):
                if time.month == start_month and time.day == start_day:
                    start_tick, start_time = tick, time
                    break
            if tick + 1 == len(times): # range not valid for this data
                continue
            for tick, time in enumerate(reversed(times)):
                if time.month == end_month and time.day == end_day:
                    # NOTE: len(times) - tick indexs end_time + t_step, this is fine because
                    # end_tick is being used as upper index in ranges
                    end_tick, end_time = len(times) - tick, time
                    break
            if tick + 1 == len(times):
                continue

            dates_crop = dates[start_tick:end_tick]
            fig = plt.figure(1, figsize=(12, 8))
            ax = fig.add_axes((.1, .3, .8, .6))
            ax.plot_date(
                dates_crop, np.array(archer.power_history)[start_tick:end_tick], 'g',
                label="Toy scheduler low-high_power - Slurm power", linewidth=0.6
            )
            day_night_shade(ax, start_time, end_time)
            ax.set_ylabel("Power (MW)")
            ax.set_xticklabels([])
            ax.set_title("Power Usage of Using low-high Power Toy Scheduler")
            ax2 = fig.add_axes((.1, .1, .8, .2))
            ax2.plot_date(
                dates_crop, np.array(archer.queue_size_history)[start_tick:end_tick], 'k',
                label="Toy scheduler low-high_power - queue size", linewidth=0.6
            )
            day_night_shade(ax2, start_time, end_time)
            ax2.set_ylabel("# Jobs")
            ax2.set_ylim(bottom=-0.1 * max(archer.queue_size_history[start_tick:end_tick]))
            ax2.axhline(0, linestyle="dashed", c="k", linewidth=0.5)
            fig.tight_layout()
            fig.savefig(os.path.join(
                PLOT_DIR, "toyscheduler_low-high_power_power_queue {}{}-{}{}{}.pdf".format(
                    start_day, start_month, end_day, end_month, save_suffix
                )
            ))
            if batch:
                plt.close()
            else:
                plt.show()

            day_powers, night_powers = [], []
            day_occupancies, night_occupancies = [], []
            for tick, date in enumerate(times[start_tick:end_tick]):
                if hour_to_timeofday(date.hour) in ["morning", "afternoon", "evening"]:
                    day_powers.append(archer.power_history[start_tick:end_tick][tick])
                    day_occupancies.append(archer.occupancy_history[start_tick:end_tick][tick])
                else:
                    night_powers.append(archer.power_history[start_tick:end_tick][tick])
                    night_occupancies.append(archer.occupancy_history[start_tick:end_tick][tick])

            print(
                "For low-high power scheduler in range {}-{} to {}-{} ".format(
                    start_day, start_month, end_day, end_month
                ) +
                "(roughly when there is a queue):\n" +
                "Mean daytime power = {:.4f} MW\t Mean nightime power = {:.4f} MW".format(
                    np.mean(day_powers), np.mean(night_powers)
                ) +
                "Mean daytime occupancy = {:.2f} %\t Mean nightime occupancy = {:.2f} %".format(
                    np.mean(day_occupancies) * 100, np.mean(night_occupancies) * 100
                )
            )

        print(
            "Mean low-high power bounded slowdown = {:.2f} \t\
             Mean fcfs bounded slowdown = {:.2f}".format(
                np.mean(archer.bd_slowdowns), np.mean(archer_fcfs.bd_slowdowns)
            )
        )

        print(
            min(archer.bd_slowdowns), max(archer.bd_slowdowns),
            (np.array(archer.bd_slowdowns) == 1).sum(), sep=" | "
        )
        print(
            min(archer_fcfs.bd_slowdowns), max(archer_fcfs.bd_slowdowns),
            (np.array(archer_fcfs.bd_slowdowns) == 1).sum(), sep=" | "
        )
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.hist(
            archer.bd_slowdowns, bins=int(max(archer.bd_slowdowns)),
            range=(min(archer.bd_slowdowns),max(archer.bd_slowdowns)), histtype="step",
            label="low-high_power"
        )
        ax.hist(
            archer_fcfs.bd_slowdowns, bins=int(max(archer.bd_slowdowns)),
            range=(min(archer.bd_slowdowns),max(archer.bd_slowdowns)), histtype="step",
            label="fcfs"
        )
        ax.set_yscale("log")
        ax.set_title("Bounded slowdowns")
        plt.legend()
        fig.savefig(os.path.join(
            PLOT_DIR, "toyscheduler_low-high_power_boundedslowdowns{}.pdf".format(save_suffix)
        ))
        if batch:
            plt.close()
        else:
            plt.show()

    if "scan_plots" in plots:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        for interval, archer_entry in archer.items():
            bd_slowdown = np.mean(archer_entry.bd_slowdowns[1000:-1000])

            if type(interval) == int:
                low_or_high = lambda hr: "low" if (hr // interval) % 2 == 0 else "high"
            elif type(interval) == tuple:
                low_or_high = lambda hr: (
                    "low" if (hr % interval[1][1]) < interval[0][1] else "high"
                )

            low_powers, high_powers = [], []
            for tick, date in enumerate(times[1000:-1000]):
                if low_or_high(date.hour) == "low":
                    low_powers.append(archer_entry.power_history[1000:-1000][tick])
                else:
                    high_powers.append(archer_entry.power_history[1000:-1000][tick])

            print(interval, len(low_powers), len(high_powers), sep="\t")
            mean_low_power = np.mean(low_powers)
            mean_high_power = np.mean(high_powers)

            ax.scatter(bd_slowdown, (mean_high_power - mean_low_power) * 1000, s=6, label=str(interval))

        ax.set_ylabel("MeanHighPower - MeanLowPower (kW)")
        ax.set_xlabel("Mean Bounded Slowdown")
        plt.legend()
        fig.savefig(os.path.join(
            PLOT_DIR,
            "toyscheduler_low-high_power_boundedslowdown_powerdifference_scan{}.pdf".format(
                save_suffix
            )
        ))
        if batch:
            plt.close()
        else:
            plt.show()


def main(args):
    df_jobs = prep_job_data(
        args.data, args.cache,
        "toy_scheduler_df_{}".format("predpower" if args.use_power_preds else "truepower"),
        [
            "JobID", "Start", "End", "Submit", "Elapsed", "ConsumedEnergyRaw", "AllocNodes",
            "Timelimit", "ReqCPUS", "ReqNodes", "Group", "QOS", "ReqMem"
        ],
        model=joblib.load(args.use_power_preds) if args.use_power_preds else None, rows=args.rows
    )
    t0 = df_jobs.Start.min()

    if args.read_sim_from:
        print("Reading sim results from {} ...".format(args.read_sim_from))
        with open(args.read_sim_from, "rb") as f:
            data = pickle.load(f)
        archer = data["archer"]
        if args.plot_v_fcfs:
            archer_fcfs = data["archer_fcfs"]

    else:
        if args.scan_low_high_power:
            archer = {}
            for switch_hr in [6, 12, 18, 24, 30, 36, 42, 48]:
                print(
                    "Running sim for scheduler low-high_power swithching at {} hr \
                     intervals...".format(switch_hr)
                )
                low_or_high = lambda time: (
                    "low" if (
                        (((time - t0) // timedelta(hours=1)) // switch_hr) % 2 == 0
                    )
                    else "high"
                )
                archer[((0,switch_hr),(switch_hr,switch_hr * 2))] = run_sim(
                    df_jobs,
                    ARCHER2(
                        t0, baseline_power=BASELINE_POWER, slurmtocab_factor=SLURMTOCAB_FACTOR,
                        node_down_mean=NODEDOWN_MEAN
                    ),
                    "custom_low-high_power", t0, seed=0, verbose=args.verbose,
                    custom_low_or_high=low_or_high
                )
            # ((low_start,low_end),(high_start,high_end))
            for switch_intervals in [((0,12),(12,48)), ((0,24),(24,72))]:
                print(
                    "Running sim for scheduler low-high_power swithching at {} hr \
                     intervals...".format(switch_intervals)
                )
                low_or_high = lambda hr: (
                    "low" if (
                        (((time - t0) // timedelta(hours=1)) % switch_interval[1][1]) <
                        switch_interval[0][1]
                    )
                    else "high"
                )
                archer[((0,switch_hr),(switch_hr,switch_hr * 2))] = run_sim(
                    df_jobs,
                    ARCHER2(
                        t0, baseline_power=BASELINE_POWER, slurmtocab_factor=SLURMTOCAB_FACTOR,
                        node_down_mean=NODEDOWN_MEAN
                    ),
                    "custom_low-high_power", t0, seed=0, verbose=args.verbose,
                    custom_low_or_high=low_or_high
                )
        else:
            print("Running sim for scheduler low-high_power...")
            archer = run_sim(
                df_jobs,
                ARCHER2(
                    t0, baseline_power=BASELINE_POWER, slurmtocab_factor=SLURMTOCAB_FACTOR,
                    node_down_mean=NODEDOWN_MEAN
                ),
                "fcfs" if args.fcfs else "low-high_power", t0, seed=0, verbose=args.verbose
            )
            if args.plot_v_fcfs:
                print("Running sim for scheduler fcfs...")
                archer_fcfs = run_sim(
                    df_jobs,
                    ARCHER2(
                        t0, baseline_power=BASELINE_POWER, slurmtocab_factor=SLURMTOCAB_FACTOR,
                        node_down_mean=NODEDOWN_MEAN
                    ),
                    "fcfs", t0, seed=0, verbose=args.verbose
                )

    if args.dump_sim_to:
        data = { "archer" : archer }
        if args.plot_v_fcfs:
            data["archer_fcfs"] = archer_fcfs
        with open(args.dump_sim_to, 'wb') as f:
            pickle.dump(data, f)

    archer_times = list(archer.values())[0].times if args.scan_low_high_power else archer.times
    start, end = archer_times[0], archer_times[-1]
    times = pd.DatetimeIndex(archer_times)
    dates = matplotlib.dates.date2num(times)

    plots = []
    if args.cab_power_plot:
        plots.append("cab_power_plot")
    if args.plot_v_fcfs:
        plots.append("plot_v_fcfs")
    if args.scan_low_high_power:
        plots.append("scan_plots")

    if "plot_v_fcfs" in plots:
        times_fcfs = pd.DatetimeIndex(archer_fcfs.times)
        dates_fcfs = matplotlib.dates.date2num(times_fcfs)
        plot_blob(
            plots, archer, start, end, times, dates, archer_fcfs=archer_fcfs,
            times_fcfs=times_fcfs, dates_fcfs=dates_fcfs, batch=args.batch
        )
    else:
        plot_blob(plots, archer, start, end, times, dates, batch=args.batch)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("data", type=str)

    scheduler_group = parser.add_mutually_exclusive_group()
    scheduler_group.add_argument(
        "--fcfs", action="store_true",
        help="Use only FCFS scheduler for validation"
    )
    scheduler_group.add_argument(
        "--low_high_power", action="store_true",
        help="Alternate between scheduling lowest power and highest power jobs"
    )
    scheduler_group.add_argument(
        "--scan_low_high_power", action="store_true",
        help="Scan different intervals for low-high power scheduling"
    )

    parser.add_argument(
        "--use_power_preds", type=str, default="",
        help="Use PowerPerNode from a trained model"
    )

    parser.add_argument(
        "--cab_power_plot", action="store_true",
        help="Plot toy scheduler power history with power from the cabinet data"
    )

    parser.add_argument(
        "--plot_v_fcfs", action="store_true",
        help="Plot the low-high-power scheduler power usage against the power usage for a fcfs " +
             "scheduler"
    )

    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--cache", type=str, default="", help="How to use the cache (save|load)")
    parser.add_argument("--batch", action='store_true')
    parser.add_argument("--verbose", action="store_false")
    parser.add_argument("--dump_sim_to", type=str, default="", help="Pickle sim results")
    parser.add_argument("--read_sim_from", type=str, default="", help="Read pickled sim results")

    args = parser.parse_args()

    if args.plot_v_fcfs and args.fcfs:
        raise argparse.ArgumentError("plot_v_fcfs incompatible with fcfs scheduler")

    if args.rows and args.cache == "load":
        print("Note: rows cannot be set if loading data from cache")

    return args

if __name__ == '__main__':
    main(parse_arguments())


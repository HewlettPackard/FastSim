"""
"""

import argparse, os
from glob import glob
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import matplotlib.dates
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

from funcs import parse_cache, timelimit_str_to_timedelta, hour_to_timeofday


CABS_DIR="/home/y02/shared/power"
PLOT_DIR="/work/y02/y02/awilkins/archer2_jobdata/plots"

class Job():
    def __init__(self, submit : datetime, nodes, runtime : timedelta, node_power):
        self.nodes = nodes
        self.runtime = runtime
        self.node_power = node_power
        self.submit = submit

        self.start = None
        self.end = None

    def start_job(self, time : datetime):
        self.start = time
        self.end = time + self.runtime
        return self


class ARCHER2():
    def __init__(self, init_time : datetime, baseline_power=1500, slurmtocab_factor=1.0, node_down_mean=0):
        self.nodes_free = 5860
        self.nodes_drained = 0
        self.nodes_drained_carryover = 0
        self.power_usage = baseline_power # kW
        self.slurmtocab_factor = slurmtocab_factor
        self.node_down_mean = node_down_mean
        self.time = init_time

        self.running_jobs = []

        self.power_history = [] # MW
        self.occupancy_history = [] # %

    def has_space(self, job : Job):
        return True if self.nodes_free - self.nodes_drained >= job.nodes else False

    def submit(self, job : Job):
        if self.has_space(job):
            self.running_jobs.append(job)
            self.nodes_free -= job.nodes
            self.power_usage += (job.node_power * job.nodes * self.slurmtocab_factor) / 1000
        else:
            print("No free nodes, job not submitted")

    def step(self, time : timedelta):
        self.occupancy_history.append(5860 - self.nodes_free)
        self.power_history.append(self.power_usage / 1000)

        self.time += time

        self.running_jobs.sort(key=lambda job: job.end)

        while self.running_jobs and self.running_jobs[0].end <= self.time:
            job = self.running_jobs.pop(0)
            self.nodes_free += job.nodes
            self.power_usage -= (job.node_power * job.nodes * self.slurmtocab_factor) / 1000

        # Simulate drained nodes every hour at most
        if self.time.hour != (self.time - time).hour:
            num_drain = max(
                (
                    round(np.random.normal(loc=self.node_down_mean, scale=self.node_down_mean / 2)) +
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


def run_sim(df_jobs, system, scheduler, t_step, t0, seed=None, verbose=False):
    time = t0 + t_step
    queue = []

    df_jobs = df_jobs.copy()
    np.random.seed(seed)
    cnt = 0
    while len(df_jobs):
        for _, job_row in df_jobs.loc[(df_jobs.Submit < time)].iterrows():
            queue.append(
                Job(job_row.Submit, job_row.AllocNodes, job_row.Elapsed, job_row.PowerPerNode)
            )
        df_jobs = df_jobs.loc[~(df_jobs.Submit < time)]

        system.step(t_step)

        if scheduler == "fcfs":
            queue.sort(key=lambda job: job.submit)
        elif scheduler == "low-high_power":
            # high power priority at off-peak, low power priority at peak
            # To ensure large jobs don't get ignored, each time low/high priority is switched, put
            # the job that has been queueing longest at the front of the queue
            queue.sort(key=lambda job: job.submit)
            if hour_to_timeofday(time.hour) in ["morning", "afternoon", "evening"]:
                queue[1:] = sorted(queue[1:], key=lambda job: job.node_power)
            else:
                queue[1:] = sorted(queue[1:], key=lambda job: job.node_power, reverse=True)

        for job in list(queue):
            if system.has_space(job):
                system.submit(queue.pop(0).start_job(time))
            else:
                break

        # Checking why utilisation drops for low power priority.
        # There is lowish power 3000 node job that is never getting time to be submitted
        # if system.nodes_free > 1000 and len(queue):
        #     print(time, hour_to_timeofday(time.hour), queue[0].nodes, queue[0].node_power)

        if verbose and cnt % 25 == 0:
            print(
                "{} (step {}):\n".format(time, cnt) +
                "Utilisation = {:.2f}%\tNodesDrained = {}({})\tPower = {:.4f} MW\tQueueSize = {}\t\
                 RunningJobs = {}".format(
                    (1 - (system.nodes_free / 5860)) * 100, system.nodes_drained,
                    system.nodes_drained_carryover, system.power_usage / 1000, len(queue),
                    len(system.running_jobs)
                )
            )

        time += t_step
        cnt += 1

    return system


def main(args):
    df_jobs = parse_cache(
        args.data, args.cache, ".".join(os.path.basename(args.data).split(".")[:-1]),
        "toy_scheduler_df",
        ["JobID", "Start", "End", "Submit", "Elapsed", "ConsumedEnergyRaw", "AllocNodes"],
        nrows=args.rows
    )

    df_jobs.AllocNodes = df_jobs.AllocNodes.astype(str)
    df_jobs.AllocNodes = df_jobs.AllocNodes.replace(
        { "K" : "e+03", "M" : "e+06", "G" : "e+09", "T" : "e+12" }, regex=True
    ).astype(float).astype(int)

    df_jobs["PowerPerNode"] = df_jobs.apply(
        lambda row: float(row.Power) / float(row.AllocNodes), axis=1
    )

    df_jobs.Elapsed = df_jobs.Elapsed.apply(lambda row: timelimit_str_to_timedelta(row))

    df_jobs.Submit = pd.to_datetime(df_jobs.Submit, format="%Y-%m-%dT%H:%M:%S")

    scheduler = "fcfs" if args.fcfs else "low-high_power"
    t0 = df_jobs.Start.min()
    t_step = timedelta(minutes=args.stepsize)
    # Factors are from linear fit (see powerusage_cabs_against_slurm_shifted_grouped.pdf)
    # node_down_mean is just from assuming todays sinfo -R is typical (there were also big partial
    # shutdowns at the start of the slurm data I am not accounting for)
    print("Running sim for scheduler {}...".format(scheduler))
    archer = run_sim(
        df_jobs, ARCHER2(t0, baseline_power=1789, slurmtocab_factor=0.517, node_down_mean=291),
        scheduler, t_step, t0, seed=0, verbose=args.verbose
    )
    if args.plot_v_fcfs:
        print("Running sim for scheduler fcfs...")
        archer_fcfs = run_sim(
            df_jobs, ARCHER2(t0, baseline_power=1789, slurmtocab_factor=0.517, node_down_mean=291),
            "fcfs", t_step, t0, seed=0, verbose=args.verbose
        )

    start, end = t0 + t_step, archer.time
    t_toy = pd.date_range(start, end, periods=len(archer.power_history))
    dates_toy = matplotlib.dates.date2num(t_toy)

    # TODO might want to shift the slurm info back as done in plot_sacct.py,
    # need to check the size of the shift in time rather than ticks then just hardcode it
    if args.cab_power_plot:
        start, end = t0 + t_step, archer.time
        t_toy = pd.date_range(start, end, periods=len(archer.power_history))
        dates_toy = matplotlib.dates.date2num(t_toy)

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
            dates_toy, np.array(archer.power_history), 'g', label="Toy scheduler - Slurm power",
            linewidth=0.6
        )
        ax.plot_date(dates_cabs, power_usage_cabs, 'r', label="Cabinet power", linewidth=0.6)
        ax.set_ylabel("Power (MW)")
        plt.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(PATH_DIR, "toyscheduler_{}_cabs.pdf".format(scheduler)))
        plt.show()

    if args.plot_v_fcfs:
        start, end = t0, archer.time - t_step
        t_toy = pd.date_range(start, end, periods=len(archer.power_history))
        dates_toy = matplotlib.dates.date2num(t_toy)

        # Sanity check
        if sum(archer.power_history) != sum(archer_fcfs.power_history):
            print("AHHH {} - {}".format(sum(archer.power_history), sum(archer_fcfs.power_history)))

        fig = plt.figure(1, figsize=(12, 8))
        ax = fig.add_axes((.1, .3, .8, .6))
        ax.plot_date(
            dates_toy, np.array(archer.power_history), 'g',
            label="Toy scheduler low-high_power - Slurm power", linewidth=0.6
        )
        ax.plot_date(
            dates_toy, np.array(archer_fcfs.power_history), 'r',
            label="Toy scheduler fcfs - Slurm power", linewidth=0.6
        )
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
        ax.set_ylabel("Power (MW)")
        ax.set_xticklabels([])
        plt.legend()
        ax2 = fig.add_axes((.1, .1, .8, .2))
        ax2.plot_date(
            dates_toy, (np.array(archer.occupancy_history) / 5860) * 100, 'g',
            label="Toy scheduler low-high_power - system occupancy", linewidth=0.6
        )
        ax2.plot_date(
            dates_toy, (np.array(archer_fcfs.occupancy_history) / 5860) * 100, 'r',
            label="Toy scheduler fcfs - system occupancy", linewidth=0.6
        )
        for day_num in range(round((end - start).days + 0.5) + 1):
            day = (start + timedelta(days=day_num)).replace(hour=0, minute=0, second=0)
            ax2.axvspan(
                matplotlib.dates.date2num(day - timedelta(hours=4)),
                matplotlib.dates.date2num(day + timedelta(hours=8)),
                label="_", color="gray", alpha=0.3
            )
            ax2.axvspan(
                matplotlib.dates.date2num(day + timedelta(hours=8)),
                matplotlib.dates.date2num(day + timedelta(hours=20)),
                label="_", color="lightgray", alpha=0.3
            )
        ax2.set_ylabel("System Occupancy (%)")
        plt.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "toyscheduler_low-high_power_fcfs_power_occupancy.pdf"))
        plt.show()

        day_powers, night_powers = [], []
        day_occupancies, night_occupancies = [], []
        for tick, date in enumerate(t_toy):
            if hour_to_timeofday(date.hour) in ["morning", "afternoon", "evening"]:
                day_powers.append(archer.power_history[tick])
                day_occupancies.append(archer.occupancy_history[tick] / 5860)
            else:
                night_powers.append(archer.power_history[tick])
                night_occupancies.append(archer.occupancy_history[tick] / 5860)

        print(
            "For low-high power scheduler:\n" +
            "Mean daytime power = {:.4f} MW\tMean nightime power = {:.4f} MW".format(
                np.mean(day_powers), np.mean(night_powers)
            ) +
            "Mean daytime occupancy = {:.2f} %\tMean nightime occupancy = {:.2f} %".format(
                np.mean(day_occupancies) * 100, np.mean(night_occupancies) * 100
            )
        )


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("data", type=str)

    scheduler_group = parser.add_mutually_exclusive_group()
    scheduler_group.add_argument(
        "--fcfs", action="store_true",
        help="Use only FCFS scheduler for validation"
    )
    scheduler_group.add_argument(
        "--low-high_power", action="store_true",
        help="Alternate between scheduling lowest power and highest power jobs"
    )

    parser.add_argument(
        "--stepsize", type=float, default=5,
        help="Size of time step to use for simulation in minutes"
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
    parser.add_argument("--verbose", action="store_false")

    args = parser.parse_args()

    if args.plot_v_fcfs and args.fcfs:
        raise argparse.ArgumentError("plot_v_fcfs incompatible with fcfs scheduler")

    if args.rows and args.cache == "load":
        print("Note: rows cannot be set if loading data from cache")

    return args

if __name__ == '__main__':
    main(parse_arguments())


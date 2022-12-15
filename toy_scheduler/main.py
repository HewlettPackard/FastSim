import argparse, os, joblib, sys
import dill as pickle
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.dates

from classes import Archer2
from simulation_funcs import prep_job_data, run_sim
from plotting import plot_blob
from globals import * # TODO: dont do this
sys.path.append("/work/y02/y02/awilkins/archer2_jobdata")
from funcs import hour_to_timeofday


""" Priority Sorters """

class FIFOSorter():
    def __init__(self):
        pass

    def sort(self, queue, time):
        return sorted(queue, key=lambda job: job.submit)


class SimpleLowHighPowerSorter():
    def __init__(self):
        pass

    def sort(self, queue, time):
        if hour_to_timeofday(time.hour) in ["morning", "afternoon", "evening"]:
            return sorted(queue, key=lambda job: job.node_power)
        else:
            return sorted(queue, key=lambda job: job.node_power, reverse=True)


class LowHighPowerSorter():
    def __init__(self, switch_interval, t0):
        self.switch_interval = switch_interval
        self.t0 = t0

    def sort(self, queue, time):
        if (
            (((time - self.t0) // timedelta(hours=1)) % self.switch_interval[1][1]) <
            self.switch_interval[0][1]
        ):
            return sorted(queue, key=lambda job: job.node_power)
        else:
            return sorted(queue, key=lambda job: job.node_power, reverse=True)


class DataStartSorter():
    def __init__(self):
        pass

    def sort(self, queue, time):
        return sorted(queue, key=lambda job: job.true_job_start)


class AgeSizeSorter():
    def __init__(self, prioritise_small, size_weight, age_weight=1.0, noise_params=None):
        self.age_weight = age_weight
        if prioritise_small:
            self.size_factor_calc = lambda job: (1 / job.nodes) * size_weight
        else:
            self.size_factor_calc = lambda job: (job.nodes / 5860) * size_weight
        if noise_params:
            self.noise_factor_calc = (
                lambda job: (
                    np.clip(
                        np.random.normal(noise_params["mu"], noise_params["sigma"]), 0.0, 1.0
                    ) *
                    noise_params["weight"]
                )
            )
        else:
            self.noise_factor_calc = lambda job: 0


    def sort(self, queue, time):
        sorted_queue = sorted(
            queue,
            key=(
                lambda job: (
                    min((time - job.submit).total_seconds() / (24 * (60**2)) / 14, 1) *
                    self.age_weight + self.size_factor_calc(job) + self.noise_factor_calc(job)
                )
            ),
            reverse=True
        )
        return sorted_queue

""" End Priority Sorters """

""" Low Freq Response Calcs """

class AppGroupMeanResponses():
    pass

""" End Low Freq Response Calcs"""


def main(args):
    df_jobs = prep_job_data(
        args.data, args.cache,
        "toy_scheduler_df_{}".format("predpower" if args.use_power_preds else "truepower"),
        joblib.load(args.use_power_preds) if args.use_power_preds else None, rows=args.rows
    )
    t0 = df_jobs.Start.min()

    if args.read_sim_from:
        print("Reading sim results from {} ...".format(args.read_sim_from))
        with open(args.read_sim_from, "rb") as f:
            data = pickle.load(f)
        archer = data["archer"]
        archer_fcfs = data["archer_fcfs"]

    else:
        if args.scan_low_high_power:
            archer = {}
            # ((low_start,low_end),(high_start,high_end))
            switch_intervals = [
                ((0,6),(6,12)), ((0,12),(12,24)), ((0,18),(18,36)), ((0,24),(24,48)),
                ((0,36),(36,72)), ((0,48),(48,96)), ((0,12),(12,48)), ((0,24),(24,72)),
                ((0,8),(8,24))
            ]
            for switch_interval in switch_intervals:
                print(
                    "Running sim for scheduler low-high_power switching at {} hr \
                     intervals...".format(switch_interval)
                )
                archer[switch_interval] = run_sim(
                    df_jobs,
                    Archer2(
                        t0, baseline_power=BASELINE_POWER, slurmtocab_factor=SLURMTOCAB_FACTOR,
                        node_down_mean=NODEDOWN_MEAN, backfill_opts=BACKFILL_OPTS
                    ),
                    t0, LowHighPowerSorter(switch_interval, t0), seed=0, verbose=args.verbose,
                    min_step=MIN_STEP
                )

        elif args.true_job_start_test:
            print("Running sim for scheduler using job start times from data...")
            archer = run_sim(
                df_jobs,
                Archer2(
                    t0, baseline_power=BASELINE_POWER, slurmtocab_factor=SLURMTOCAB_FACTOR,
                    node_down_mean=NODEDOWN_MEAN, backfill_opts=BACKFILL_OPTS
                ),
                t0, DataStartSorter(), seed=0, verbose=args.verbose, min_step=MIN_STEP,
                no_retained=True
            )

        elif args.scan_job_size_weights or args.scan_job_size_weights_noise:
            archer = {}
            # size_weights = [0.01, 0.05, 0.1, 0.5, 1]
            # size_weights = [2, 2.25, 2.5, 3]
            size_weights = [2.25]
            if args.scan_job_size_weights_noise:
                noise_params_lst = [
                    { "mu" : 0.5, "sigma" : 0.5, "weight" : weight } for weight in (
                        [1, 2, 5]
                    )
                ]
            else:
                noise_params_lst = [None]

            for size_weight in size_weights:
                for noise_params in noise_params_lst:
                    print(
                        "Running sim for scheduler with age and priority small job size with" +
                        "size weight {} and noise params {} ...".format(size_weight, noise_params)
                    )
                    key = (size_weight, noise_params["weight"]) if noise_params else size_weight
                    archer[key] = run_sim(
                        df_jobs,
                        Archer2(
                            t0, baseline_power=BASELINE_POWER, slurmtocab_factor=SLURMTOCAB_FACTOR,
                            node_down_mean=NODEDOWN_MEAN, backfill_opts=BACKFILL_OPTS
                        ),
                        t0, AgeSizeSorter(True, size_weight, noise_params=noise_params), seed=0,
                        verbose=args.verbose, min_step=MIN_STEP
                    )
            print("Running sim for scheduler using job start times from data...")
            archer[-1] = run_sim(
                df_jobs,
                Archer2(
                    t0, baseline_power=BASELINE_POWER, slurmtocab_factor=SLURMTOCAB_FACTOR,
                    node_down_mean=NODEDOWN_MEAN, backfill_opts=BACKFILL_OPTS
                ),
                t0, DataStartSorter(), seed=0, verbose=args.verbose, min_step=MIN_STEP,
                no_retained=True
            )
            if args.scan_job_size_weights:
                print("Running sim for scheduler with priority small job size")
                archer[999] = run_sim(
                    df_jobs,
                    Archer2(
                        t0, baseline_power=BASELINE_POWER, slurmtocab_factor=SLURMTOCAB_FACTOR,
                        node_down_mean=NODEDOWN_MEAN, backfill_opts=BACKFILL_OPTS
                    ),
                    t0, AgeSizeSorter(True, 1.0, age_weight=0.0), seed=0, verbose=args.verbose,
                    min_step=MIN_STEP
                )

        elif args.test_frequencies:
            kde = joblib.load(KDE_MODEL_2)
            archer = {}
            size_weight, noise_params = 2.25, None
            small_queue_cuts = [0, 10, 50,  100, 500, 1000, float("inf")]
            for small_queue_cut in small_queue_cuts:
                print(
                    "Running sim for scheduler with age and priority small job size with size" +
                    "weight {} and noise params {}. Setting jobs to 2GHz when quene < {}".format(
                        size_weight, noise_params, small_queue_cut
                    )
                )
                archer[small_queue_cut] = run_sim(
                    df_jobs,
                    Archer2(
                        t0, baseline_power=BASELINE_POWER, slurmtocab_factor=SLURMTOCAB_FACTOR,
                        node_down_mean=NODEDOWN_MEAN, backfill_opts=BACKFILL_OPTS,
                        low_freq_condition=lambda queue: len(queue.queue) < small_queue_cut,
                        low_freq_kde=kde, max_time_factor=1.2
                    ),
                    t0, AgeSizeSorter(True, size_weight, noise_params=noise_params), seed=0,
                    verbose=args.verbose, min_step=MIN_STEP
                )
            print(
                "Running sim for scheduler with age and priority small job size with size" +
                "weight {} and noise params {}. No 2GHz jobs".format(size_weight, noise_params)
            )
            archer[-1] = run_sim(
                df_jobs,
                Archer2(
                    t0, baseline_power=BASELINE_POWER, slurmtocab_factor=SLURMTOCAB_FACTOR,
                    node_down_mean=NODEDOWN_MEAN, backfill_opts=BACKFILL_OPTS
                ),
                t0, AgeSizeSorter(True, size_weight, noise_params=noise_params), seed=0,
                verbose=args.verbose, min_step=MIN_STEP
            )

        else:
            print("Running sim for scheduler low-high_power...")
            archer = run_sim(
                df_jobs,
                Archer2(
                    t0, baseline_power=BASELINE_POWER, slurmtocab_factor=SLURMTOCAB_FACTOR,
                    node_down_mean=NODEDOWN_MEAN, backfill_opts=BACKFILL_OPTS
                ),
                t0, SimpleLowHighPowerSorter(), seed=0, verbose=args.verbose, min_step=MIN_STEP
            )

        print("Running sim for scheduler fcfs...")
        archer_fcfs = run_sim(
            df_jobs,
            Archer2(
                t0, baseline_power=BASELINE_POWER, slurmtocab_factor=SLURMTOCAB_FACTOR,
                node_down_mean=NODEDOWN_MEAN, backfill_opts=BACKFILL_OPTS
            ),
            t0, FIFOSorter(), seed=0, verbose=args.verbose, min_step=MIN_STEP
        )

    if args.dump_sim_to:
        data = { "archer" : archer }
        data["archer_fcfs"] = archer_fcfs
        with open(args.dump_sim_to, 'wb') as f:
            pickle.dump(data, f)

    if (
        args.scan_low_high_power or args.scan_job_size_weights or
        args.scan_job_size_weights_noise or args.test_frequencies
    ):
        archer_times, times, dates, start, end = {}, {}, {}, {}, {}
        for interval, archer_entry in archer.items():
            archer_times[interval] = archer_entry.times
            start[interval], end[interval] = archer_entry.times[0], archer_entry.times[-1]
            times[interval] = pd.DatetimeIndex(archer_times[interval])
            dates[interval] = matplotlib.dates.date2num(times[interval])
    else:
        archer_times = archer.times
        start, end = archer_times[0], archer_times[-1]
        times = pd.DatetimeIndex(archer_times)
        dates = matplotlib.dates.date2num(times)
    times_fcfs = pd.DatetimeIndex(archer_fcfs.times)
    dates_fcfs = matplotlib.dates.date2num(times_fcfs)

    plots = []
    if args.cab_power_plot:
        plots.append("cab_power_plot")
    if args.plot_v_fcfs:
        plots.append("plot_v_fcfs")
    if args.scan_low_high_power:
        plots.append("scan_plots")
    if args.true_job_start_test:
        plots.append("true_job_start")
    if args.scan_job_size_weights:
        plots.append("scan_size_weights_plots")
    if args.scan_job_size_weights_noise:
        plots.append("scan_size_weights_noise_plots")
    if args.test_frequencies:
        plots.append("test_frequencies")

    if plots:
        plot_blob(
            plots, archer, start, end, times, dates, archer_fcfs=archer_fcfs,
            times_fcfs=times_fcfs, dates_fcfs=dates_fcfs, batch=args.batch, df_jobs=df_jobs,
            save_suffix=args.save_suffix
        )


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("data", type=str)

    scheduler_group = parser.add_mutually_exclusive_group()
    scheduler_group.add_argument(
        "--low_high_power", action="store_true",
        help="Alternate between scheduling lowest power and highest power jobs"
    )
    scheduler_group.add_argument(
        "--scan_low_high_power", action="store_true",
        help="Scan different intervals for low-high power scheduling"
    )
    scheduler_group.add_argument(
        "--true_job_start_test", action="store_true",
        help="Submit jobs as they were in data for a sanity check"
    )
    scheduler_group.add_argument(
        "--scan_job_size_weights", action="store_true",
        help="Scan for different values for the job size factor weight in an age + size" +
             "(priority small jobs) priority"
    )
    scheduler_group.add_argument(
        "--scan_job_size_weights_noise", action="store_true",
        help="Scan for different values for the job size factor weight in an age + size" +
             "(priority small jobs) priority with noise in priority calculation"
    )
    scheduler_group.add_argument(
        "--test_frequencies", action="store_true",
        help="Test setting lower job cpu frequency at different times"
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

    parser.add_argument(
        "--save_suffix", type=str, default="",
        help="Suffix to put with the saved plots"
    )

    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--cache", type=str, default="", help="How to use the cache (save|load)")
    parser.add_argument("--batch", action='store_true')
    parser.add_argument("--verbose", action="store_false")
    parser.add_argument("--dump_sim_to", type=str, default="", help="Pickle sim results")
    parser.add_argument("--read_sim_from", type=str, default="", help="Read pickled sim results")

    args = parser.parse_args()

    if args.rows and args.cache == "load":
        print("Note: rows cannot be set if loading data from cache")

    return args

if __name__ == '__main__':
    main(parse_arguments())


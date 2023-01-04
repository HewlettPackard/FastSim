import argparse, os, joblib, sys
from collections import OrderedDict
import dill as pickle
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib.dates

from classes import Archer2
from fairshare import FairTree
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

class MFPrioritySorter():
    def __init__(
        self, assoc_file, calc_period, decay_halflife, simulation_length, init_time, size_weight,
        age_weight, fairshare_weight, max_age
    ):
        self.size_weight = size_weight
        self.age_weight = age_weight
        self.fairshare_weight = fairshare_weight
        self.max_age = max_age

        self.fairtree = FairTree(
            assoc_file, calc_period, decay_halflife, simulation_length, init_time
        )

    def sort(self, queue, time):
        sorted_queue = sorted(
            queue,
            key=(
                lambda job: (
                    (
                        min((time - job.submit).total_seconds() / self.max_age.total_seconds(), 1)
                        * self.age_weight
                    ) +
                    (
                        (job.nodes / 5860) * self.size_weight
                    ) +
                    (
                        self.fairtree.uniq_users[job.account][job.user].fairshare_factor
                        * self.fairshare_weight
                    )
                )
            ),
            reverse=True
        )
        return sorted_queue


""" End Priority Sorters """

""" Low Freq Response Calcs """

class AppGroupMeanResponses():
    def __init__(self, group_weight_mean_power_time_factor):
        self.group_interval_factors = OrderedDict()
        low = 0
        for weight, factors in group_weight_mean_power_time_factor.items():
            self.group_interval_factors[(low, low + weight)] = factors
            low = low + weight
        if low > 1.0:
            raise ValueError("Weights do not sum to {} (should be 1)".format(low))

    def get_factors(self):
        random_num = np.random.rand()
        for interval in self.group_interval_factors.keys():
            if random_num >= interval[0] and random_num < interval[1]:
                break

        return self.group_interval_factors[interval]


class KDEResponses():
    def __init__(self, model_loc, minmax_time_factor=(1, None), minmax_power_factor=(None, 1)):
        self.kde = joblib.load(model_loc)
        self.minmax_time_factor = minmax_time_factor
        self.minmax_power_factor = minmax_power_factor

    def get_factors(self):
        power_factor, time_factor = self.kde.sample(1)[0]
        power_factor = np.clip(power_factor, *self.minmax_power_factor)
        time_factor = np.clip(time_factor, *self.minmax_time_factor)
        return power_factor, time_factor

""" End Low Freq Response Calcs"""

""" Low Freq Conditions """

# class SmallQueue():
#     def __init__(self, queue_cut):
#         self.queue_cut = queue_cut
#         self.queue_size = 0

#     def __call__(self, queue):
#         small_queue = self.queue_size <= self.queue_cut
#         self.queue_size = len(queue.queue)
#         return small_queue

""" End Low Freq Conditions """


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
                    Archer2(t0, node_down_mean=NODEDOWN_MEAN, backfill_opts=BACKFILL_OPTS),
                    t0, LowHighPowerSorter(switch_interval, t0), seed=0, verbose=args.verbose,
                    min_step=MIN_STEP
                )

        elif args.true_job_start_test:
            print("Running sim for scheduler using job start times from data...")
            archer = run_sim(
                df_jobs,
                Archer2(t0, node_down_mean=0, backfill_opts=BACKFILL_OPTS), t0, DataStartSorter(),
                seed=0, verbose=args.verbose, min_step=MIN_STEP, no_retained=True
            )

        elif args.scan_job_size_weights or args.scan_job_size_weights_noise:
            archer = {}
            # size_weights = [0.01, 0.05, 0.1, 0.5, 1]
            # size_weights = [2, 2.25, 2.5, 3]
            size_weights = [2.0, 2.25, 2.5, 3.0]
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
                        Archer2(t0, node_down_mean=NODEDOWN_MEAN, backfill_opts=BACKFILL_OPTS),
                        t0, AgeSizeSorter(True, size_weight, noise_params=noise_params), seed=0,
                        verbose=args.verbose, min_step=MIN_STEP
                    )
            print("Running sim for scheduler using job start times from data...")
            archer[-1] = run_sim(
                df_jobs, Archer2(t0, node_down_mean=0, backfill_opts=BACKFILL_OPTS), t0,
                DataStartSorter(), seed=0, verbose=args.verbose, min_step=MIN_STEP,
                no_retained=True
            )
            if args.scan_job_size_weights:
                print("Running sim for scheduler with priority small job size")
                archer[999] = run_sim(
                    df_jobs,
                    Archer2(t0, node_down_mean=NODEDOWN_MEAN, backfill_opts=BACKFILL_OPTS),
                    t0, AgeSizeSorter(True, 1.0, age_weight=0.0), seed=0, verbose=args.verbose,
                    min_step=MIN_STEP
                )

        elif args.test_frequencies:
            if args.low_freq_response_calc == "app_means_2ghz":
                low_freq_calc = AppGroupMeanResponses(
                    {
                        0.25 : (0.808709, 1.178027), 0.25 : (0.763424, 1.049210),
                        0.25 : (0.860554, 1.013500), 0.25 : (0.785378, 1.012437)
                    }
                )
                low_freq_reqtime_factor = 1.125 # 2.25 / 2
            elif args.low_freq_response_calc == "app_means_1.5ghz":
                low_freq_calc = AppGroupMeanResponses(
                    {
                        0.25 : (0.734774, 1.551513), 0.25 : (0.730868, 1.100771),
                        0.25 : (0.805355, 1.130684), 0.25 : (0.752383, 1.084178)
                    }
                )
                low_freq_reqtime_factor = 1.5 # 2.25 / 1.5
            elif args.low_freq_response_calc == "kde_2ghz":
                low_freq_calc = KDEResponses(
                    KDE_MODEL_2GHZ, minmax_time_factor=(1, 1.5), minmax_power_factor=(0.5, 1)
                )
                low_freq_reqtime_factor = 1.5 # 2.25 / 2
            elif args.low_freq_response_calc == "kde_1.5ghz":
                low_freq_calc = KDEResponses(
                    KDE_MODEL_1_5GHZ, minmax_time_factor=(1, 1.5), minmax_power_factor=(0.5, 1)
                )
                low_freq_reqtime_factor = 1.5 # 2.25 / 1.5

            archer = {}
            # 2.25, None best for NODEDOWN_MEAN=0 2.5, None best for NODEDOWN_MEAN=100
            size_weight, noise_params = 2.5, None
            small_queue_cuts = [0, 10, 50, 100, 200, 300, 400, 500, 1000, float("inf")]
            for small_queue_cut in small_queue_cuts:
                print(
                    "Running sim for scheduler with age and priority small job size with size" +
                    "weight {} and noise params {}. Setting jobs to 2GHz when quene <= {}".format(
                        size_weight, noise_params, small_queue_cut
                    )
                )
                archer[small_queue_cut] = run_sim(
                    df_jobs,
                    Archer2(
                        t0, node_down_mean=NODEDOWN_MEAN, backfill_opts=BACKFILL_OPTS,
                        low_freq_condition=lambda queue_size: queue_size <= small_queue_cut,
                        low_freq_calc=low_freq_calc,
                        low_freq_reqtime_factor=low_freq_reqtime_factor
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
                Archer2(t0, node_down_mean=NODEDOWN_MEAN, backfill_opts=BACKFILL_OPTS),
                t0, AgeSizeSorter(True, size_weight, noise_params=noise_params), seed=0,
                verbose=args.verbose, min_step=MIN_STEP
            )

        elif args.test_mf_priority:
            archer = {}
            # ARCHER2 defaults
            archer[0] = run_sim(
                df_jobs,
                Archer2(t0, node_down_mean=NODEDOWN_MEAN, backfill_opts=BACKFILL_OPTS),
                t0,
                MFPrioritySorter(
                    ASSOCS_FILE, timedelta(minutes=5), timedelta(days=2), df_jobs.End.max() - t0,
                    t0, 100, 500, 300, timedelta(days=14)
                ),
                verbose=args.verbose, min_step=timedelta(seconds=0), mf_priority_calc_step=True,
                no_retained=True
            )
            archer[1] = run_sim(
                df_jobs,
                Archer2(t0, node_down_mean=NODEDOWN_MEAN, backfill_opts=BACKFILL_OPTS),
                t0,
                MFPrioritySorter(
                    ASSOCS_FILE, timedelta(minutes=5), timedelta(days=2), df_jobs.End.max() - t0,
                    t0, 100, 500, 0, timedelta(days=14)
                ),
                verbose=args.verbose, min_step=timedelta(seconds=0), mf_priority_calc_step=True,
                no_retained=True
            )
            archer[-1] = run_sim(
                df_jobs, Archer2(t0, node_down_mean=0, backfill_opts=BACKFILL_OPTS), t0,
                DataStartSorter(), seed=0, verbose=args.verbose, min_step=MIN_STEP,
                no_retained=True
            )

        else:
            print("Running sim for scheduler low-high_power...")
            archer = run_sim(
                df_jobs,
                Archer2(t0, node_down_mean=NODEDOWN_MEAN, backfill_opts=BACKFILL_OPTS),
                t0, SimpleLowHighPowerSorter(), seed=0, verbose=args.verbose, min_step=MIN_STEP
            )

        print("Running sim for scheduler fcfs...")
        archer_fcfs = run_sim(
            df_jobs,
            Archer2(t0, node_down_mean=NODEDOWN_MEAN, backfill_opts=BACKFILL_OPTS),
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
        for key, archer_entry in archer.items():
            archer_times[key] = archer_entry.times
            start[key], end[key] = archer_entry.times[0], archer_entry.times[-1]
            times[key] = pd.DatetimeIndex(archer_times[key])
            dates[key] = matplotlib.dates.date2num(times[key])
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
    scheduler_group.add_argument(
        "--test_mf_priority", action="store_true",
        help="Test of the mf priority implementation (mainly testing that fairshare is working)"
    )

    parser.add_argument(
        "--low_freq_response_calc", type=str, default="app_means",
        help="(kde_{2,1.5}ghz|app_means_{2,1.5}ghz)"
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


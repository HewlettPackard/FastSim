import argparse, os, pickle, joblib, sys
from datetime import timedelta

from classes import Archer2
from simulation_funcs import prep_job_data, run_sim
from plotting import plot_blob
from globals import * # TODO: dont do this
sys.path.append("/work/y02/y02/awilkins/archer2_jobdata")
from funcs import hour_to_timeofday


""" Priority Sorters """

def fifo_sorter(queue, time):
    return sorted(queue, key=lambda job: job.submit)

def simple_low_high_power_sorter(queue, time):
    if hour_to_timeofday(time.hour) in ["morning", "afternoon", "evening"]:
        return sorted(queue, key=lambda job: job.node_power)
    else:
        return sorted(queue, key=lambda job: job.node_power, reverse=True)

def low_high_power_sorter_factory(switch_interval, t0):
    def low_high_power_sorter(queue, time):
        if (((time - t0) // timedelta(hours=1)) % switch_interval[1][1]) < switch_interval[0][1]:
            return sorted(queue, key=lambda job: job.node_power)
        else:
            return sorted(queue, key=lambda job: job.node_power, reverse=True)
    return low_high_power_sorter
    
""" End Priority Sorters """


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
                    "Running sim for scheduler low-high_power swithching at {} hr \
                     intervals...".format(switch_interval)
                )
                archer[switch_interval] = run_sim(
                    df_jobs,
                    Archer2(
                        t0, baseline_power=BASELINE_POWER, slurmtocab_factor=SLURMTOCAB_FACTOR,
                        node_down_mean=NODEDOWN_MEAN, backfill_opts=BACKFILL_OPTS
                    ),
                    t0, low_high_power_sorter_factory(switch_interval, t0), seed=0,
                    verbose=args.verbose, min_step=MIN_STEP
                )

            print("Running sim for scheduler fcfs...")
            archer[switch_interval] = run_sim(
                df_jobs,
                Archer2(
                    t0, baseline_power=BASELINE_POWER, slurmtocab_factor=SLURMTOCAB_FACTOR,
                    node_down_mean=NODEDOWN_MEAN, backfill_opts=BACKFILL_OPTS
                ),
                t0, fifo_sorter, seed=0, verbose=args.verbose, min_step=MIN_STEP
            )
        else:
            print("Running sim for scheduler low-high_power...")
            archer = run_sim(
                df_jobs,
                Archer2(
                    t0, baseline_power=BASELINE_POWER, slurmtocab_factor=SLURMTOCAB_FACTOR,
                    node_down_mean=NODEDOWN_MEAN, backfill_opts=BACKFILL_OPTS
                ),
                t0, simple_low_high_power_sorter, seed=0, verbose=args.verbose, min_step=MIN_STEP
            )

            print("Running sim for scheduler fcfs...")
            archer_fcfs = run_sim(
                df_jobs,
                Archer2(
                    t0, baseline_power=BASELINE_POWER, slurmtocab_factor=SLURMTOCAB_FACTOR,
                    node_down_mean=NODEDOWN_MEAN, backfill_opts=BACKFILL_OPTS
                ),
                t0, fifo_sorter, seed=0, verbose=args.verbose, min_step=MIN_STEP
            )

    if args.dump_sim_to:
        data = { "archer" : archer }
        data["archer_fcfs"] = archer_fcfs
        with open(args.dump_sim_to, 'wb') as f:
            pickle.dump(data, f)

    if args.scan_low_high_power:
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
            times_fcfs=times_fcfs, dates_fcfs=dates_fcfs, batch=args.batch,
            save_suffix=args.save_suffix
        )
    elif "scan_plots" in plots:
        times_fcfs = pd.DatetimeIndex(archer_fcfs.times)
        dates_fcfs = matplotlib.dates.date2num(times_fcfs)
        plot_blob(
            plots, archer, start, end, times, dates, archer_fcfs=archer_fcfs,
            times_fcfs=times_fcfs, dates_fcfs=dates_fcfs, batch=args.batch, df_jobs=df_jobs,
            save_suffix=args.save_suffix
        )
    else:
        plot_blob(
            plots, archer, start, end, times, dates, batch=args.batch, save_suffix=args.save_suffix
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


import os
from glob import glob
import datetime
from datetime import timedelta
import dill as pickle
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.dates
from matplotlib import pyplot as plt
from tqdm import tqdm

from classes import Archer2, Job
from fairshare import FairTree
from globals import *


""" Helper Plotting Thingys """

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


def interval_shade(ax, start, end, interval):
    for cycle_num in range((end - start) // timedelta(hours=1) // interval[1][1] + 1):
        ax.axvspan(
            matplotlib.dates.date2num(start + timedelta(hours=cycle_num * interval[1][1])),
            matplotlib.dates.date2num(
                start + timedelta(hours=cycle_num * interval[1][1]) +
                timedelta(hours=interval[0][1])
            ),
            label="low power priority" if not cycle_num else "_", color="gray", alpha=0.3
        )
        ax.axvspan(
            matplotlib.dates.date2num(
                start + timedelta(hours=cycle_num * interval[1][1]) +
                timedelta(hours=interval[0][1])
            ),
            matplotlib.dates.date2num(
                start + timedelta(hours=cycle_num * interval[1][1]) +
                timedelta(hours=interval[1][1])
            ),
            label="high power priority" if not cycle_num else "_", color="lightgray", alpha=0.3
        )


def small_queue_shade(ax, queue_history, times, small_queue_cut):
    small_queue_locs = np.array(queue_history) < small_queue_cut
    at_small_queue = small_queue_locs[0]
    small_queue_start, small_queue_end = times[0] if at_small_queue else None, None
    first_low, first_high = True, True
    for tick, at_small_queue in enumerate(small_queue_locs):
        if small_queue_start and at_small_queue:
            continue

        if small_queue_start and not at_small_queue:
            small_queue_end = times[tick - 1]
            ax.axvspan(
                matplotlib.dates.date2num(small_queue_start),
                matplotlib.dates.date2num(small_queue_end),
                label="low frequency" if first_low else "_", color="lightgray", alpha=0.3
            )
            first_low = False
            small_queue_start = None
            continue

        if not small_queue_start and not at_small_queue:
            continue

        if not small_queue_start and at_small_queue:
            small_queue_start = times[tick]
            ax.axvspan(
                matplotlib.dates.date2num(small_queue_end),
                matplotlib.dates.date2num(small_queue_start),
                label="low frequency" if first_high else "_", color="gray", alpha=0.3
            )
            first_high = False
            small_queue_end = None


# Treating slurm to cab as a scaling factor + baseline power of any nodes without jobs runnning
# and so not reported by slurm
def slurm_to_cab(slurm_power, occupancy): # MW, [0,1]
    baseline_power = 1.692
    full_slurm_to_cab = 1.185
    return slurm_power * full_slurm_to_cab + (1 - occupancy) * baseline_power


def get_idle_node_energy(occupancies, times):
    baseline_power = 1.692 * 1e-3 # GW

    idle_energy = 0
    t_deltas = np.diff(times)
    for tick, occupancy in enumerate(occupancies[:-1]):
        idle_energy += (1 - occupancy) * baseline_power  * t_deltas[tick].total_seconds()

    return idle_energy


def bdslowdowns_allocnodes_hist2d(
    archer_true, archer, archer_title, allocnodes_true=[], bd_slowdowns_true=[], clip=1000
):
    bd_slowdowns_true = bd_slowdowns_true if bd_slowdowns_true else (
        archer_true.bd_slowdowns[clip:-clip - 1]
    )
    bd_slowdowns = archer.bd_slowdowns[clip:-clip - 1]

    fig, ax = plt.subplots(1, 2, figsize=(14, 8))

    allocnodes_true = allocnodes_true if allocnodes_true else [
        job.nodes for job in archer_true.job_history[clip:-clip - 1]
    ]
    allocnodes = [ job.nodes for job in archer.job_history[clip:-clip - 1] ]
    bins_allocnodes = np.array(
        list(range(1, 7, 1)) + list(range(8, 17, 2)) + list(range(22, 101, 6)) +
        list(range(200, 1001, 100)) + list(range(1500, 3001, 500))
    )
    bins_bd_slowdowns = np.array(
        [ 1 + 0.5 * i for i in range(9) ] + list(range(6, 11, 1)) + list(range(20, 101, 10)) +
        list(range(150, 301, 50))
    )
    # I know bottom left will be most populated bin
    vmax = max(
        sum([
            1 if (nodes == 1 and slowdown < 2) else 0 for nodes, slowdown in (
                zip(allocnodes_true, bd_slowdowns_true)
            )
        ]),
        sum([
            1 if (nodes == 1 and slowdown < 2) else 0 for nodes, slowdown in (
                zip(allocnodes, bd_slowdowns)
            )
        ])
    )

    ax[0].hist2d(
            allocnodes_true, bd_slowdowns_true, bins=[bins_allocnodes, bins_bd_slowdowns],
            cmap='jet', norm=matplotlib.colors.LogNorm(vmin=1, vmax=vmax)
    )
    ax[0].set_title("ARCHER2 True Start Times")
    ax[0].set_ylabel("Job Bounded Slowdown")

    h = ax[1].hist2d(
        allocnodes, bd_slowdowns, bins=[bins_allocnodes, bins_bd_slowdowns], cmap='jet',
        norm=matplotlib.colors.LogNorm(vmin=1, vmax=vmax)
    )
    ax[1].set_title(archer_title)

    for a in ax:
        a.set_xlabel("AllocNodes")
        a.set_xscale("log")
        a.set_yscale("log")

    fig.tight_layout()

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
    fig.colorbar(h[3], cax=cbar_ax)

    return fig, ax


def bdslowdowns_allocnodes_hist2d_true_fifo_mf(archer_true, archer_fifo, archer_mf):
    fig, ax = plt.subplots(1, 3, figsize=(32, 8))

    bins_allocnodes = np.array(
        list(range(1, 7, 1)) + list(range(8, 17, 2)) + list(range(22, 101, 6)) +
        list(range(200, 1001, 100)) + list(range(1500, 3001, 500))
    )
    bins_bd_slowdowns = np.array(
        [ 1 + 0.5 * i for i in range(9) ] + list(range(6, 11, 1)) + list(range(20, 101, 10)) +
        list(range(150, 301, 50))
    )

    allocnodes_fifo = [ job.nodes for job in archer_fifo.job_history ]
    allocnodes_mf = [ job.nodes for job in archer_mf.job_history ]

    vmax = max(
        sum([
            1 if (nodes == 1 and slowdown < 2) else 0 for nodes, slowdown in (
                zip(allocnodes_true, true_bd_slowdowns)
            )
        ]),
        sum([
            1 if (nodes == 1 and slowdown < 2) else 0 for nodes, slowdown in (
                zip(allocnodes_fifo, archer_fifo.bd_slowdowns)
            )
        ]),
        sum([
            1 if (nodes == 1 and slowdown < 2) else 0 for nodes, slowdown in (
                zip(allocnodes_mf, archer_mf.bd_slowdowns)
            )
        ])
    )

    ax[0].hist2d(
            allocnodes_true, true_bd_slowdowns, bins=[bins_allocnodes, bins_bd_slowdowns],
            cmap='jet', norm=matplotlib.colors.LogNorm(vmin=1, vmax=vmax)
    )
    ax[0].set_title("ARCHER2 Data", fontsize=22)
    ax[0].set_ylabel("Job Bounded Slowdown", fontsize=20)

    h = ax[1].hist2d(
        allocnodes_fifo, archer_fifo.bd_slowdowns, bins=[bins_allocnodes, bins_bd_slowdowns],
        cmap='jet', norm=matplotlib.colors.LogNorm(vmin=1, vmax=vmax)
    )
    ax[1].set_title("FIFO", fontsize=22)

    h = ax[2].hist2d(
        allocnodes_mf, archer_mf.bd_slowdowns, bins=[bins_allocnodes, bins_bd_slowdowns], cmap='jet',
        norm=matplotlib.colors.LogNorm(vmin=1, vmax=vmax)
    )
    ax[2].set_title("Our Simulation", fontsize=22)

    for a in ax:
        a.set_xscale("log")
        a.set_yscale("log")
    ax[1].set_xlabel("Job Num Nodes", fontsize=20)

    fig.tight_layout()

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
    fig.colorbar(h[3], cax=cbar_ax)

    return fig, ax

def bdslowdowns_allocnodes_hist2d_true_fifo_mf_noclass(
    true_bd_slowdowns, true_allocnodes, mf_bd_slowdowns, mf_allocnodes, fifo_bd_slowdowns,
    fifo_allocnodes
):
    fig, ax = plt.subplots(1, 3, figsize=(24, 8))

    bins_allocnodes = np.array(
        list(range(1, 7, 1)) + list(range(8, 17, 2)) + list(range(22, 101, 6)) +
        list(range(200, 1001, 100)) + list(range(1500, 5001, 500))
    )
    bins_bd_slowdowns = np.array(
        [ 1 + 0.5 * i for i in range(9) ] + list(range(6, 11, 1)) + list(range(20, 101, 10)) +
        list(range(150, 301, 50)) + list(range(401, 901, 100))
    )

    vmax = max(
        sum([
            1 if (nodes == 1 and slowdown < 2) else 0 for nodes, slowdown in (
                zip(true_allocnodes, true_bd_slowdowns)
            )
        ]),
        sum([
            1 if (nodes == 1 and slowdown < 2) else 0 for nodes, slowdown in (
                zip(fifo_allocnodes, fifo_bd_slowdowns)
            )
        ]),
        sum([
            1 if (nodes == 1 and slowdown < 2) else 0 for nodes, slowdown in (
                zip(mf_allocnodes, mf_bd_slowdowns)
            )
        ])
    )

    ax[0].hist2d(
            true_allocnodes, true_bd_slowdowns, bins=[bins_allocnodes, bins_bd_slowdowns],
            cmap='jet', norm=matplotlib.colors.LogNorm(vmin=1, vmax=vmax)
    )
    ax[0].set_title("ARCHER2 Data", fontsize=22)
    ax[0].set_ylabel("Job Bounded Slowdown", fontsize=20)

    h = ax[1].hist2d(
        fifo_allocnodes, fifo_bd_slowdowns, bins=[bins_allocnodes, bins_bd_slowdowns],
        cmap='jet', norm=matplotlib.colors.LogNorm(vmin=1, vmax=vmax)
    )
    ax[1].set_title("FIFO", fontsize=22)

    h = ax[2].hist2d(
        mf_allocnodes, mf_bd_slowdowns, bins=[bins_allocnodes, bins_bd_slowdowns], cmap='jet',
        norm=matplotlib.colors.LogNorm(vmin=1, vmax=vmax)
    )
    ax[2].set_title("Our Simulation", fontsize=22)

    for a in ax:
        a.set_xscale("log")
        a.set_yscale("log")
    ax[1].set_xlabel("Job Num Nodes", fontsize=20)

    fig.tight_layout()

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
    fig.colorbar(h[3], cax=cbar_ax)

    return fig, ax

def update_job_class_as_required(archer, archer_fcfs):
    for system in archer.values():
        if not hasattr(system.job_history[0], "launch_time"):
            system.job_history = [
                Job(
                    job.id if hasattr(job, "id") else "None", job.submit, job.nodes, job.runtime,
                    job.reqtime, job.node_power, job.true_node_power, job.true_job_start,
                    job.user if hasattr(job, "user") else "None",
                    job.account if hasattr(job, "account") else "None",
                    job.qos if hasattr(job, "qos") else "None",
                    job.partition if hasattr(job, "partition") else "None", "None", "None"
                ) for job in system.job_history
            ]

    if not hasattr(archer_fcfs.job_history[0], "launch_time"):
        archer_fcfs.job_history = [
            Job(
                job.id if hasattr(job, "id") else "None", job.submit, job.nodes, job.runtime,
                job.reqtime, job.node_power, job.true_node_power, job.true_job_start,
                job.user if hasattr(job, "user") else "None",
                job.account if hasattr(job, "account") else "None",
                job.qos if hasattr(job, "qos") else "None",
                job.partition if hasattr(job, "partition") else "None", "", "None"
            ) for job in archer_fcfs.job_history
        ]

""" End Helper Plotting Thingys """


def plot_blob(
    plots, archer, start, end, times, dates, archer_fcfs=None, times_fcfs=None, dates_fcfs=None,
    save_suffix="", batch=False, df_jobs=None
):
    # For legacy experiments - I ballsed this up and don't need it anymore since not using launch
    # time for slowdown calculations
    # update_job_class_as_required(archer, archer_fcfs)

    if "cab_power_plot" in plots:
        cabs = {
            datetime.datetime.strptime(os.path.basename(path), "system_%y%m%d") : path
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

            t0 = archer_entry.init_time
            if type(interval) == int:
                low_or_high = lambda time: (
                    "low" if (
                        (((time - t0) // timedelta(hours=1)) // interval) % 2 == 0
                    )
                    else "high"
                )
            elif type(interval) == tuple:
                low_or_high = lambda time: (
                    "low" if (
                        (((time - t0) // timedelta(hours=1)) % interval[1][1]) < interval[0][1]
                    )
                    else "high"
                )

            low_powers, high_powers = [], []
            power = archer_entry.power_history[1000:-1000]
            occupancy = archer_entry.occupancy_history[1000:-1000]
            # baseline_power = 1.692
            for tick, date in enumerate(times[interval][1000:-1000]):
                if low_or_high(date) == "low":
                    low_powers.append(slurm_to_cab(power[tick], occupancy[tick]))
                    # low_powers.append(power[tick] + (1 - occupancy[tick]) * baseline_power)
                else:
                    high_powers.append(slurm_to_cab(power[tick], occupancy[tick]))
                    # high_powers.append(power[tick] + (1 - occupancy[tick]) * baseline_power)

            mean_low_power = np.mean(low_powers)
            mean_high_power = np.mean(high_powers)

            x, y = bd_slowdown, (mean_high_power - mean_low_power) * 1000
            ax.scatter(x, y, s=64)
            ax.annotate(
                "{}hr on {}hr off".format(interval[0][1], interval[1][1] - interval[1][0]),
                (x + 0.025,y), fontsize=12
            )

        # shift mean power measurement window by some hours
        hr_shift = 0
        if hr_shift:
            for interval, archer_entry in archer.items():
                bd_slowdown = np.mean(archer_entry.bd_slowdowns[1000:-1000])

                t0 = archer_entry.init_time
                if type(interval) == int:
                    low_or_high = lambda time: (
                        "low" if (
                            (
                                (((time - t0) // timedelta(hours=1)) % interval) > hr_shift and
                                (((time - t0) // timedelta(hours=1)) // interval) % 2 == 0
                            )
                            or
                            (
                                (((time - t0) // timedelta(hours=1)) % interval) <= hr_shift and
                                (((time - t0) // timedelta(hours=1)) // interval) % 2 == 1
                            )
                        )
                        else "high"
                    )
                elif type(interval) == tuple:
                    low_or_high = lambda time: (
                        "low" if (
                            (
                                hr_shift <
                                (((time - t0) // timedelta(hours=1)) % interval[1][1]) <
                                interval[0][1]
                            )
                            or (
                                interval[0][1] <=
                                (((time - t0) // timedelta(hours=1)) % interval[1][1]) <=
                                interval[0][1] + hr_shift
                            )
                        )
                        else "high"
                    )

                low_powers, high_powers = [], []
                power = archer_entry.power_history[1000:-1000]
                occupancy = archer_entry.occupancy_history[1000:-1000]
                # baseline_power = 1.692
                for tick, date in enumerate(times[interval][1000:-1000]):
                    if low_or_high(date) == "low":
                        low_powers.append(slurm_to_cab(power[tick], occupancy[tick]))
                        # low_powers.append(power[tick] + (1 - occupancy[tick]) * baseline_power)
                    else:
                        high_powers.append(slurm_to_cab(power[tick], occupancy[tick]))
                        # high_powers.append(power[tick] + (1 - occupancy[tick]) * baseline_power)

                mean_low_power = np.mean(low_powers)
                mean_high_power = np.mean(high_powers)

                x, y = bd_slowdown, (mean_high_power - mean_low_power) * 1000
                ax.scatter(x, y, s=64)
                ax.annotate(
                    "{}hr on {}hr off ({}hr delay)".format(
                        interval[0][1], interval[1][1] - interval[1][0], hr_shift
                    ),
                    (x + 0.025,y), fontsize=12
                )

        # ARCHER2 data baseline
        bd_slowdowns_data = [
            max(
                (job_row.End - job_row.Submit)/max(job_row.Elapsed, BD_THRESHOLD), 1
            ) for _, job_row in df_jobs.iterrows()
        ][1000:-1000]
        x, y = np.mean(bd_slowdowns_data), 0
        ax.scatter(x, y, s=64, c='k')
        ax.annotate("ARCHER2 Data", (x + 0.025,y), fontsize=12)

        # FIFO baseline
        bd_slowdown = np.mean(archer_fcfs.bd_slowdowns[1000:-1000])
        x, y = bd_slowdown, 0
        ax.scatter(x, y, s=64, c='k')
        ax.annotate("FIFO", (x + 0.025,y), fontsize=12)

        ax.set_ylabel("MeanHighPower - MeanLowPower (kW)")
        ax.set_xlabel("Mean Bounded Slowdown")
        plt.grid()
        fig.tight_layout()
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

        # Examine intervals I want to see
        interval = ((0,24),(24,72))
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.plot_date(dates[interval], archer[interval].power_history, 'g', linewidth=0.6)
        interval_shade(ax, start[interval], end[interval], interval)
        ax.set_ylabel("Power (MW)")
        plt.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(
            PLOT_DIR, "toyscheduler_low-high_power_0242472_scan{}.pdf".format(save_suffix)
        ))
        if batch:
            plt.close()
        else:
            plt.show()

        interval = ((0,24),(24,48))
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.plot_date(dates[interval], archer[interval].power_history, 'g', linewidth=0.6)
        interval_shade(ax, start[interval], end[interval], interval)
        ax.set_ylabel("Power (MW)", fontsize=20)
        ax.xaxis.label.set_size(18)
        ax.yaxis.label.set_size(18)
        plt.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(
            PLOT_DIR, "toyscheduler_low-high_power_0242448_scan{}.pdf".format(save_suffix)
        ))
        if batch:
            plt.close()
        else:
            plt.show()

        interval = ((0,24),(24,48)) # This looks the clearest
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.plot_date(dates[interval], archer[interval].power_history, 'g', linewidth=0.6)
        interval_shade(ax, start[interval], end[interval], interval)
        ax.set_xlim(datetime.date(2022, 11, 22), datetime.date(2022, 12, 2))
        ax.set_ylim(2.0, 3.2)
        ax.set_ylabel("Power (MW)", fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        plt.legend(fontsize=18)
        fig.tight_layout()
        fig.savefig(os.path.join(
            PLOT_DIR, "toyscheduler_low-high_power_0242448_zoomedin_scan{}.pdf".format(save_suffix)
        ))
        if batch:
            plt.close()
        else:
            plt.show()

        interval = ((0,12),(12,24))
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.plot_date(dates[interval], archer[interval].power_history, 'g', linewidth=0.6)
        interval_shade(ax, start[interval], end[interval], interval)
        ax.set_xlim(datetime.date(2022, 11, 22), datetime.date(2022, 12, 2))
        ax.set_ylim(2.0, 3.2)
        ax.set_ylabel("Power (MW)", fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.tick_params(axis='both', which='minor', labelsize=14)
        plt.legend(fontsize=18)
        fig.tight_layout()
        fig.savefig(os.path.join(
            PLOT_DIR, "toyscheduler_low-high_power_0121224_zoomedin_scan{}.pdf".format(save_suffix)
        ))
        fig.savefig(
            os.path.join(
                PLOT_DIR,
                "toyscheduler_low-high_power_0121224_zoomedin_scan_fuckppt{}.png".format(
                    save_suffix
                ),
            ),
            dpi=600
        )
        if batch:
            plt.close()
        else:
            plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        print("max(archer[{}].bd_slowdowns[1000:-1000])={}".format(
            interval, max(archer[interval].bd_slowdowns[1000:-1000])
        ))
        print("max(bd_slowdowns_data)={}".format(max(bd_slowdowns_data)))
        ax.hist(
            archer[interval].bd_slowdowns[1000:-1000],
            bins=int(max(archer[interval].bd_slowdowns)),
            range=(min(archer[interval].bd_slowdowns),max(archer[interval].bd_slowdowns)),
            histtype="step", label="My scheduler"
        )
        ax.hist(
            bd_slowdowns_data, bins=int(max(archer[interval].bd_slowdowns)),
            range=(min(archer[interval].bd_slowdowns),max(archer[interval].bd_slowdowns)),
            histtype="step", label="data"
        )
        plt.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(
            PLOT_DIR,
            "toyscheduler_low-high_power_0242448_data_bdslowdowns_scan{}.pdf".format(save_suffix)
        ))
        if batch:
            plt.close()
        else:
            plt.show()

        allocnodes_true = [ job_row.AllocNodes for _, job_row in df_jobs.iterrows() ][1000:-1000]
        fig, ax = bdslowdowns_allocnodes_hist2d(
            None, archer[interval], "{}hr on {}hr off".format(interval[0][1], interval[1][1]),
            allocnodes_true=allocnodes_true, bd_slowdowns_true=bd_slowdowns_data
        )
        fig.savefig(os.path.join(
            PLOT_DIR,
            "toyscheduler_low-high_power_0242448_data_bdslowdowns_allocnodesscan{}.pdf".format(
                save_suffix
            )
        ))
        if batch:
            plt.close()
        else:
            plt.show()

        interval = ((0,12),(12,48))
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.plot_date(dates[interval], archer[interval].power_history, 'g', linewidth=0.6)
        interval_shade(ax, start[interval], end[interval], interval)
        ax.set_ylabel("Power (MW)")
        plt.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(
            PLOT_DIR, "toyscheduler_low-high_power_0121248_scan{}.pdf".format(save_suffix)
        ))
        if batch:
            plt.close()
        else:
            plt.show()

    if "true_job_start" in plots:
        print("fifo mean bd slowdown={}+-{} true start mean bd slowdown={}+-{}".format(
            np.mean(archer_fcfs.bd_slowdowns), np.std(archer_fcfs.bd_slowdowns),
            np.mean(archer.bd_slowdowns), np.std(archer.bd_slowdowns)
        ))
        print("fifo mean utilisation={}+-{} true start mean utilisation={}+-{}".format(
            np.mean(archer_fcfs.occupancy_history), np.std(archer_fcfs.occupancy_history),
            np.mean(archer.occupancy_history), np.std(archer.occupancy_history)
        ))

        fig, ax = bdslowdowns_allocnodes_hist2d(archer, archer_fcfs, "FIFO")
        fig.savefig(os.path.join(
            PLOT_DIR,
            "toyscheduler_test_true_starts_bdslowdowns_allocnodes_{}.pdf".format(save_suffix)
        ))
        if batch:
            plt.close()
        else:
            plt.show()

    if "scan_size_weights_plots" in plots:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        x = np.arange(0, len(archer) + 1)
        x_labels, bd_slowdowns, bd_slowdowns_err = [], [], []
        for size_weight, archer_entry in archer.items():
            if size_weight == -1:
                x_labels.append("true data")
            elif size_weight == 999:
                x_labels.append("smallest first")
            else:
                x_labels.append(size_weight)
            bd_slowdowns.append(np.mean(archer_entry.bd_slowdowns[1000:-1000]))
            bd_slowdowns_err.append(np.std(archer_entry.bd_slowdowns[1000:-1000]))
        x_labels.append("fifo")
        bd_slowdowns.append(np.mean(archer_fcfs.bd_slowdowns[1000:-1000]))
        bd_slowdowns_err.append(np.std(archer_fcfs.bd_slowdowns[1000:-1000]))
        x_labels = [
            label for label, _ in sorted(zip(x_labels, bd_slowdowns), key=lambda pair: pair[1])
        ]
        bd_slowdowns_err = [
            err for err, _ in sorted(zip(bd_slowdowns_err, bd_slowdowns), key=lambda pair: pair[1])
        ]
        bd_slowdowns.sort()
        print(x_labels, bd_slowdowns, bd_slowdowns_err, sep='\n')

        ax.bar(x, bd_slowdowns, yerr=bd_slowdowns_err)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.grid(axis="y")
        ax.set_ylabel("Mean bounded slowdown")
        fig.tight_layout()
        fig.savefig(os.path.join(
            PLOT_DIR,
            "toyscheduler_priority_small_and_age_bdslowdowns_scan{}.pdf".format(save_suffix)
        ))
        if batch:
            plt.close()
        else:
            plt.show()


        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.bar(x, bd_slowdowns)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.grid(axis="y")
        ax.set_ylabel("Mean bounded slowdown")
        fig.tight_layout()
        fig.savefig(os.path.join(
            PLOT_DIR,
            "toyscheduler_priority_small_and_age_bdslowdowns_scan_noerrs{}.pdf".format(save_suffix)
        ))
        if batch:
            plt.close()
        else:
            plt.show()

        weight_to_compare = 2.25
        fig, ax = bdslowdowns_allocnodes_hist2d(
            archer[-1], archer[weight_to_compare],
            "Age and Priority Small (size weight {})".format(weight_to_compare)
        )
        fig.savefig(os.path.join(
            PLOT_DIR,
            "toyscheduler_priority_small_and_age_bdslowdowns_allocnodes_weight1{}.pdf".format(
                save_suffix
            )
        ))
        if batch:
            plt.close()
        else:
            plt.show()

    if "scan_size_weights_noise_plots" in plots:
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))

        x = np.arange(0, len(archer) + 1)
        x_labels, bd_slowdowns, bd_slowdowns_err = [], [], []
        for size_noise_weight, archer_entry in archer.items():
            if size_noise_weight == -1:
                x_labels.append("true data")
            else:
                x_labels.append(size_noise_weight)
            bd_slowdowns.append(np.mean(archer_entry.bd_slowdowns[1000:-1000]))
            bd_slowdowns_err.append(np.std(archer_entry.bd_slowdowns[1000:-1000]))
        x_labels.append("fifo")
        bd_slowdowns.append(np.mean(archer_fcfs.bd_slowdowns[1000:-1000]))
        bd_slowdowns_err.append(np.std(archer_fcfs.bd_slowdowns[1000:-1000]))
        x_labels = [
            label for label, _ in sorted(zip(x_labels, bd_slowdowns), key=lambda pair: pair[1])
        ]
        bd_slowdowns_err = [
            err for err, _ in sorted(zip(bd_slowdowns_err, bd_slowdowns), key=lambda pair: pair[1])
        ]
        bd_slowdowns.sort()
        print(x_labels, bd_slowdowns, bd_slowdowns_err, sep='\n')

        ax.bar(x, bd_slowdowns, yerr=bd_slowdowns_err)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.grid(axis="y")
        ax.set_ylabel("Mean bounded slowdown")
        fig.tight_layout()
        fig.savefig(os.path.join(
            PLOT_DIR,
            "toyscheduler_priority_small_and_age_noise_bdslowdowns_scan{}.pdf".format(save_suffix)
        ))
        if batch:
            plt.close()
        else:
            plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.bar(x, bd_slowdowns)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.grid(axis="y")
        ax.set_ylabel("Mean bounded slowdown")
        fig.tight_layout()
        fig.savefig(os.path.join(
            PLOT_DIR,
            "toyscheduler_priority_small_and_age_noise_bdslowdowns_scan_noerrs{}.pdf".format(
                save_suffix
            )
        ))
        if batch:
            plt.close()
        else:
            plt.show()

        fig, ax = bdslowdowns_allocnodes_hist2d(
            archer[-1], archer[(2.25, 2)], "Age and Priority Small (size weight 1)"
        )
        fig.savefig(os.path.join(
            PLOT_DIR,
            "toyscheduler_priority_small_and_age_noise_bdslowdowns_allocnodes_sizeweight225" +
            "noiseweight2{}.pdf".format(save_suffix)
        ))
        if batch:
            plt.close()
        else:
            plt.show()

    if "test_frequencies" in plots:
        total_energy_nolowfreq = (
            archer[-1].total_energy +
            get_idle_node_energy(archer[-1].occupancy_history, archer[-1].times)
        )
        for queue_cut, archer_entry in archer.items():
            power = [
                slurm_to_cab(power, archer_entry.occupancy_history[tick]) for tick, power in (
                    enumerate(archer_entry.power_history)
                )
            ]
            total_energy = (
                archer_entry.total_energy +
                get_idle_node_energy(archer_entry.occupancy_history, archer_entry.times)
            )
            print(
                queue_cut, np.mean(archer_entry.bd_slowdowns), np.mean(power), total_energy,
                total_energy / total_energy_nolowfreq * 100, sep=" | "
            )

            fig = plt.figure(1, figsize=(12, 8))
            ax = fig.add_axes((.1, .3, .8, .6))
            ax.plot_date(
                dates[queue_cut], power, 'g', label="cab power (converted from slurm)",
                linewidth=0.6
            )
            # small_queue_shade(ax, archer_entry.queue_size_history, archer_entry.times, queue_cut)
            ax.set_ylabel("Power (MW)")
            ax.set_xticklabels([])
            ax.set_title("Power Usage with {} Queue Cut for Low Frequency Jobs".format(queue_cut))
            plt.legend()
            ax2 = fig.add_axes((.1, .1, .8, .2))
            ax2.plot_date(
                dates[queue_cut], archer_entry.queue_size_history, 'k', label="queue size",
                linewidth=0.6
            )
            ax2.axhline(queue_cut, c='r', linewidth=0.5)
            plt.legend()
            fig.tight_layout()
            save_queue_cuts = [200]
            if queue_cut in save_queue_cuts:
                fig.savefig(os.path.join(
                    PLOT_DIR,
                    (
                        "toyscheduler_priority_small_and_age_lowfreq_" +
                        "queuecut{}_power_queue{}.pdf".format(queue_cut, save_suffix)
                    )
                ))
            if batch:
                plt.close()
            else:
                plt.close()
                # plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        x = np.arange(0, len(archer))
        x_labels, bd_slowdowns, bd_slowdowns_err = [], [], []
        for small_queue_cut, archer_entry in archer.items():
            if small_queue_cut == -1:
                x_labels.append("no low freq")
            else:
                x_labels.append(str(small_queue_cut))
            bd_slowdowns.append(np.mean(archer_entry.bd_slowdowns))
            bd_slowdowns_err.append(np.std(archer_entry.bd_slowdowns))
        x_labels = [
            label for label, _ in sorted(zip(x_labels, bd_slowdowns), key=lambda pair: pair[1])
        ]
        bd_slowdowns_err = [
            err for err, _ in sorted(zip(bd_slowdowns_err, bd_slowdowns), key=lambda pair: pair[1])
        ]
        bd_slowdowns.sort()
        print(x_labels, bd_slowdowns, bd_slowdowns_err, sep='\n')

        ax.bar(x, bd_slowdowns, yerr=bd_slowdowns_err)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.grid(axis="y")
        ax.set_ylabel("Mean bounded slowdown")
        fig.tight_layout()
        fig.savefig(os.path.join(
            PLOT_DIR,
            "toyscheduler_priority_small_and_age_lowfreq_queuecut_bdslowdowns{}.pdf".format(
                save_suffix
            )
        ))
        if batch:
            plt.close()
        else:
            # plt.show()
            plt.close()

        # Consider 95th quantile start instead of max
        baseline_energy = total_energy_nolowfreq
        baseline_avg_slowdown = np.mean(archer[-1].bd_slowdowns)
        wait_times = [
            (job.start - job.submit).total_seconds() for job in archer[-1].job_history if (
                job.submit >= start[-1]
            )
        ]
        baseline_avg_wait, baseline_max_wait = np.mean(wait_times), max(wait_times)
        responses = [
            (job.end - job.submit).total_seconds() for job in archer[-1].job_history if (
                job.submit >= start[-1]
            )
        ]
        baseline_avg_response = np.mean(responses)

        spider_plot_data = {}
        for queue_cut in [0, 100, 200, float("inf")]:
            archer_entry = archer[queue_cut]
            energy = (
                (
                    archer_entry.total_energy +
                    get_idle_node_energy(archer_entry.occupancy_history, archer_entry.times)
                ) /
                baseline_energy
            )
            avg_slowdown = np.mean(archer_entry.bd_slowdowns) / baseline_avg_slowdown
            wait_times = [
                (job.start - job.submit).total_seconds() for job in archer_entry.job_history if (
                    job.submit >= start[queue_cut]
                )
            ]
            avg_wait = np.mean(wait_times) / baseline_avg_wait
            max_wait = max(wait_times) / baseline_max_wait
            responses = [
                (job.end - job.submit).total_seconds() for job in archer_entry.job_history if (
                    job.submit >= start[-1]
                )
            ]
            avg_response = np.mean(responses) / baseline_avg_response

            spider_plot_data[queue_cut] = {
                "energy" : energy, "avg_wait" : avg_wait, "max_wait" : max_wait,
                "avg_slowdown" : avg_slowdown, "avg_response" : avg_response
            }

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, polar=True)

        categories = ["energy", "avg_slowdown", "avg_wait", "max_wait", "avg_response"]
        angles = [ i / float(len(categories)) * 2 * np.pi for i in range(len(categories)) ]
        angles += angles[:1]

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        category_labels = [
            r"$(\mathrm{energy})^{-1}$", r"$(\mathrm{avg\_slowdown})^{-1}$",
            r"$(\mathrm{avg\_wait})^{-1}$", r"$(\mathrm{max\_wait})^{-1}$",
            r"$(\mathrm{avg\_response})^{-1}$"
        ]
        plt.xticks(angles[:-1], category_labels, size=14)
        for label in ax.get_xticklabels():
            if label.get_text() == r"$(\mathrm{energy})^{-1}$":
                label.set_verticalalignment("bottom")
            if label.get_text() == r"$(\mathrm{avg\_slowdown})^{-1}$":
                label.set_verticalalignment("bottom")
                label.set_horizontalalignment("left")
            elif label.get_text() == r"$(\mathrm{avg\_wait})^{-1}$":
                label.set_verticalalignment("top")
                label.set_horizontalalignment("left")
            elif label.get_text() == r"$(\mathrm{max\_wait})^{-1}$":
                label.set_verticalalignment("top")
                label.set_horizontalalignment("right")
            elif label.get_text() == r"$(\mathrm{avg\_response})^{-1}$":
                label.set_verticalalignment("bottom")
                label.set_horizontalalignment("right")
        ax.tick_params(axis='x', which='major', pad=10)
        ax.set_rlabel_position(0)
        plt.yticks([0.75,1,1.25], ["0.75","1","1.25"], color="grey", size=14)
        plt.ylim(0.7,1.3)

        for queue_cut, perf_data in spider_plot_data.items():
            vals = [ 1 / perf_data[metric] for metric in categories ]
            vals += vals[:1]
            colour = next(ax._get_lines.prop_cycler)["color"]
            ax.plot(angles, vals, linewidth=2, linestyle='solid', c=colour, label=str(queue_cut))

        ax.plot(
            angles, [1] * len(angles), linewidth=3, linestyle='solid', c='k', label="no low freq"

        )
        plt.legend(loc='center', bbox_to_anchor=(0.5, -0.075), fontsize=16, ncol=5)
        plt.subplots_adjust(left=0.0, top=0.9, right=1.00, bottom=0.1)
        fig.savefig(os.path.join(
            PLOT_DIR,
            "toyscheduler_priority_small_and_age_lowfreq_queuecut_perfspiderplt{}.pdf".format(
                save_suffix
            )
        ))
        fig.savefig(
            os.path.join(
                PLOT_DIR,
                (
                    "toyscheduler_priority_small_and_age_lowfreq_queuecut_perfspiderplt_" +
                    "fuckppt{}.png".format(save_suffix)
                )
            ),
            dpi=600
        )
        if batch:
            plt.close()
        else:
            plt.show()

    if "test_mf_priority" in plots:
        data_bd_slowdowns = [
            max(
                (job.true_job_start + job.runtime - job.submit) / max(job.runtime, BD_THRESHOLD),
                1
            ) for job in archer[0].job_history
        ]
        print(
            "true start sorter mean bd slowdown={}+-{}\n".format(
                np.mean(archer[-1].bd_slowdowns), np.std(archer[-1].bd_slowdowns)
            ) +
            "true starts mean bd slowdown={}+-{}\n".format(
                np.mean(data_bd_slowdowns), np.std(data_bd_slowdowns)
            ) +
            "MF priority w/ fairshare mean bd slowdown={}+-{}\n".format(
                np.mean(archer[0].bd_slowdowns), np.std(archer[0].bd_slowdowns)
            ) +
            "MF priority w/o fairshare mean bd slowdown={}+-{}".format(
                np.mean(archer[1].bd_slowdowns), np.std(archer[1].bd_slowdowns)
            )
        )

        fig, ax = bdslowdowns_allocnodes_hist2d(
            archer[-1], archer[0], "MF Priority w/ Fairshare", clip=0
        )
        fig.savefig(os.path.join(
            PLOT_DIR,
            "toyscheduler_test_mf_priority_fairshare_bdslowdowns_allocnodes{}.pdf".format(
                save_suffix
            )
        ))
        if batch:
            plt.close()
        else:
            plt.show()

        fig, ax = bdslowdowns_allocnodes_hist2d(
            archer[-1], archer[1], "MF Priority w/o Fairshare", clip=0
        )
        fig.savefig(os.path.join(
            PLOT_DIR,
            "toyscheduler_test_mf_priority_nofairshare_bdslowdowns_allocnodes{}.pdf".format(
                save_suffix
            )
        ))
        if batch:
            plt.close()
        else:
            plt.show()

        print("Reading baseline fifo sim results")
        with open("/work/y02/y02/awilkins/pandas_cache/toy_scheduler/fifo_baseline.pkl", "rb") as f:
            data = pickle.load(f)
        archer_fifo = data["archer"][0]

        print(
            "Baseline FIFO mean bd slowdown={}+-{}".format(
                np.mean(archer_fifo.bd_slowdowns), np.std(archer_fifo.bd_slowdowns)
            )
        )

        fig, ax = bdslowdowns_allocnodes_hist2d_true_fifo_mf(archer[-1], archer_fifo, archer[0])
        fig.savefig(os.path.join(
            PLOT_DIR,
            (
                "toyscheduler_test_mf_priority_fairshare_withfifobaseline_bdslowdowns_" +
                "allocnodes{}.pdf".format(
                    save_suffix
                )
            )
        ))
        fig.savefig(
            os.path.join(
                PLOT_DIR,
                (
                    "toyscheduler_test_mf_priority_fairshare_withfifobaseline_bdslowdowns_" +
                    "allocnodes_fuckppt{}.png".format(
                        save_suffix
                    )
                )
            ),
            dpi=600
        )
        if batch:
            plt.close()
        else:
            plt.show()

        wait_times_true = {
            job.id : (job.true_job_start - job.submit).total_seconds() for job in (
                archer[0].job_history
            )
        }
        wait_times_true_sorter = {
            job.id : (job.start - job.submit).total_seconds() for job in archer[-1].job_history
        }
        wait_times_fairshare = {
            job.id : (job.start - job.submit).total_seconds() for job in archer[0].job_history
        }
        wait_times_nofairshare = {
            job.id : (job.start - job.submit).total_seconds() for job in archer[1].job_history
        }
        print("MF priority w/ fairshare wait time distance with true start times={}hrs".format(
            np.sqrt(sum([
                (wait_times_true[id] - wait_times_fairshare[id])**2 for id in (
                    wait_times_true.keys()
                )
            ])) /
            60**2
        ))
        print("MF priority w/o fairshare wait time distance with true start times={}hrs".format(
            np.sqrt(sum([
                (wait_times_true[id] - wait_times_nofairshare[id])**2 for id in (
                    wait_times_true.keys()
                )
            ])) /
            60**2
        ))

        print(
            "===Repeating but now cutting out first 28 days of jobs in order to somewhat clear" +
            " any effect from unkown initial usages==="
        )

        start_time = archer[0].init_time + timedelta(days=28)
        for key in archer.keys():
            archer[key].job_history = [
                job for job in archer[key].job_history if job.submit > start_time
            ]
            archer[key].bd_slowdowns = [
                max((job.end - job.submit) / max(job.runtime, BD_THRESHOLD), 1) for job in (
                    archer[key].job_history
                )
            ]

        data_bd_slowdowns = [
            max(
                (job.true_job_start + job.runtime - job.submit) / max(job.runtime, BD_THRESHOLD),
                1
            ) for job in archer[0].job_history
        ]
        print(
            "true start sorter mean bd slowdown={}+-{}\n".format(
                np.mean(archer[-1].bd_slowdowns), np.std(archer[-1].bd_slowdowns)
            ) +
            "true starts mean bd slowdown={}+-{}\n".format(
                np.mean(data_bd_slowdowns), np.std(data_bd_slowdowns)
            ) +
            "MF priority w/ fairshare mean bd slowdown={}+-{}\n".format(
                np.mean(archer[0].bd_slowdowns), np.std(archer[0].bd_slowdowns)
            ) +
            "MF priority w/o fairshare mean bd slowdown={}+-{}".format(
                np.mean(archer[1].bd_slowdowns), np.std(archer[1].bd_slowdowns)
            )
        )

        fig, ax = bdslowdowns_allocnodes_hist2d(archer[-1], archer[0], "MF Priority w/ Fairshare")
        fig.savefig(os.path.join(
            PLOT_DIR,
            "toyscheduler_test_mf_priority_fairshare_bdslowdowns_allocnodes_skip28days" +
            "{}.pdf".format(
                save_suffix
            )
        ))
        if batch:
            plt.close()
        else:
            plt.show()

        fig, ax = bdslowdowns_allocnodes_hist2d(archer[-1], archer[1], "MF Priority w/o Fairshare")
        fig.savefig(os.path.join(
            PLOT_DIR,
            "toyscheduler_test_mf_priority_nofairshare_bdslowdowns_allocnodes_skip28days" +
            "{}.pdf".format(
                save_suffix
            )
        ))
        if batch:
            plt.close()
        else:
            plt.show()

        archer_fifo.job_history = [
            job for job in archer_fifo.job_history if job.submit > start_time
        ]
        archer_fifo.bd_slowdowns = [
            max((job.end - job.submit) / max(job.runtime, BD_THRESHOLD), 1) for job in (
                archer_fifo.job_history
            )
        ]

        fig, ax = bdslowdowns_allocnodes_hist2d_true_fifo_mf(archer[-1], archer_fifo, archer[0])
        fig.savefig(os.path.join(
            PLOT_DIR,
            (
                "toyscheduler_test_mf_priority_fairshare_withfifobaseline_bdslowdowns_" +
                "allocnodes_skip28days{}.pdf".format(
                    save_suffix
                )
            )
        ))
        fig.savefig(
            os.path.join(
                PLOT_DIR,
                (
                    "toyscheduler_test_mf_priority_fairshare_withfifobaseline_bdslowdowns_" +
                    "allocnodes_skip28ays_fuckppt{}.png".format(
                        save_suffix
                    )
                )
            ),
            dpi=600
        )
        if batch:
            plt.close()
        else:
            plt.show()

        wait_times_true = {
            job.id : (job.true_job_start - job.submit).total_seconds() for job in (
                archer[0].job_history
            )
        }
        wait_times_true = {
            job.id : (job.start - job.submit).total_seconds() for job in archer[-1].job_history
        }
        wait_times_fairshare = {
            job.id : (job.start - job.submit).total_seconds() for job in archer[0].job_history
        }
        wait_times_nofairshare = {
            job.id : (job.start - job.submit).total_seconds() for job in archer[1].job_history
        }
        print("MF priority w/ fairshare wait time distance with true start times={}hrs".format(
            np.sqrt(sum([
                (wait_times_true[id] - wait_times_fairshare[id])**2 for id in (
                    wait_times_true.keys()
                )
            ])) /
            60**2
        ))
        print("MF priority w/o fairshare wait time distance with true start times={}hrs".format(
            np.sqrt(sum([
                (wait_times_true[id] - wait_times_nofairshare[id])**2 for id in (
                    wait_times_true.keys()
                )
            ])) /
            60**2
        ))

    if "test_refactor" in plots:
        # TODO Redo this baseline fifo to not count the ignore in eval jobs, this will require
        # turning off partitions, QOS, and dependencies. This will be easier once a lot of this
        # is moved in a consts global. Wait to do this so I dont have to hack away at code
        print("Reading baseline fifo sim results")
        with open("/work/y02/y02/awilkins/pandas_cache/toy_scheduler/fifo_baseline.pkl", "rb") as (
            f
        ):
            data = pickle.load(f)
        archer_fifo = data["archer"][0]

        max_submit = max(archer[0].job_history, key=lambda job: job.submit).submit
        job_history = [
            job for job in archer[0].job_history if (
                archer[0].init_time + timedelta(days=4) < job.submit <
                max_submit - timedelta(days=4)
            )
        ]

        print(
            "Ignoring {} out of {} jobs in evaulation\n".format(
                sum(1 for job in job_history if job.ignore_in_eval), len(job_history)
            )
        )

        data_bd_slowdowns = [
            max(
                (job.true_job_start + job.runtime - job.true_submit) / max(job.runtime, BD_THRESHOLD),
                1
            ) for job in job_history if not job.ignore_in_eval
        ]
        sim_bd_slowdowns = [
            max(
                (job.start + job.runtime - job.submit) / max(job.runtime, BD_THRESHOLD), 1
            ) for job in job_history if not job.ignore_in_eval
        ]
        no_eval_ids = [ job.id for job in job_history if job.ignore_in_eval ]
        fifo_bd_slowdowns = [
            max(
                (job.start + job.runtime - job.submit) / max(job.runtime, BD_THRESHOLD), 1
            ) for job in archer_fifo.job_history if job.id not in no_eval_ids
        ]
        data_wait_times = [
            (
                (job.true_job_start + job.runtime - job.true_submit).total_seconds() / 60 / 60
            ) for job in job_history if not job.ignore_in_eval
        ]
        sim_wait_times = [
            (
                (job.start + job.runtime - job.submit).total_seconds() / 60 / 60
            ) for job in job_history if not job.ignore_in_eval
        ]
        fifo_wait_times = [
            (
                (job.start + job.runtime - job.submit).total_seconds() / 60 / 60
            ) for job in archer_fifo.job_history if job.id not in no_eval_ids
        ]
        print(
            "True starts mean bd slowdown={}+-{} (total = {})\n".format(
                np.mean(data_bd_slowdowns), np.std(data_bd_slowdowns), np.sum(data_bd_slowdowns)
            ) +
            "Scheduling sim mean bd slowdown={}+-{} (total = {})\n".format(
                np.mean(sim_bd_slowdowns), np.std(sim_bd_slowdowns), np.sum(sim_bd_slowdowns)
            ) +
            "FIFO baseline sim mean bd slowdown={}+-{}\n".format(
                np.mean(fifo_bd_slowdowns), np.std(fifo_bd_slowdowns)
            ) +
            "True starts mean wait time={}+-{} hrs (total = {} hrs)\n".format(
                np.mean(data_wait_times), np.std(data_wait_times), np.sum(data_wait_times)
            ) +
            "Scheduling sim mean wait time={}+-{}hrs (total = {} hrs)\n".format(
                np.mean(sim_wait_times), np.std(sim_wait_times), np.sum(sim_wait_times)
            ) +
            "FIFO baseline sim mean wait time={}+-{}hr\n".format(
                np.mean(fifo_wait_times), np.std(fifo_wait_times)
            )
        )

        data_bd_slowdowns_no_lowpriority = [
            max(
                (job.true_job_start + job.runtime - job.true_submit) / max(job.runtime, BD_THRESHOLD),
                1
            ) for job in job_history if not job.ignore_in_eval and job.qos.name != "lowpriority"
        ]
        sim_bd_slowdowns_no_lowpriority = [
            max(
                (job.start + job.runtime - job.submit) / max(job.runtime, BD_THRESHOLD), 1
            ) for job in job_history if not job.ignore_in_eval and job.qos.name != "lowpriority"
        ]
        data_wait_times_no_lowpriority = [
            (
                (job.true_job_start + job.runtime - job.true_submit).total_seconds() / 60 / 60
            ) for job in job_history if not job.ignore_in_eval and job.qos.name != "lowpriority"
        ]
        sim_wait_times_no_lowpriority = [
            (
                (job.start + job.runtime - job.submit).total_seconds() / 60 / 60
            ) for job in job_history if not job.ignore_in_eval and job.qos.name != "lowpriority"
        ]

        print(
            "No lowpriority jobs counted:\n" +
            "True starts mean bd slowdown={}+-{}\n".format(
                np.mean(data_bd_slowdowns_no_lowpriority), np.std(data_bd_slowdowns_no_lowpriority)
            ) +
            "Scheduling sim mean bd slowdown={}+-{}\n".format(
                np.mean(sim_bd_slowdowns_no_lowpriority), np.std(sim_bd_slowdowns_no_lowpriority)
            ) +
            "True starts mean wait time={}+-{}hr\n".format(
                np.mean(data_wait_times_no_lowpriority), np.std(data_wait_times_no_lowpriority)
            ) +
            "Scheduling sim mean wait time={}+-{}hr\n".format(
                np.mean(sim_wait_times_no_lowpriority), np.std(sim_wait_times_no_lowpriority)
            )
        )

        data_allocnodes = [ job.nodes for job in job_history if not job.ignore_in_eval ]
        fifo_allocnodes = [
            job.nodes for job in archer_fifo.job_history if job.id not in no_eval_ids
        ]
        fig, ax = bdslowdowns_allocnodes_hist2d_true_fifo_mf_noclass(
            data_bd_slowdowns, data_allocnodes, sim_bd_slowdowns, data_allocnodes,
            fifo_bd_slowdowns, fifo_allocnodes
        )
        fig.savefig(os.path.join(
            PLOT_DIR,
            "toyscheduler_test_refactor_withfifobaseline_bdslowdowns_allocnodes{}.pdf".format(
                save_suffix
            )
        ))
        if batch:
            plt.close()
        else:
            plt.show()

        start_time = archer[0].init_time + timedelta(days=20)

        data_bd_slowdowns_skip = [
            max(
                (job.true_job_start + job.runtime - job.true_submit) / max(job.runtime, BD_THRESHOLD),
                1
            ) for job in job_history if not job.ignore_in_eval and job.submit > start_time
        ]
        sim_bd_slowdowns_skip = [
            max(
                (job.start + job.runtime - job.submit) / max(job.runtime, BD_THRESHOLD), 1
            ) for job in job_history if not job.ignore_in_eval and job.submit > start_time
        ]
        fifo_bd_slowdowns_skip = [
            max(
                (job.start + job.runtime - job.submit) / max(job.runtime, BD_THRESHOLD), 1
            ) for job in archer_fifo.job_history if (
                job.submit > start_time and job.id not in no_eval_ids
            )
        ]
        data_wait_times_skip = [
            (
                (job.true_job_start + job.runtime - job.true_submit).total_seconds() / 60 / 60
            ) for job in job_history if not job.ignore_in_eval and job.submit > start_time
        ]
        sim_wait_times_skip = [
            (
                (job.start + job.runtime - job.submit).total_seconds() / 60 / 60
            ) for job in job_history if not job.ignore_in_eval and job.submit > start_time
        ]
        fifo_wait_times_skip = [
            (
                (job.start + job.runtime - job.submit).total_seconds() / 60 / 60
            ) for job in archer_fifo.job_history if (
                job.submit > start_time and job.id not in no_eval_ids
            )
        ]
        print(
            "=== Skip first 20 days ===\n" +
            "True starts mean bd slowdown={}+-{}\n".format(
                np.mean(data_bd_slowdowns_skip), np.std(data_bd_slowdowns_skip)
            ) +
            "Scheduling sim mean bd slowdown={}+-{}\n".format(
                np.mean(sim_bd_slowdowns_skip), np.std(sim_bd_slowdowns_skip)
            ) +
            "FIFO baseline sim mean bd slowdown={}+-{}\n".format(
                np.mean(fifo_bd_slowdowns_skip), np.std(fifo_bd_slowdowns_skip)
            ) +
            "True starts mean wait time={}+-{}hr\n".format(
                np.mean(data_wait_times_skip), np.std(data_wait_times_skip)
            ) +
            "Scheduling sim mean wait time={}+-{}hr\n".format(
                np.mean(sim_wait_times_skip), np.std(sim_wait_times_skip)
            ) +
            "FIFO baseline sim mean wait time={}+-{}hr\n".format(
                np.mean(fifo_wait_times_skip), np.std(fifo_wait_times_skip)
            )
        )

        data_allocnodes_skip = [
            job.nodes for job in job_history if not job.ignore_in_eval and job.submit > start_time
        ]
        fifo_allocnodes_skip = [
            job.nodes for job in archer_fifo.job_history if (
                job.submit > start_time and job.id not in no_eval_ids
            )
        ]
        fig, ax = bdslowdowns_allocnodes_hist2d_true_fifo_mf_noclass(
            data_bd_slowdowns_skip, data_allocnodes_skip, sim_bd_slowdowns_skip,
            data_allocnodes_skip, fifo_bd_slowdowns_skip, fifo_allocnodes_skip
        )
        fig.savefig(os.path.join(
            PLOT_DIR,
            (
                "toyscheduler_test_refactor_withfifobaseline_bdslowdowns_allocnodes" +
                "_skip28days{}.pdf".format(
                    save_suffix
                )
            )
        ))
        if batch:
            plt.close()
        else:
            plt.show()

        print("Jobs with > 100 difference in bd_slowdown:")
        for job in job_history:
            if job.ignore_in_eval:
                continue

            bd_slowdown_true = max(
                (job.true_job_start + job.runtime - job.true_submit) / max(job.runtime, BD_THRESHOLD), 1
            )
            bd_slowdown_sim = max(
                (job.start + job.runtime - job.submit) / max(job.runtime, BD_THRESHOLD), 1
            )
            if np.abs(bd_slowdown_true - bd_slowdown_sim) > 100:
                print(
                    bd_slowdown_true, bd_slowdown_sim, job.user, job.submit, job.true_submit, job.start,
                    job.true_job_start, job.id, job.nodes, job.name, job.partition, job.qos.name,
                    job.dependency.conditions if job.dependency else None, job.reason
                )
        print()

        assoc_tree = FairTree(
            ASSOCS_FILE, timedelta(minutes=1), timedelta(minutes=1), archer[0].init_time
        )

        print("Jobs from proj-e761 or proj-e697:")
        for job in job_history:
            if job.ignore_in_eval:
                continue

            # proj-e761, proj-e697
            proj = assoc_tree.uniq_users[job.account][job.user].parent.parent.name
            if proj in ["proj-e761", "proj-e697"]:
                print(
                    proj, job.user, job.submit, job.true_submit, job.start, job.true_job_start, job.id, job.nodes,
                    job.name, job.partition, job.qos.name,
                    job.dependency.conditions if job.dependency else None, job.reason
                )
        print()

        proj_sim_wait, proj_data_wait = defaultdict(list), defaultdict(list)
        proj_nodehours = defaultdict(float)
        for job in job_history:
            proj = assoc_tree.uniq_users[job.account][job.user].parent.parent.name
            proj_sim_wait[proj].append((job.start - job.submit).total_seconds()/ 60 / 60)
            proj_data_wait[proj].append(
                (job.true_job_start - job.true_submit).total_seconds() / 60 / 60
            )
            proj_nodehours[proj] += job.nodes * job.runtime.total_seconds() / 60 / 60

        top_projs = [
            proj for proj, _ in (
                sorted(proj_nodehours.items(), key=lambda keyval: keyval[1], reverse=True)[:10]
            )
        ]

        proj_sim_wait_mean = { proj : np.mean(waits) for proj, waits in proj_sim_wait.items() }
        proj_data_wait_mean = { proj : np.mean(waits) for proj, waits in proj_data_wait.items() }
        proj_sim_wait_err = { proj : np.std(waits) for proj, waits in proj_sim_wait.items() }
        proj_data_wait_err = { proj : np.std(waits) for proj, waits in proj_data_wait.items() }
        sorted_sim_wait = [
            (proj, wait_mean, proj_sim_wait_err[proj]) for proj, wait_mean in (
                sorted(
                    proj_sim_wait_mean.items(), key=lambda proj_wait: proj_wait[1], reverse=True
                )
            ) if proj in top_projs
        ]
        sorted_data_wait = [
            (proj, wait_mean, proj_data_wait_err[proj]) for proj, wait_mean in (
                sorted(
                    proj_data_wait_mean.items(), key=lambda proj_wait: proj_wait[1], reverse=True
                )
            ) if proj in top_projs
        ]

        print(
            "Scheduling sim top projects by mean wait times:\n" +
            "\n".join(
                "{}.\t{}\t- {} += {}".format(
                    i, proj_wait[0], proj_wait[1], proj_wait[2]
                ) for i, proj_wait in enumerate(sorted_sim_wait)
            ) +
            "\n"
        )
        print(
            "True starts top projects by mean wait times:\n" +
            "\n".join(
                "{}.\t{}\t- {} += {}".format(
                    i, proj_wait[0], proj_wait[1], proj_wait[2]
                ) for i, proj_wait in enumerate(sorted_data_wait)
            ) +
            "\n"
        )

        top_projs.sort(key=lambda proj: proj_data_wait[proj], reverse=True)
        sim_mean_waits = [ proj_sim_wait_mean[proj] for proj in top_projs ]
        data_mean_waits = [ proj_data_wait_mean[proj] for proj in top_projs ]
        x = np.arange(len(top_projs))

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        sim_bars = ax.bar(x - 2 * 0.2 / 3, sim_mean_waits, 0.2, label="Scheduling Sim")
        data_bars = ax.bar(x + 2 * 0.2 / 3, data_mean_waits, 0.2, label="Data")
        ax.set_ylabel("Mean Wait Time (hrs)")
        ax.set_xticks(x, top_projs)
        ax.legend()
        ax.bar_label(sim_bars, padding=3, fmt="%.1f")
        ax.bar_label(data_bars, padding=3, fmt="%.1f")
        fig.tight_layout()
        fig.savefig(os.path.join(
            PLOT_DIR, "test_refactor_top_projs_mean_waits{}.pdf".format(save_suffix)
        ))
        if batch:
            plt.close()
        else:
            plt.show()

        hours = [
            archer[0].init_time.replace(minute=0, second=0) + timedelta(hours=hr) for hr in (
                range(int(
                    (
                        max_submit - timedelta(days=14) - archer[0].init_time
                    ).total_seconds() / 60 / 60
                ))
            )
        ]

        # Rolling window mean wait time
        sim_submit_hour_waits = defaultdict(list)
        for job in job_history:
            sim_submit_hour_waits[job.submit.replace(minute=0, second=0)].append(
                (job.start - job.submit).total_seconds() / 60 / 60
            )

        sim_mean_wait_times_rolling_window = np.zeros(len(hours))
        sim_mean_wait_times_rolling_window_err = np.zeros(len(hours))
        wait_times_rolling_window, wait_times_rolling_window_hour_lens = [], []
        for hr_num in range(336):
            wait_times_rolling_window += sim_submit_hour_waits[hours[0] + timedelta(hours=hr_num)]
            wait_times_rolling_window_hour_lens.append(
                len(sim_submit_hour_waits[hours[0] + timedelta(hours=hr_num)])
            )
        for i_hour, hour in enumerate(hours):
            sim_mean_wait_times_rolling_window[i_hour] = np.mean(wait_times_rolling_window)
            sim_mean_wait_times_rolling_window_err[i_hour] = np.std(wait_times_rolling_window)
            wait_times_rolling_window = (
                wait_times_rolling_window[wait_times_rolling_window_hour_lens.pop(0):]
            )
            wait_times_rolling_window += sim_submit_hour_waits[hour + timedelta(hours=336)]
            wait_times_rolling_window_hour_lens.append(
                len(sim_submit_hour_waits[hour + timedelta(hours=336)])
            )

        data_submit_hour_waits = defaultdict(list)
        for job in job_history:
            data_submit_hour_waits[job.submit.replace(minute=0, second=0)].append(
                (job.true_job_start - job.true_submit).total_seconds() / 60 / 60
            )

        data_mean_wait_times_rolling_window = np.zeros(len(hours))
        data_mean_wait_times_rolling_window_err = np.zeros(len(hours))
        wait_times_rolling_window, wait_times_rolling_window_hour_lens = [], []
        for hr_num in range(336):
            wait_times_rolling_window += data_submit_hour_waits[hours[0] + timedelta(hours=hr_num)]
            wait_times_rolling_window_hour_lens.append(
                len(data_submit_hour_waits[hours[0] + timedelta(hours=hr_num)])
            )
        for i_hour, hour in enumerate(hours):
            data_mean_wait_times_rolling_window[i_hour] = np.mean(wait_times_rolling_window)
            data_mean_wait_times_rolling_window_err[i_hour] = np.std(wait_times_rolling_window)
            wait_times_rolling_window = (
                wait_times_rolling_window[wait_times_rolling_window_hour_lens.pop(0):]
            )
            wait_times_rolling_window += data_submit_hour_waits[hour + timedelta(hours=336)]
            wait_times_rolling_window_hour_lens.append(
                len(data_submit_hour_waits[hour + timedelta(hours=336)])
            )

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        hour_dates = matplotlib.dates.date2num([ hour + timedelta(days=7) for hour in hours ])
        ax.plot_date(
            hour_dates, sim_mean_wait_times_rolling_window, 'g', label="Scheduling sim",
            linewidth=0.6
        )
        ax.fill_between(
            hour_dates,
            sim_mean_wait_times_rolling_window - sim_mean_wait_times_rolling_window_err,
            sim_mean_wait_times_rolling_window + sim_mean_wait_times_rolling_window_err,
            edgecolor='g', facecolor='g', alpha=0.2, linewidth=0
        )
        ax.plot_date(
            hour_dates, data_mean_wait_times_rolling_window, 'r', label="Data", linewidth=0.6
        )
        ax.fill_between(
            hour_dates,
            data_mean_wait_times_rolling_window - data_mean_wait_times_rolling_window_err,
            data_mean_wait_times_rolling_window + data_mean_wait_times_rolling_window_err,
            edgecolor='r', facecolor='r', alpha=0.2, linewidth=0
        )
        ax.set_ylabel("Mean Wait Time in Window")
        ax.set_xlabel("Middle Hour of 2 Week Rolling Window")
        ax.grid(axis="x")
        plt.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(
            PLOT_DIR, "test_refactor_wait_times_rolling_window{}.pdf".format(save_suffix)
        ))
        if batch:
            plt.close()
        else:
            plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot_date(
            hour_dates, sim_mean_wait_times_rolling_window, 'g', label="Scheduling sim",
            linewidth=0.6
        )
        ax.plot_date(
            hour_dates, data_mean_wait_times_rolling_window, 'r', label="Data", linewidth=0.6
        )
        ax.set_ylabel("Mean Wait Time in Window")
        ax.set_xlabel("Middle Hour of 2 Week Rolling Window")
        ax.grid(axis="x")
        plt.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(
            PLOT_DIR, "test_refactor_wait_times_rolling_window_noerr{}.pdf".format(save_suffix)
        ))
        if batch:
            plt.close()
        else:
            plt.show()

        # Rolling window mean bd slowdown
        sim_submit_hour_bdslowdowns = defaultdict(list)
        for job in job_history:
            sim_submit_hour_bdslowdowns[job.submit.replace(minute=0, second=0)].append(
                max((job.end - job.submit) / max(job.runtime, BD_THRESHOLD), 1)
            )

        sim_mean_bdslowdowns_rolling_window = np.zeros(len(hours))
        sim_mean_bdslowdowns_rolling_window_err = np.zeros(len(hours))
        bdslowdowns_rolling_window, bdslowdowns_rolling_window_hour_lens = [], []
        for hr_num in range(336):
            bdslowdowns_rolling_window += (
                sim_submit_hour_bdslowdowns[hours[0] + timedelta(hours=hr_num)]
            )
            bdslowdowns_rolling_window_hour_lens.append(
                len(sim_submit_hour_bdslowdowns[hours[0] + timedelta(hours=hr_num)])
            )
        for i_hour, hour in enumerate(hours):
            sim_mean_bdslowdowns_rolling_window[i_hour] = np.mean(bdslowdowns_rolling_window)
            sim_mean_bdslowdowns_rolling_window_err[i_hour] = np.std(bdslowdowns_rolling_window)
            bdslowdowns_rolling_window = (
                bdslowdowns_rolling_window[bdslowdowns_rolling_window_hour_lens.pop(0):]
            )
            bdslowdowns_rolling_window += sim_submit_hour_bdslowdowns[hour + timedelta(hours=336)]
            bdslowdowns_rolling_window_hour_lens.append(
                len(sim_submit_hour_bdslowdowns[hour + timedelta(hours=336)])
            )

        data_submit_hour_bdslowdowns = defaultdict(list)
        for job in job_history:
            data_submit_hour_bdslowdowns[job.submit.replace(minute=0, second=0)].append(
                max(
                    (
                        (job.true_job_start + job.runtime - job.true_submit) /
                        max(job.runtime, BD_THRESHOLD)
                    ),
                    1
                )
            )

        data_mean_bdslowdowns_rolling_window = np.zeros(len(hours))
        data_mean_bdslowdowns_rolling_window_err = np.zeros(len(hours))
        bdslowdowns_rolling_window, bdslowdowns_rolling_window_hour_lens = [], []
        for hr_num in range(336):
            bdslowdowns_rolling_window += (
                data_submit_hour_bdslowdowns[hours[0] + timedelta(hours=hr_num)]
            )
            bdslowdowns_rolling_window_hour_lens.append(
                len(data_submit_hour_bdslowdowns[hours[0] + timedelta(hours=hr_num)])
            )
        for i_hour, hour in enumerate(hours):
            data_mean_bdslowdowns_rolling_window[i_hour] = np.mean(bdslowdowns_rolling_window)
            data_mean_bdslowdowns_rolling_window_err[i_hour] = np.std(bdslowdowns_rolling_window)
            bdslowdowns_rolling_window = (
                bdslowdowns_rolling_window[bdslowdowns_rolling_window_hour_lens.pop(0):]
            )
            bdslowdowns_rolling_window += data_submit_hour_bdslowdowns[hour + timedelta(hours=336)]
            bdslowdowns_rolling_window_hour_lens.append(
                len(data_submit_hour_bdslowdowns[hour + timedelta(hours=336)])
            )

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot_date(
            hour_dates, sim_mean_bdslowdowns_rolling_window, 'g', label="Scheduling sim",
            linewidth=0.6
        )
        ax.fill_between(
            hour_dates,
            sim_mean_bdslowdowns_rolling_window - sim_mean_bdslowdowns_rolling_window_err,
            sim_mean_bdslowdowns_rolling_window + sim_mean_bdslowdowns_rolling_window_err,
            edgecolor='g', facecolor='g', alpha=0.2, linewidth=0
        )
        ax.plot_date(
            hour_dates, data_mean_bdslowdowns_rolling_window, 'r', label="Data", linewidth=0.6
        )
        ax.fill_between(
            hour_dates,
            data_mean_bdslowdowns_rolling_window - data_mean_bdslowdowns_rolling_window_err,
            data_mean_bdslowdowns_rolling_window + data_mean_bdslowdowns_rolling_window_err,
            edgecolor='r', facecolor='r', alpha=0.2, linewidth=0
        )
        ax.set_ylabel("Mean Bounded Slowdown in Window")
        ax.set_xlabel("Middle Hour of 2 Week Rolling Window")
        ax.grid(axis="x")
        plt.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(
            PLOT_DIR, "test_refactor_bd_slowdowns_rolling_window{}.pdf".format(save_suffix)
        ))
        if batch:
            plt.close()
        else:
            plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot_date(
            hour_dates, sim_mean_bdslowdowns_rolling_window, 'g', label="Scheduling sim",
            linewidth=0.6
        )
        ax.plot_date(
            hour_dates, data_mean_bdslowdowns_rolling_window, 'r', label="Data", linewidth=0.6
        )
        ax.set_ylabel("Mean Bounded Slowdown in Window")
        ax.set_xlabel("Middle Hour of 2 Week Rolling Window")
        ax.grid(axis="x")
        plt.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(
            PLOT_DIR, "test_refactor_bd_slowdowns_rolling_window_noerr{}.pdf".format(save_suffix)
        ))
        if batch:
            plt.close()
        else:
            plt.show()

        sim_end_hour_cnt = defaultdict(int)
        for job in job_history:
            sim_end_hour_cnt[job.end.replace(minute=0, second=0)] += 1

        sim_throughput_cumulative = [sim_end_hour_cnt[hours[0]]]
        for hour in hours[1:]:
            sim_throughput_cumulative.append(
                sim_throughput_cumulative[-1] + sim_end_hour_cnt[hour]
            )

        data_end_hour_cnt = defaultdict(int)
        for job in job_history:
            data_end_hour_cnt[(job.true_job_start + job.runtime).replace(minute=0, second=0)] += 1

        data_throughput_cumulative = [data_end_hour_cnt[hours[0]]]
        for hour in hours[1:]:
            data_throughput_cumulative.append(
                data_throughput_cumulative[-1] + data_end_hour_cnt[hour]
            )

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        hour_dates_start_hour = matplotlib.dates.date2num([ hour for hour in hours ])
        ax.plot_date(
            hour_dates_start_hour, sim_throughput_cumulative, 'g', label="Scheduling sim",
            linewidth=0.6
        )
        ax.plot_date(
            hour_dates_start_hour, data_throughput_cumulative, 'r', label="Data", linewidth=0.6
        )
        ax.set_ylabel("Cumulative Throughput")
        ax.set_xlabel("Date (hour resolution)")
        ax.grid(axis="x")
        plt.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(
            PLOT_DIR, "test_refactor_throughput_cumulative{}.pdf".format(save_suffix)
        ))
        if batch:
            plt.close()
        else:
            plt.show()

        sim_max_end = max(job_history, key=lambda job: job.end).end
        sim_min_start = min(job_history, key=lambda job: job.start).start
        data_max_end_job = max(job_history, key=lambda job: job.true_job_start + job.runtime)
        data_max_end = data_max_end_job.true_job_start + data_max_end_job.runtime
        data_min_start = min(job_history, key=lambda job: job.true_job_start).true_job_start

        sim_allocated_nodes = np.zeros(int((sim_max_end - sim_min_start).total_seconds() / 60))
        data_allocated_nodes = np.zeros(int((data_max_end - data_min_start).total_seconds() / 60))

        # sim_mins_allocated_nodes = defaultdict(int)
        # data_mins_allocated_nodes = defaultdict(int)

        for job in tqdm(job_history):
            l_mins = int((job.start - sim_min_start).total_seconds() / 60) + 1
            u_mins = int((job.end - sim_min_start).total_seconds() / 60)
            sim_allocated_nodes[l_mins:u_mins] += job.nodes

            l_mins = int((job.true_job_start - data_min_start).total_seconds() / 60) + 1
            u_mins = int((job.true_job_start + job.runtime - data_min_start).total_seconds() / 60)
            data_allocated_nodes[l_mins:u_mins] += job.nodes

            # minute = job.start.replace(minute=0) + timedelta(minutes=1)
            # while minute < job.end:
            #     sim_mins_allocated_nodes[minute] += job.nodes
            #     minute += timedelta(minutes=1)
            # minute = job.true_job_start.replace(minute=0) + timedelta(minutes=1)
            # while minute < job.true_job_start + job.runtime:
            #     data_mins_allocated_nodes[minute] += job.nodes
            #     minute += timedelta(minutes=1)

        # for minute in sim_mins_allocated_nodes.keys():
        #     sim_nodes = sim_mins_allocated_nodes[minute]
        #     data_nodes = data_mins_allocated_nodes[minute]
        #     if sim_nodes > 5860 or data_nodes > 5860:
        #         print("\n", minute, sim_nodes, data_nodes, "\n", sep="\t")
        #     print(minute, sim_nodes, data_nodes, end="\r", sep="\t")

        print("Scheduling sim mean(max) allocations nodes = {} +- {} ({})".format(
            np.mean(sim_allocated_nodes[2880:-2880]), np.std(sim_allocated_nodes[2880:-2880]),
            np.max(sim_allocated_nodes)
        ))
        print("Data mean(max) allocations nodes = {} +- {} ({})".format(
            np.mean(data_allocated_nodes[2880:-2880]), np.std(data_allocated_nodes[2880:-2880]),
            np.max(data_allocated_nodes)
        ))

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        data_minutes = [
            data_min_start + timedelta(minutes=min_num) for min_num in (
                range(len(data_allocated_nodes))
            )
        ]
        sim_minutes = [
            sim_min_start + timedelta(minutes=min_num) for min_num in (
                range(len(sim_allocated_nodes))
            )
        ]
        ax.plot_date(sim_minutes, sim_allocated_nodes, 'g', label="Scheduling sim", linewidth=0.1)
        ax.plot_date(data_minutes, data_allocated_nodes, 'r', label="Data", linewidth=0.1)
        ax.set_ylabel("Date (minute resolution)")
        ax.set_xlabel("Number of Allocated Nodes")
        ax.grid(axis="y")
        plt.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(
            PLOT_DIR, "test_refactor_total_allocnodes_bytime{}.pdf".format(save_suffix)
        ))
        if batch:
            plt.close()
        else:
            plt.show()


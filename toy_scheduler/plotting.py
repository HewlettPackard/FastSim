import os
from glob import glob
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import matplotlib.dates
from matplotlib import pyplot as plt

from classes import Archer2
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

""" End Helper Plotting Thingys """


def plot_blob(
    plots, archer, start, end, times, dates, archer_fcfs=None, times_fcfs=None, dates_fcfs=None,
    save_suffix="", batch=False, df_jobs=None
):
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

        interval = ((0,24),(24,48)) # This looks the clearest
        fig, ax = plt.subplots(1, 1, figsize=(14, 8))
        ax.plot_date(dates[interval], archer[interval].power_history, 'g', linewidth=0.6)
        interval_shade(ax, start[interval], end[interval], interval)
        ax.set_ylabel("Power (MW)")
        plt.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(
            PLOT_DIR, "toyscheduler_low-high_power_0242448_scan{}.pdf".format(save_suffix)
        ))
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
        plt.xticks(angles[:-1], [ "1/{}".format(category) for category in categories ], size=12)
        ax.set_rlabel_position(0)
        plt.yticks([0.75,1,1.25], ["0.75","1","1.25"], color="grey", size=10)
        plt.ylim(0.7,1.3)

        for queue_cut, perf_data in spider_plot_data.items():
            vals = [ 1 / perf_data[metric] for metric in categories ]
            vals += vals[:1]
            colour = next(ax._get_lines.prop_cycler)["color"]
            ax.plot(angles, vals, linewidth=1, linestyle='solid', c=colour, label=str(queue_cut))

        ax.plot(
            angles, [1] * len(angles), linewidth=2, linestyle='solid', c='k', label="no low freq"

        )
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.15))
        plt.tight_layout()
        fig.savefig(os.path.join(
            PLOT_DIR,
            "toyscheduler_priority_small_and_age_lowfreq_queuecut_perfspiderplt{}.pdf".format(
                save_suffix
            )
        ))
        if batch:
            plt.close()
        else:
            plt.show()

    if "test_mf_priority" in plots:
        print(
            "MF priority w/ fairshare mean bd slowdown={}+-{}".format(
                np.mean(archer[0].bd_slowdowns), np.std(archer[0].bd_slowdowns)
            ) +
            "true start mean bd slowdown={}+-{}".format(
                np.mean(archer[-1].bd_slowdowns), np.std(archer[-1].bd_slowdowns)
            )
        )
        print(
            "MF priority w/o fairshare mean bd slowdown={}+-{}".format(
                np.mean(archer[1].bd_slowdowns), np.std(archer[1].bd_slowdowns)
            ) +
            "true start mean bd slowdown={}+-{}".format(
                np.mean(archer[-1].bd_slowdowns), np.std(archer[-1].bd_slowdowns)
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

        wait_times_true = {
            job.id : (job.start - job.submit).total_seconds for job in archer[-1].job_history
        }
        wait_times_fairshare = {
            job.id : (job.start - job.submit).total_seconds for job in archer[0].job_history
        }
        wait_times_nofairshare = {
            job.id : (job.start - job.submit).total_seconds for job in archer[1].job_history
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
            "any effect from unkown initial usages==="
        )

        start_time = archer[0].init_time + timedelta(days=28)
        archer[-1].job_history = [
            job for job in archer[-1].job_history if job.Submit > start_time
        ]
        archer[0].job_history = [ job for job in archer[0].job_history if job.Submit > start_time ]
        archer[1].job_history = [ job for job in archer[1].job_history if job.Submit > start_time ]
        archer[-1].bd_slowdowns = [
            max(job.end - job.submit) / max(job.runtime, BD_THRESHOLD), 1) for job in (
                archer[-1].job_history
            )
        ]
        archer[0].bd_slowdowns = [
            max(job.end - job.submit) / max(job.runtime, BD_THRESHOLD), 1) for job in (
                archer[0].job_history
            )
        ]
        archer[1].bd_slowdowns = [
            max(job.end - job.submit) / max(job.runtime, BD_THRESHOLD), 1) for job in (
                archer[1].job_history
            )
        ]

        print(
            "MF priority w/ fairshare mean bd slowdown={}+-{}".format(
                np.mean(archer[0].bd_slowdowns), np.std(archer[0].bd_slowdowns)
            ) +
            "true start mean bd slowdown={}+-{}".format(
                np.mean(archer[-1].bd_slowdowns), np.std(archer[-1].bd_slowdowns)
            )
        )
        print(
            "MF priority w/o fairshare mean bd slowdown={}+-{}".format(
                np.mean(archer[1].bd_slowdowns), np.std(archer[1].bd_slowdowns)
            ) +
            "true start mean bd slowdown={}+-{}".format(
                np.mean(archer[-1].bd_slowdowns), np.std(archer[-1].bd_slowdowns)
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

        wait_times_true = {
            job.id : (job.start - job.submit).total_seconds for job in archer[-1].job_history
        }
        wait_times_fairshare = {
            job.id : (job.start - job.submit).total_seconds for job in archer[0].job_history
        }
        wait_times_nofairshare = {
            job.id : (job.start - job.submit).total_seconds for job in archer[1].job_history
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


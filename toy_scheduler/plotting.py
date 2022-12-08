import os
from glob import glob

import pandas as pd
import numpy as np
import matplotlib.dates
from matplotlib import pyplot as plt

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
            baseline_power = 1.692
            for tick, date in enumerate(times[interval][1000:-1000]):
                if low_or_high(date) == "low":
                    low_powers.append(power[tick] + (1 - occupancy[tick]) * baseline_power)
                else:
                    high_powers.append(power[tick] + (1 - occupancy[tick]) * baseline_power)

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
                baseline_power = 1.692
                for tick, date in enumerate(times[interval][1000:-1000]):
                    if low_or_high(date) == "low":
                        low_powers.append(power[tick] + (1 - occupancy[tick]) * baseline_power)
                    else:
                        high_powers.append(power[tick] + (1 - occupancy[tick]) * baseline_power)

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
        bd_slowdown_data = np.mean(
            [
                max(
                    (job_row.End - job_row.Submit)/max(job_row.Elapsed, BD_THRESHOLD), 1
                ) for _, job_row in df_jobs.iterrows()
            ][1000:-1000]
        )
        # NOTE: No interval to compute low and high powers with so don't need to do this, just set
        # mean high - mean low to zero
        # classTempQueue():
        #     def __init__(self, submit, runtime, powerpernode, nodes, start, end):
        #         self.submit = submit
        #         self.runtime = runtime
        #         self.power = powerpernode * nodes
        #         self.start = start
        #         self.end = end
        # jobs = [
        #     Job(
        #         job_row.Submit, job_row.Elapsed, job_row.PowerPerNode, job_row.AllocNodes,
        #         job_row.Start, job_row.End
        #     ) for _, job_row in df_jobs.sort_values("Start").iterrows()
        # ]
        # times = [ job.start for job in jobs ] + [ job.end for job in jobs ]
        # list(set(times)).sort()
        # powers = np.zeros_like(times)
        # for job in jobs:
        #     powers[times.index(job.start):times.index(jobs.end)] += job.power / 1e+6
        # print(np.mean(powers), np.max(powers))
        x, y = bd_slowdown_data, 0
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
        ax.hist(
            archer[interval].bd_slowdowns, bins=int(max(archer[interval].bd_slowdowns)),
            range=(min(archer[interval].bd_slowdowns),max(archer[interval].bd_slowdowns)),
            histtype="step", label="My scheduler"
        )
        ax.hist(
            bd_slowdown_data, bins=int(max(archer[interval].bd_slowdowns)),
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

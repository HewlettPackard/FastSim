"""
Script to plot slurm job data dumped using `sacct -ap --starttime=YYYY-MM-DD -o ...`
"""

import argparse, os, sys, datetime
from glob import glob
from collections import defaultdict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.dates
from tqdm import tqdm

from sklearn.linear_model import RANSACRegressor, HuberRegressor, LinearRegression
from sklearn.preprocessing import StandardScaler

from funcs import power_print_dump, parse_cache, timelimit_str_to_timedelta


# with pd.option_context('display.max_rows', None):

def power_usage_plot(df, timesteps, cache, data_name):
    df_power = parse_cache(
        df, cache, data_name, "power_usage_df", ["JobID", "Start", "End", "ConsumedEnergyRaw"]
    )

    power_print_dump(df_power)

    t = pd.date_range(min(df_power.Start), max(df_power.End), periods=timesteps)

    power_usage = np.zeros(t.values.size - 1)

    for i in range(len(power_usage)):
        slice = df_power.loc[(df_power.Start <= t[i]) & (df_power.End > t[i + 1])] #, ["Power"]]
        power_usage[i] = slice.Power.sum()

    dates = matplotlib.dates.date2num(t[:-1])
    plt.plot_date(dates, power_usage, 'g', linewidth=0.8)
    plt.ylabel("Power (W)")
    plt.title("IT Equipment Power consumption on ARCHER2 from slurm data")
    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    fig.tight_layout()
    fig.savefig("plots/powerusage.pdf")
    plt.show()


def power_usage_exitcodes(df, timesteps, cache, data_name):
    df_power = parse_cache(
        df, cache, data_name, "power_usage_exitcodes_df",
        ["JobID", "Start", "End", "ConsumedEnergyRaw", "ExitCode"]
    )

    power_print_dump(df_power)

    df_power_success = df_power.loc[df_power.ExitCode == "0:0"]
    df_power_fail = df_power.loc[df_power.ExitCode != "0:0"]

    print("\n{} successful jobs, {} failed jobs\n".format(
        len(df_power_success), len(df_power_fail))
    )

    t = pd.date_range(min(df_power.Start), max(df_power.End), periods=timesteps)

    power_usage_success = np.zeros(t.values.size - 1)
    power_usage_fail = np.zeros(t.values.size - 1)

    for i in range(len(power_usage_success)):
        slice = df_power_success.loc[
            (df_power_success.Start <= t[i]) & (df_power_success.End > t[i + 1]), ["Power"]
            ]
        power_usage_success[i] = slice.Power.sum()
    for i in range(len(power_usage_fail)):
        slice = df_power_fail.loc[
            (df_power_fail.Start <= t[i]) & (df_power_fail.End > t[i + 1]), ["Power"]
            ]
        power_usage_fail[i] = slice.Power.sum()

    dates = matplotlib.dates.date2num(t[:-1])
    plt.plot_date(dates, power_usage_success, 'g', linewidth=0.8, label="0 exit code")
    plt.plot_date(dates, power_usage_fail, 'r', linewidth=0.8, label="!0 exit code")
    plt.legend()
    plt.ylabel("Power (W)")
    plt.title("IT Equipment Power consumption on ARCHER2 from slurm data")
    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    fig.tight_layout()
    fig.savefig("plots/powerusage_exitcodes.pdf")
    plt.show()


def power_usage_cabs(df, timesteps, cache, data_name, cab_dir):
    df_power = parse_cache(
        df, cache, data_name, "power_usage_cabs_df",
        ["JobID", "Start", "End", "ConsumedEnergyRaw"]
    )

    cabs = {
        datetime.datetime.strptime(os.path.basename(path), "system_%y%m%d") : path
        for path in glob(os.path.join(cab_dir, "system_[2]*"))
    }

    start, end = min(df_power.Start), max(df_power.End)

    for date, cab in cabs.items():
        if date > start and (date + datetime.timedelta(days=1)) < end:
            df_cab = pd.read_csv(cab, delimiter=" ", names=["Time", "Power"])
            df_cab.Time = pd.to_datetime(df_cab.Time, format="%H:%M:%S")
            df_cab.Time = df_cab.Time.apply(
                lambda row:
                datetime.timedelta(hours=row.hour, minutes=row.minute, seconds=row.second) + date
            )

            try:
                df_cabs = pd.concat([df_cabs, df_cab])
            except NameError:
                df_cabs = df_cab

    t_slurm = pd.date_range(min(df_power.Start), max(df_power.End), periods=timesteps)
    t_cabs = pd.DatetimeIndex(df_cabs.sort_values("Time").Time.values)

    power_usage_slurm = np.zeros(t_slurm.values.size - 1)
    for i in range(len(power_usage_slurm)):
        slice = df_power.loc[(df_power.Start <= t_slurm[i]) & (df_power.End > t_slurm[i + 1])]
        power_usage_slurm[i] = slice.Power.sum()

    power_usage_cabs = np.zeros(t_cabs.values.size - 1)
    for i in range(len(power_usage_cabs)):
        slice = df_cabs.loc[(df_cabs.Time == t_cabs[i])]
        power_usage_cabs[i] = slice.Power.iloc[0] * 1000

    dates_slurm = matplotlib.dates.date2num(t_slurm[:-1])
    dates_cabs = matplotlib.dates.date2num(t_cabs[:-1])
    plt.plot_date(dates_slurm, power_usage_slurm, 'g', linewidth=0.8, label="Slurm logs")
    plt.plot_date(dates_cabs, power_usage_cabs, 'r', linewidth=0.8, label="cabinet power dumps")
    plt.legend()
    plt.ylabel("Power (W)")
    plt.title("Power consumption on ARCHER2")
    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    fig.tight_layout()
    fig.savefig("plots/powerusage_cabs.pdf")
    plt.show()

    print("t_cabs: len={}. min={}, max={}".format(len(t_cabs), min(t_cabs), max(t_cabs)))
    print("t_slurm: len={}. min={}, max={}".format(len(t_slurm), min(t_slurm), max(t_slurm)))

    # NOTE: make sure  min(t_slurm) < min(t_cabs) && max(t_slurm) > max(t_cabs) first
    power_usage_slurm_tcabs = np.zeros(t_cabs.values.size - 1)
    for i in range(len(power_usage_slurm_tcabs)):
        slice = df_power.loc[(df_power.Start <= t_cabs[i]) & (df_power.End > t_cabs[i + 1])]
        power_usage_slurm_tcabs[i] = slice.Power.sum()

    power_usage_residuals = power_usage_cabs - power_usage_slurm_tcabs

    fig = plt.figure(1)

    ax_main=fig.add_axes((.1,.3,.8,.6))
    plt.plot_date(dates_cabs, power_usage_slurm_tcabs, 'g', linewidth=0.8, label="Slurm logs")
    plt.plot_date(dates_cabs, power_usage_cabs, 'r', linewidth=0.8, label="cabinet power dumps")
    plt.legend()
    plt.ylabel("Power (W)")
    plt.title("Power consumption on ARCHER2")
    ax_main.set_xticklabels([])

    ax_residuals = fig.add_axes((.1,.1,.8,.2))
    plt.plot_date(dates_cabs, power_usage_residuals, 'k', linewidth=0.6, label="cabinet power dumps - Slurm logs")
    plt.legend()
    plt.ylabel("PCab - PSlurm (W)")
    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    fig.tight_layout()
    fig.savefig("plots/powerusage_cabs_residuals.pdf")
    plt.show()

    plt.scatter(power_usage_slurm_tcabs, power_usage_cabs, s=2)
    plt.xlabel("PSlurm")
    plt.ylabel("PCabs")
    plt.xlim(left=-0.1e+6, right=4e+6)
    plt.ylim(bottom=-0.1e+6, top=4e+6)
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig("plots/powerusage_cabs_against_slurm.pdf")
    plt.show()

    cabs_fine = {
        datetime.datetime.strptime(os.path.basename(path), "cabs_%y%m%d") : path
        for path in glob(os.path.join(cab_dir, "cabs_[2]*"))
    }

    for date, cab in cabs_fine.items():
        if date > start and (date + datetime.timedelta(days=1)) < end:
            df_cab = pd.read_csv(
                cab, delimiter=" ", index_col=False,
                names=["Time", "Cabinet", "Total?", "Component", "Power"]
            )
            df_cab = df_cab.loc[
                (df_cab.Component == "total_cab_power"), ["Time", "Cabinet", "Power"]
            ]

            df_cab.Time = pd.to_datetime(df_cab.Time, format="%H:%M:%S")
            df_cab.Time = df_cab.Time.apply(
                lambda row:
                datetime.timedelta(hours=row.hour, minutes=row.minute, seconds=row.second) + date
            )

            try:
                df_cabs_fine = pd.concat([df_cabs_fine, df_cab])
            except NameError:
                df_cabs_fine = df_cab

    t_cabs_fine = pd.DatetimeIndex(df_cabs_fine.sort_values("Time").Time.unique())

    power_usage_cabs_fine = np.zeros(t_cabs_fine.values.size - 1)
    for i in range(len(power_usage_cabs_fine)):
        slice = df_cabs_fine.loc[(df_cabs_fine.Time == t_cabs_fine[i])]
        power_usage_cabs_fine[i] = slice.Power.sum() * 1000

    dates_cabs_fine = matplotlib.dates.date2num(t_cabs_fine[:-1])

    plt.plot_date(dates_slurm, power_usage_slurm, 'g', linewidth=0.8, label="Slurm logs")
    plt.plot_date(dates_cabs_fine, power_usage_cabs_fine, 'r', linewidth=0.8, label="All cabinets")

    power_usage_all_cabs = []
    cabs_itr = tqdm(
        df_cabs_fine.Cabinet.unique(), desc="Plotting individual cabinet power usage..."
    )
    for i_cab, cab in enumerate(cabs_itr):
        if cache == "load":
            power_usage_cab = np.load(
                "/work/y02/y02/awilkins/pandas_cache/{}/power_usage_{}.npy".format(data_name, cab)
            )
        else:
            power_usage_cab = np.zeros(t_cabs_fine.values.size - 1)
            cab_slice = df_cabs_fine.loc[(df_cabs_fine.Cabinet == cab)]
            for tick in range(len(power_usage_cab)):
                cab_time_slice = cab_slice.loc[(cab_slice.Time == t_cabs_fine[tick])]
                try:
                    power_usage_cab[tick] = cab_time_slice.Power.iloc[0] * 1000
                except IndexError:
                    # Some cabinets are missing entries (switched off for maintenance?)
                    continue

        if cache == "save":
            np.save(
                "/work/y02/y02/awilkins/pandas_cache/{}/power_usage_{}.npy".format(data_name, cab),
                power_usage_cab
            )

        plt.plot_date(
            dates_cabs_fine, power_usage_cab, 'b', linewidth=0.5, alpha=0.3,
            label="Individual cabinets" if not i_cab else ""
        )

        power_usage_all_cabs.append(power_usage_cab)

    plt.yscale("log")
    plt.legend()
    plt.ylabel("Power (W)")
    plt.title("Power consumption on ARCHER2")
    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    fig.tight_layout()
    fig.savefig("plots/powerusage_cabs_individual.pdf")
    plt.show()

    # NOTE: This shift is coming from daylight savings which started on 29 October. I think the
    #       cab power is just GMT while the slurm times take into account daylight savings. So
    #       before end of 29 October slurm was in BST whilst the cab dump was GMT, hence the need
    #       to shift slurm power back by an hour
    shift = np.diff(power_usage_slurm_tcabs).argmax() - np.diff(power_usage_cabs).argmax()
    print("Shifting by {}".format(shift))
    power_usage_slurm_tcabs_shifted = np.roll(power_usage_slurm_tcabs, -shift)
    power_usage_slurm_tcabs_shifted[-shift:] = 0

    power_usage_shifted_residuals = power_usage_cabs - power_usage_slurm_tcabs_shifted

    fig = plt.figure(1)

    ax_main=fig.add_axes((.1,.3,.8,.6))
    plt.plot_date(dates_cabs, power_usage_slurm_tcabs_shifted, 'g', linewidth=0.8, label="Slurm logs (shifted)")
    plt.plot_date(dates_cabs, power_usage_cabs, 'r', linewidth=0.8, label="cabinet power dumps")
    plt.legend()
    plt.ylabel("Power (W)")
    plt.title("Power consumption on ARCHER2")
    ax_main.set_xticklabels([])

    ax_residuals = fig.add_axes((.1,.1,.8,.2))
    plt.plot_date(dates_cabs, power_usage_shifted_residuals, 'k', linewidth=0.6, label="cabinet power dumps - Slurm logs")
    plt.legend()
    plt.ylabel("PCab - PSlurm (W)")
    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    fig.tight_layout()
    fig.savefig("plots/powerusage_cabs_shifted_residuals.pdf")
    plt.show()

    plt.scatter(power_usage_slurm_tcabs_shifted, power_usage_cabs, s=1)
    plt.xlabel("PSlurm (shifted)")
    plt.ylabel("PCabs")
    plt.xlim(left=-0.1e+6, right=4e+6)
    plt.ylim(bottom=-0.1e+6, top=4e+6)
    fig = plt.gcf()
    fig.set_size_inches(12, 10)
    fig.tight_layout()
    fig.savefig("plots/powerusage_cabs_against_slurm_shifted.pdf")
    plt.show()

    cabs_off, cabs_partial = np.zeros_like(power_usage_all_cabs[0], dtype=int), np.zeros_like(power_usage_all_cabs[0], dtype=int)
    for t in range(len(power_usage_all_cabs[0])):
        for power_usage in power_usage_all_cabs:
            if power_usage[t] == 0:
                cabs_off[t] += 1
            elif power_usage[t] <= 5e+3:
                cabs_partial[t] += 1

    print("cabs off count:")
    for num in np.unique(cabs_off):
        print(num, (cabs_off == num).sum(), end=" - ")
    print("\ncabs partially off count:")
    for num in np.unique(cabs_partial):
        print(num, (cabs_partial == num).sum(), end=" - ")
    print()

    power_usage_slurm_tcabsfine = np.zeros(t_cabs_fine.values.size - 1)
    for i in range(len(power_usage_slurm_tcabsfine)):
        slice = df_power.loc[(df_power.Start <= t_cabs_fine[i]) & (df_power.End > t_cabs_fine[i + 1])]
        power_usage_slurm_tcabsfine[i] = slice.Power.sum()

    power_usage_allcabssum = np.array(power_usage_all_cabs).sum(axis=0)

    shift = np.diff(power_usage_slurm_tcabsfine).argmax() - np.diff(power_usage_allcabssum).argmax()
    print("Shifting by {}".format(shift))
    power_usage_slurm_tcabsfine_shifted = np.roll(power_usage_slurm_tcabsfine, -shift)
    power_usage_slurm_tcabsfine_shifted = power_usage_slurm_tcabsfine_shifted[:-shift]
    power_usage_allcabssum = power_usage_allcabssum[:-shift]
    cabs_off, cabs_partial = cabs_off[:-shift], cabs_partial[:-shift]

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    colors = iter(['b', 'g', 'r', 'y'])

    for num, cnt in np.array(np.unique(cabs_off, return_counts=True)).T:
        if cnt <= 1000:
            continue

        mask = (cabs_off == num)
        x = power_usage_slurm_tcabsfine_shifted[mask]
        y = power_usage_allcabssum[mask]

        model = RANSACRegressor()
        model.fit(x[:, None], y)
        preds = model.predict(x[:, None]).ravel()

        plt.scatter(
            x, y, c=next(colors), s=1,
            label="{} cabs off (RANSAC: {}x + {})".format(
                num, model.estimator_.coef_[0], model.estimator_.intercept_
            )
        )
        plt.plot(x, preds, c='k')

    for num, cnt in np.array(np.unique(cabs_partial, return_counts=True)).T:
        if num == 0 or cnt <= 1000:
            continue

        mask = (cabs_partial == num)
        x = power_usage_slurm_tcabsfine_shifted[mask]
        y = power_usage_allcabssum[mask]

        model = RANSACRegressor()
        model.fit(x[:, None], y)
        preds = model.predict(x[:, None]).ravel()

        plt.scatter(
            x, y, c=next(colors), s=1,
            label="{} cabs partially off (RANSAC: {}x + {})".format(
                num, model.estimator_.coef_[0], model.estimator_.intercept_
            )
        )
        plt.plot(x, preds, c='k')

    plt.xlabel("PSlurm (shifted)")
    plt.ylabel("PCabs")
    plt.xlim(left=-0.1e+6, right=4e+6)
    plt.ylim(bottom=-0.1e+6, top=4e+6)
    plt.legend()
    fig.tight_layout()
    fig.savefig("plots/powerusage_cabs_against_slurm_shifted_grouped.pdf")
    plt.show()

    print("Cropping out summer time and forcing intercept 1629 kW")
    for tick, time in enumerate(t_cabs_fine):
        if time.month == 10 and time.day == 30:
            start_tick, start_time = tick, time
            break

    power_usage_allcabssum = np.array(power_usage_all_cabs).sum(axis=0)
    x = power_usage_slurm_tcabsfine[start_tick:]
    y = power_usage_allcabssum[start_tick:] - 1629 * 1e+3

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    colors = iter(['b', 'g', 'r', 'y'])

    model = RANSACRegressor(estimator=LinearRegression(fit_intercept=False))
    model.fit(x[:, None], y)
    preds = model.predict(x[:, None]).ravel()

    plt.scatter(
        x, y, c=next(colors), s=1,
        label="{} cabs off (RANSAC: {}x + {})".format(
            num, model.estimator_.coef_[0], model.estimator_.intercept_
        )
    )
    plt.plot(x, preds, c='k')

    plt.xlabel("PSlurm")
    plt.ylabel("PCabs - pCabsIdle")
    plt.xlim(left=0.9 * min(x), right=1.1 * max(x))
    plt.ylim(bottom=0.9 * min(y), top=1.1 * max(y))
    plt.legend()
    fig.tight_layout()
    fig.savefig("plots/powerusage_cabs_against_slurm_gmt_crop.pdf")
    plt.show()

def power_usage_allocnodes(df, timesteps, cache, data_name):
    df_power = parse_cache(
        df, cache, data_name, "power_usage_allocnodes_df",
        ["JobID", "Start", "End", "ConsumedEnergyRaw", "AllocNodes", "AllocCPUS"]
    )

    df_power = df_power.loc[
        (
            (df_power.AllocNodes != "Unknown") & (df_power.AllocNodes.notna()) &
            (df_power.AllocNodes != "0")
        )
    ]
    df_power.AllocNodes = df_power.apply(
        lambda row: int(str(row.AllocNodes).replace("K", "000")), axis=1
    )

    # print(np.sort(df_power.AllocNodes.unique()))
    # print(max(df_power.AllocNodes.unique()), min(df_power.AllocNodes.unique()), df_power.AllocNodes.unique().mean())

    # Inclusive upper bounds of bins
    allocnodes_bins = [1, 4, 16, 64, 256, float("inf")]

    df_power["PowerPerNode"] = df_power.apply(
        lambda row: float(row.Power) / float(row.AllocNodes), axis=1
    )

    t = pd.date_range(min(df_power.Start), max(df_power.End), periods=timesteps)
    dates = matplotlib.dates.date2num(t[:-1])

    print("AllocNodes bin - #Jobs - Mean power usage per node over time full window")
    lower_lim = 0
    for upper_lim in allocnodes_bins:
        slice = df_power.loc[
            (df_power.AllocNodes > lower_lim) & (df_power.AllocNodes <= upper_lim)
        ]
        print("({},{}] - {} - {} W".format(
            lower_lim, upper_lim, len(slice), slice.PowerPerNode.mean())
        )

        power_usage = np.zeros(t.values.size - 1)
        for i in range(len(power_usage)):
            slice_time = slice.loc[(slice.Start <= t[i]) & (slice.End > t[i + 1])]
            power_usage[i] = slice_time.PowerPerNode.mean()

        plt.plot_date(dates, power_usage, '-', linewidth=0.8, label="({},{}]".format(lower_lim, upper_lim))
        lower_lim = upper_lim

    power_usage = np.zeros(t.values.size - 1)
    for i in range(len(power_usage)):
        slice_time = df_power.loc[(df_power.Start <= t[i]) & (df_power.End > t[i + 1])]
        power_usage[i] = slice_time.PowerPerNode.mean()

    plt.plot_date(dates, power_usage, 'k--', linewidth=0.8, label="mean")

    plt.legend()
    plt.ylabel("Mean Power per Node (W)")
    plt.title("ARCHER2 Job Power Consumption by AllocNodes")
    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    fig.tight_layout()
    fig.savefig("plots/powerusage_pernode_allocnodes.pdf")
    plt.show()

    print("AllocNodes bin - #Jobs - Mean total power usage")
    lower_lim = 0
    for upper_lim in allocnodes_bins:
        slice = df_power.loc[
            (df_power.AllocNodes > lower_lim) & (df_power.AllocNodes <= upper_lim)
        ]
        print("({},{}] - {} - {} W".format(
            lower_lim, upper_lim, len(slice),
            slice.ConsumedEnergyRaw.sum() / (max(slice.End) - min(slice.Start)).total_seconds()
        ))

        power_usage = np.zeros(t.values.size - 1)
        for i in range(len(power_usage)):
            slice_time = slice.loc[(slice.Start <= t[i]) & (slice.End > t[i + 1])]
            power_usage[i] = slice_time.Power.sum()

        plt.plot_date(dates, power_usage, '-', linewidth=0.8, label="({},{}]".format(lower_lim, upper_lim))
        lower_lim = upper_lim

    # power_usage = np.zeros(t.values.size - 1)
    # for i in range(len(power_usage)):
    #     slice_time = df_power.loc[(df_power.Start <= t[i]) & (df_power.End > t[i + 1])]
    #     power_usage[i] = slice_time.Power.sum()

    # plt.plot_date(dates, power_usage, 'k--', linewidth=0.8, label="Total")

    plt.legend()
    plt.ylabel("Total Power Usage (W)")
    plt.title("ARCHER2 Job Power Consumption by AllocNodes")
    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    fig.tight_layout()
    fig.savefig("plots/powerusage_total_allocnodes.pdf")
    plt.show()


def power_usage_user(df, timesteps, cache, data_name):
    df_power = parse_cache(
        df, cache, data_name, "power_usage_user_df",
        ["JobID", "Start", "End", "ConsumedEnergyRaw", "User"]
    )

    df_power = df_power.loc[
        (df_power.User != "Unknown") & (df_power.User.notna()) & (df_power.User != "")
    ]

    user_usage = defaultdict(int)
    for _, row in df_power.iterrows():
        user_usage[row.User] += row.ConsumedEnergyRaw

    top_users = list(reversed(sorted(user_usage.items(), key=lambda keyval: keyval[1])[-10:]))
    print("Top users:\n{}".format(top_users))

    t = pd.date_range(min(df_power.Start), max(df_power.End), periods=timesteps)
    dates = matplotlib.dates.date2num(t[:-1])

    for i_user, (user, _) in enumerate(top_users):
        slice = df_power.loc[(df_power.User == user)]

        power_usage = np.zeros(t.values.size - 1)
        for i_tick in range(len(power_usage)):
            slice_time = slice.loc[(slice.Start <= t[i_tick]) & (slice.End > t[i_tick + 1])]
            power_usage[i_tick] = slice_time.Power.sum()

        plt.plot_date(dates, power_usage, '-', linewidth=0.8, label="#{} user".format(i_user))

    power_usage = np.zeros(t.values.size - 1)
    slice = df_power.loc[(~df_power.User.isin([ user for user, _ in top_users ]))]
    for i in range(len(power_usage)):
        slice_time = slice.loc[(slice.Start <= t[i]) & (slice.End > t[i + 1])]
        power_usage[i] = slice_time.Power.sum()

    plt.plot_date(
        dates, power_usage, 'k-', linewidth=0.8,
        label="All other users ({} users)".format(int(len(user_usage) - len(top_users)))
    )

    # Plot only top users
    plt.legend()
    plt.ylabel("Total Power Usage (W)")
    plt.title("ARCHER2 Job Power Consumption by User")
    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    fig.tight_layout()
    fig.savefig("plots/powerusage_user.pdf")
    plt.show()

    for i_user, (user, _) in enumerate(top_users):
        slice = df_power.loc[(df_power.User == user)]

        power_usage = np.zeros(t.values.size - 1)
        for i_tick in range(len(power_usage)):
            slice_time = slice.loc[(slice.Start <= t[i_tick]) & (slice.End > t[i_tick + 1])]
            power_usage[i_tick] = slice_time.Power.sum()

        plt.plot_date(dates, power_usage, '-', linewidth=0.8, label="#{} user".format(i_user))

    plt.legend()
    plt.ylabel("Total Power Usage (W)")
    plt.title("ARCHER2 Job Power Consumption by User")
    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    fig.tight_layout()
    fig.savefig("plots/powerusage_user_toponly.pdf")
    plt.show()

    # Group Users
    top_1percent = sorted(user_usage.items(), key=lambda keyval: keyval[1])[
        -int(len(user_usage) * 0.01):
    ]
    top_1percent = list(reversed(top_1percent))
    top_2percent = sorted(user_usage.items(), key=lambda keyval: keyval[1])[
        -int(len(user_usage) * 0.02):
    ]
    top_2percent = list(reversed(top_2percent))
    top_5percent = sorted(user_usage.items(), key=lambda keyval: keyval[1])[
        -int(len(user_usage) * 0.05):
    ]
    top_5percent = list(reversed(top_5percent))

    # slice = df_power.loc[(df_power.User.isin([ user for user, _ in top_1percent ]))]
    # power_usage = np.zeros(t.values.size - 1)
    # for i_tick in range(len(power_usage)):
    #     slice_time = slice.loc[(slice.Start <= t[i_tick]) & (slice.End > t[i_tick + 1])]
    #     power_usage[i_tick] = slice_time.Power.sum()
    # plt.plot_date(
    #     dates, power_usage, '-', linewidth=0.8, label="Top 1% ({} users)".format(len(top_1percent))
    # )

    power_usage = np.zeros(t.values.size - 1)
    slice = df_power.loc[(~df_power.User.isin([ user for user, _ in top_5percent ]))]
    for i in range(len(power_usage)):
        slice_time = slice.loc[(slice.Start <= t[i]) & (slice.End > t[i + 1])]
        power_usage[i] = slice_time.Power.sum()
    plt.plot_date(
        dates, power_usage, 'k', linewidth=0.5,
        label="Bottom 95% ({} users)".format(int(len(user_usage) - len(top_5percent)))
    )

    slice = df_power.loc[(df_power.User.isin([ user for user, _ in top_5percent ]))]
    power_usage = np.zeros(t.values.size - 1)
    for i_tick in range(len(power_usage)):
        slice_time = slice.loc[(slice.Start <= t[i_tick]) & (slice.End > t[i_tick + 1])]
        power_usage[i_tick] = slice_time.Power.sum()
    plt.plot_date(
        dates, power_usage, '-', linewidth=0.5, label="Top 5% ({} users)".format(len(top_5percent))
    )

    slice = df_power.loc[(df_power.User.isin([ user for user, _ in top_2percent ]))]
    power_usage = np.zeros(t.values.size - 1)
    for i_tick in range(len(power_usage)):
        slice_time = slice.loc[(slice.Start <= t[i_tick]) & (slice.End > t[i_tick + 1])]
        power_usage[i_tick] = slice_time.Power.sum()
    plt.plot_date(
        dates, power_usage, '-', linewidth=0.5, label="Top 2% ({} users)".format(len(top_2percent))
    )

    plt.legend()
    plt.ylabel("Total Power Usage (W)")
    plt.title("ARCHER2 Job Power Consumption by User")
    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    fig.tight_layout()
    fig.savefig("plots/powerusage_user_percentages.pdf")
    plt.show()


# NOTE Suggest using a finer timestep since working with steps rather than jobs
def power_usage_cpufreq(df, timesteps, cache, data_name):
    # TODO  Figure out whats going on with AveCPUFreq, it has many unrealisitc frequencies (thousands of GHz and 10 Hz - not just anomalies but a solid dirstributions of these). Not something easy like needing to divide by AllocNodes or AllocCPUS). Ask Andy next meeting.
    df_power = parse_cache(
        df, cache, data_name, "power_usage_cpufreq_df",
        ["JobID", "Start", "End", "ConsumedEnergyRaw", "AveCPUFreq", "AllocNodes"],
        remove_steps=False # AveCPUFreq logged for steps only
    )

    print(df_power)
    print(df_power.AveCPUFreq.unique())
    print(df_power.AllocNodes.unique())

    print(len(df_power))

    # if cache == "load":
    if False:
        df_power = pd.read_pickle(
            "/work/y02/y02/awilkins/pandas_cache/{}/{}_cleaned.pkl".format(data_name, df_name)
        )
    else:
        df_power = df_power.loc[
            (
                (df_power.AveCPUFreq != "Unknown") & (df_power.AveCPUFreq.notna()) &
                (df_power.AveCPUFreq != "") & (df_power.AveCPUFreq != "0")
            )
        ]
        df_power = df_power.loc[
            (
                (df_power.AllocNodes != "Unknown") & (df_power.AllocNodes.notna()) &
                (df_power.AllocNodes != "0")
            )
        ]
        df_power.AveCpuFreq = df_power.AveCPUFreq.replace(
            {'K': 'e+03', 'M' : 'e+06', 'G' : 'e+09', 'T' : 'e+12'}, regex=True
        ).astype(float).astype(int)
        df_power.AveCPUFreq = df_power.AllocNodes.replace(
            {'K': 'e+03', 'M' : 'e+06', 'G' : 'e+09', 'T' : 'e+12' }, regex=True
        ).astype(float).astype(int)

    if cache == "save":
        df_power.to_pickle(
            "/work/y02/y02/awilkins/pandas_cache/{}/{}_cleaned.pkl".format(data_name, df_name)
        )

    print(len(df_power))
    print(df_power.AveCPUFreq.unique())
    print(df_power.AllocNodes.unique())

    print(np.sort(df_power.AveCPUFreq.unique()))
    print(max(df_power.AveCPUFreq.unique()), min(df_power.AveCPUFreq.unique()), df_power.AveCPUFreq.unique().mean())

    # Inclusive upper bounds of bins
    avecpufreq_bins = [1000000000, 2000000000, 3000000000, 4000000000, float("inf")]

    df_power["PowerPerNode"] = df_power.apply(
        lambda row: float(row.Power) / float(row.AllocNodes), axis=1
    )

    t = pd.date_range(min(df_power.Start), max(df_power.End), periods=timesteps)
    dates = matplotlib.dates.date2num(t[:-1])

    print("AveCPUFreq bin - #Jobs (steps) - Mean power usage per node over full time window")
    lower_lim = 0
    for upper_lim in avecpufreq_bins:
        slice = df_power.loc[
            (df_power.AveCPUFreq > lower_lim) & (df_power.AveCPUFreq <= upper_lim)
        ]
        print("({},{}] - {} - {} W".format(
            lower_lim, upper_lim, len(slice), slice.PowerPerNode.mean())
        )

        power_usage = np.zeros(t.values.size - 1)
        for i in range(len(power_usage)):
            slice_time = slice.loc[(slice.Start <= t[i]) & (slice.End > t[i + 1])]
            power_usage[i] = slice_time.PowerPerNode.mean()

        plt.plot_date(
            dates, power_usage, '-', linewidth=0.8,
            label="({:.2E},{{:.2E}}] Hz".format(lower_lim, upper_lim)
        )
        lower_lim = upper_lim

    power_usage = np.zeros(t.values.size - 1)
    for i in range(len(power_usage)):
        slice_time = df_power.loc[(df_power.Start <= t[i]) & (df_power.End > t[i + 1])]
        power_usage[i] = slice_time.PowerPerNode.mean()

    plt.plot_date(dates, power_usage, 'k--', linewidth=0.8, label="mean")

    plt.legend()
    plt.ylabel("Mean Power per Node (W)")
    plt.title("ARCHER2 Job Power Consumption by AveCPUFreq (only recorded for job steps)")
    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    fig.tight_layout()
    fig.savefig("plots/powerusage_cpufreq.pdf")
    plt.show()


def power_distributions(df, cache, data_name):
    df_power = parse_cache(
        df, cache, data_name, "power_distributions_df",
        ["JobID", "Start", "End", "ConsumedEnergyRaw", "Elapsed", "AllocNodes"]
    )

    df_power.AllocNodes = df_power.AllocNodes.astype(str)
    df_power.AllocNodes = df_power.AllocNodes.replace(
        { "K" : "e+03", "M" : "e+06", "G" : "e+09", "T" : "e+12" }, regex=True
    ).astype(float).astype(int)

    df_power["PowerPerNode"] = df_power.apply(
        lambda row: float(row.Power) / float(row.AllocNodes), axis=1
    )

    # print(df_power.AllocNodes.unique())

    bins_allocnodes = np.array(
        list(range(1, 7, 1)) + list(range(8, 17, 2)) + list(range(22, 101, 6)) +
        list(range(200, 1001, 100)) + list(range(1500, 3001, 500))
    )
    print("Using AllocNodes bins:\n{}".format(bins_allocnodes))

    bins_power = np.array(list(range(100, 801, 25)))
    print("Using PowerPerNodes bins:\n{}".format(bins_power))

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    h = ax.hist2d(
        df_power.AllocNodes, df_power.PowerPerNode, bins=[bins_allocnodes, bins_power],
        range=[[0, 3000], [100, 801]], cmap="jet"
    )
    ax.set_xscale("log")
    ax.set_title("Hist of PowerPerNode v AllocNodes for Jobs")
    ax.set_xlabel("AllocNodes")
    ax.set_ylabel("PowerPerNode (W)")
    fig.colorbar(h[3], ax=ax)
    fig.tight_layout()
    plt.show()

    df_power.Elapsed = df_power.Elapsed.apply(
        lambda row: timelimit_str_to_timedelta(row).total_seconds() / 60
    )

    print(df_power.Elapsed.sort_values().unique())

    bins_elapsed = np.array(
        (
            list(range(1, 61, 10)) + list(range(90, 361, 30)) + list(range(420, 961, 60)) +
            list(range(1200, 2881, 240)) + list(range(4320, 8641, 1440))
        ),
        dtype=float
    )
    bins_elapsed /= 60
    print("Using Elapsed bins:\n{}".format(bins_elapsed))

    print("Using PowerPerNodes bins:\n{}".format(bins_power))

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    h = ax.hist2d(
        df_power.Elapsed, df_power.PowerPerNode, bins=[bins_elapsed, bins_power],
        range=[[0, 144], [100, 801]], cmap="jet"
    )
    ax.set_xscale("log")
    ax.set_title("Hist of PowerPerNode v Elapsed for Jobs")
    ax.set_xlabel("Elapsed (hours)")
    ax.set_ylabel("PowerPerNode (W)")
    fig.colorbar(h[3], ax=ax)
    fig.tight_layout()
    plt.show()

    df_power_low = df_power.loc[(df_power.PowerPerNode < 300)]
    df_power_medlow = df_power.loc[(df_power.PowerPerNode >= 300) & (df_power.PowerPerNode < 400)]
    df_power_medhigh = df_power.loc[(df_power.PowerPerNode >= 400) & (df_power.PowerPerNode < 500)]
    df_power_high = df_power.loc[(df_power.PowerPerNode >= 500)]

    low_power_util = df_power_low.apply(
        lambda row: row.AllocNodes * (row.Elapsed / 60), axis=1
    ).sum()
    medlow_power_util = df_power_medlow.apply(
        lambda row: row.AllocNodes * (row.Elapsed / 60), axis=1
    ).sum()
    medhigh_power_util = df_power_medhigh.apply(
        lambda row: row.AllocNodes * (row.Elapsed / 60), axis=1
    ).sum()
    high_power_util = df_power_high.apply(
        lambda row: row.AllocNodes * (row.Elapsed / 60), axis=1
    ).sum()

    print(low_power_util, medlow_power_util, medhigh_power_util, high_power_util)

    norm_factor = low_power_util + medlow_power_util + medhigh_power_util + high_power_util

    print("Low power (<300W) job average system utilisation: {:.2f} %".format(
        low_power_util / norm_factor * 100
    ))
    print("Medium low power (300W<= < 400W) job average system utilisation: {:.2f} %".format(
        medlow_power_util / norm_factor * 100
    ))
    print("Medium high power (400W<= < 500W) job average system utilisation: {:.2f} %".format(
        medhigh_power_util / norm_factor * 100
    ))
    print("High power (>=500W) job average system utilisation: {:.2f} %".format(
        high_power_util / norm_factor * 100
    ))

    powers = [0] + list(range(300, 700, 25)) + [700]
    utils = []
    for i, power in enumerate(powers):
        if i == 0:
            util = df_power.loc[(df_power.PowerPerNode < power)].apply(
                lambda row: row.AllocNodes * (row.Elapsed / 60), axis=1
            ).sum()
        elif i + 1 == len(powers):
            util = df_power.loc[(df_power.PowerPerNode >= power)].apply(
                lambda row: row.AllocNodes * (row.Elapsed / 60), axis=1
            ).sum()
        else:
            util = df_power.loc[
                (df_power.PowerPerNode >= power) & (df_power.PowerPerNode < powers[i + 1])
            ].apply(
                lambda row: row.AllocNodes * (row.Elapsed / 60), axis=1
            ).sum()

        utils.append((util / norm_factor) * 100)

    x = np.array(list(range(len(powers))))
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.bar(x, utils)
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["<300"] +
        [ "{}-{}".format(powers[i], powers[i + 1]) for i in range(1, len(powers) - 1) ] + [">700"]
    )
    fig.tight_layout()
    plt.show()


def misc(df, timesteps, cache, data_name):
    df_power = parse_cache(
        df, cache, data_name, "misc_df",
        ["JobID", "Start", "End", "ConsumedEnergyRaw", "Submit"]
    )

    df_power.Submit = pd.to_datetime(df_power.Submit, format="%Y-%m-%dT%H:%M:%S")

    t = pd.date_range(min(df_power.Start), max(df_power.End), periods=timesteps)

    power_usage = np.zeros(t.values.size - 1)

    for i in range(len(power_usage)):
        slice = df_power.loc[(df_power.Submit <= t[i]) & (df_power.Start > t[i + 1])]
        power_usage[i] = len(slice)

    dates= matplotlib.dates.date2num(t[:-1])
    plt.plot_date(dates, power_usage, 'g', linewidth=0.8)
    plt.legend()
    plt.ylabel("Queue Size")
    plt.title("Job Queue on ARCHER2")
    fig = plt.gcf()
    fig.set_size_inches(14, 8)
    fig.tight_layout()
    fig.savefig("plots/queue_timeseries.pdf")
    plt.show()

    df_power["PowerPerNode"] = df_power.apply(
        lambda row: float(row.Power) / float(row.AllocNodes), axis=1
    )


def main(args):
    if args.cache != "load":
        df = pd.read_csv(
            args.data,
            delimiter='|',
            header=0,
            usecols=range(args.cols) if args.cols else None,
            nrows=args.rows if args.rows else None
        )
    else:
        df = None

    if args.power_usage:
        power_usage_plot(
            df, args.time_steps, args.cache, ".".join(os.path.basename(args.data).split(".")[:-1])
        )

    if args.power_usage_exitcodes:
        power_usage_exitcodes(
            df, args.time_steps, args.cache, ".".join(os.path.basename(args.data).split(".")[:-1])
        )

    if args.power_usage_cabs:
        power_usage_cabs(
            df, args.time_steps, args.cache,
            (
                ".".join(os.path.basename(args.data).split(".")[:-1]) + "_" +
                os.path.basename(args.power_usage_cabs)
            ),
            args.power_usage_cabs
        )

    if args.power_usage_allocnodes:
        power_usage_allocnodes(
            df, args.time_steps, args.cache, ".".join(os.path.basename(args.data).split(".")[:-1])
        )

    if args.power_usage_user:
        power_usage_user(
            df, args.time_steps, args.cache, ".".join(os.path.basename(args.data).split(".")[:-1])
        )

    if args.power_usage_cpufreq:
        power_usage_cpufreq(
            df, args.time_steps, args.cache, ".".join(os.path.basename(args.data).split(".")[:-1])
        )

    if args.power_distributions:
        power_distributions(df, args.cache, ".".join(os.path.basename(args.data).split(".")[:-1]))

    if args.misc:
        misc(
            df, args.time_steps, args.cache, ".".join(os.path.basename(args.data).split(".")[:-1])
        )

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("data", type=str)

    parser.add_argument("--power_usage", action='store_true')
    parser.add_argument("--power_usage_exitcodes", action='store_true')
    parser.add_argument(
        "--power_usage_cabs", type=str, default="",
        help="Location of cabinet power usage to plot alongside slurm power data"
    )
    parser.add_argument("--power_usage_allocnodes", action='store_true')
    parser.add_argument("--power_usage_user", action='store_true')
    parser.add_argument("--power_usage_cpufreq", action='store_true')
    parser.add_argument("--power_distributions", action='store_true')
    parser.add_argument("--misc", action='store_true')

    parser.add_argument("--cols", type=int, default=0, help="Number of cols to select")
    parser.add_argument(
        "--rows", type=int, default=0,
        help="Number of rows to read in from csv (mainly used for testing)"
    )
    parser.add_argument("--time_steps", type=int, default=1000)
    parser.add_argument("--cache", type=str, default="", help="How to use the cache (save|load)")

    return parser.parse_args()

if __name__ == '__main__':
    pd.options.display.float_format = '{:.0f}'.format

    main(parse_arguments())


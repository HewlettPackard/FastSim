import os, argparse
from datetime import timedelta
import dill as pickle
from collections import defaultdict

import matplotlib.dates
from cycler import cycler
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import numpy as np
from tqdm import tqdm

from controller import Controller
from fairshare import FairTree
from helpers import mkdir_p


# TODO
# - Total power usage plots

global bd_threshold
bd_threshold = timedelta(minutes=10)


def to_plot_or_not_to_plot(batch):
    if batch:
        plt.close()
    else:
        plt.show()

def metric_property_hist2d(job_history, job_to_metric_sim, job_to_metric_data, property, metric):
    job_property, sim_metrics, data_metrics = [], [], []

    # if property == "user_usage":
        

    for job in job_history:
        if job.ignore_in_eval:
            continue

        if property == "nodes":
            if job.nodes == 0:
                continue
            job_property.append(job.nodes)
        elif property == "reqtime":
            if job.reqtime.total_seconds() == 0:
                continue
            job_property.append(job.reqtime.total_seconds() / 60)
        else:
            raise NotImplementedError(property)

        sim_metrics.append(job_to_metric_sim(job))
        data_metrics.append(job_to_metric_data(job))

    if property == "nodes":
        bins_property = np.logspace(
            np.log10(min(job_property)), np.log10(max(job_property) + 0.5), 30, dtype=int
        )
        # Merge identical bins
        _, uniq_i = np.unique(bins_property, return_index=True)
        bins_property = bins_property[np.sort(uniq_i)]
    elif property == "reqtime":
        bins_property = np.logspace(np.log10(min(job_property)), np.log10(max(job_property)), 50)

    if metric == "bdslowdown":
        min_metric = 1.0
        nbins = 100
    elif metric == "wait_time":
        min_metric = max(min(min(sim_metrics), min(data_metrics)), 1 / 60 / 60)
        nbins = 50
    max_metric = np.percentile(sim_metrics + data_metrics, 99)
    bins_metric = np.logspace(np.log10(min_metric), np.log10(max_metric), nbins)

    h_data = np.histogram2d(job_property, data_metrics, bins=[bins_property, bins_metric])
    h_sim = np.histogram2d(job_property, sim_metrics, bins=[bins_property, bins_metric])

    h_data, h_data_edges = h_data[0], (h_data[1], h_data[2])
    h_sim, h_sim_edges = h_sim[0], (h_sim[1], h_sim[2])

    h_data_col_sums = h_data.sum(axis=1)
    h_data_col_sums[(h_data_col_sums == 0)] = 1
    h_data = (h_data.T / h_data_col_sums).T
    h_sim_col_sums = h_sim.sum(axis=1)
    h_sim_col_sums[(h_sim_col_sums == 0)] = 1
    h_sim = (h_sim.T / h_sim_col_sums).T

    return h_data, h_sim, bins_property, bins_metric


def top_assoc_waits(job_history, job_to_assoc, num_top, nodehour_threshold=None):
    assoc_sim_wait, assoc_data_wait = defaultdict(list), defaultdict(list)
    assoc_nodehours = defaultdict(float)
    for job in job_history:
        if job.ignore_in_eval:
            continue

        # assoc = assoc_tree.assocs[job.assoc].parent.parent.name
        assoc = job_to_assoc(job)
        sim_wait = (job.start - job.submit).total_seconds() / 60 / 60
        data_wait = (job.true_job_start - job.true_submit).total_seconds() / 60 / 60

        assoc_nodehours[assoc] += job.nodes * job.runtime.total_seconds() / 60 / 60
        assoc_sim_wait[assoc].append(sim_wait)
        assoc_data_wait[assoc].append(data_wait)

    if nodehour_threshold is not None:
        top_assocs = [
            assoc for assoc, nodehours in assoc_nodehours.items() if nodehours > nodehour_threshold
        ]
    else:
        top_assocs = [
            assoc
            for assoc, _ in (
                sorted(
                    assoc_nodehours.items(), key=lambda keyval: keyval[1], reverse=True
                )[:num_top]
            )
        ]

    assoc_sim_wait_mean = { assoc : np.mean(waits) for assoc, waits in assoc_sim_wait.items() }
    assoc_data_wait_mean = { assoc : np.mean(waits) for assoc, waits in assoc_data_wait.items() }
    assoc_sim_wait_err = { assoc : np.std(waits) for assoc, waits in assoc_sim_wait.items() }
    assoc_data_wait_err = { assoc : np.std(waits) for assoc, waits in assoc_data_wait.items() }
    sorted_sim_wait = [
        (assoc, wait_mean, assoc_sim_wait_err[assoc], len(assoc_sim_wait[assoc]))
        for assoc, wait_mean in sorted(
            assoc_sim_wait_mean.items(), key=lambda assoc_wait: assoc_wait[1], reverse=True
        )
            if assoc in top_assocs
    ]
    sorted_data_wait = [
        (assoc, wait_mean, assoc_data_wait_err[assoc], len(assoc_data_wait[assoc]))
        for assoc, wait_mean in sorted(
            assoc_data_wait_mean.items(), key=lambda assoc_wait: assoc_wait[1], reverse=True
        )
            if assoc in top_assocs
    ]

    print(
        "Sim top assoc by mean wait times:\n" +
        "\n".join(
            "{}.\t{}\t- {} += {} ({} jobs)".format(
                i + 1, assoc_wait[0], assoc_wait[1], assoc_wait[2], assoc_wait[3]
            )
            for i, assoc_wait in enumerate(sorted_sim_wait)
        ) +
        "\n"
    )
    print(
        "True top assoc by mean wait times:\n" +
        "\n".join(
            "{}.\t{}\t- {} += {} ({} jobs)".format(
                i + 1, assoc_wait[0], assoc_wait[1], assoc_wait[2], assoc_wait[3]
            )
            for i, assoc_wait in enumerate(sorted_data_wait)
        ) +
        "\n"
    )

    top_assocs.sort(key=lambda assoc: assoc_data_wait_mean[assoc], reverse=True)

    sim_mean_waits = [ assoc_sim_wait_mean[assoc] for assoc in top_assocs ]
    data_mean_waits = [ assoc_data_wait_mean[assoc] for assoc in top_assocs ]

    return top_assocs, sim_mean_waits, data_mean_waits


def group_waits(job_history, job_to_group):
    sim_group_waits, data_group_waits = defaultdict(list), defaultdict(list)
    for job in job_history:
        if job.ignore_in_eval:
            continue
        sim_group_waits[job_to_group(job)].append(
            (job.start - job.submit).total_seconds() / 60 / 60
        )
        data_group_waits[job_to_group(job)].append(
            (job.true_job_start - job.true_submit).total_seconds() / 60 / 60
        )

    print("Num Jobs by group:")
    print(
        " | ".join("{} - {}".format(group, len(waits)) for group, waits in sim_group_waits.items())
    )

    sim_group_mean_waits = { group : np.mean(waits) for group, waits in sim_group_waits.items() }
    data_group_mean_waits = { group : np.mean(waits) for group, waits in data_group_waits.items() }
    sorted_group = [
        group
        for group, _ in sorted(
            data_group_mean_waits.items(), key=lambda group_wait: group_wait[1], reverse=True
        )
    ]
    sim_mean_waits = [ sim_group_mean_waits[group] for group in sorted_group ]
    data_mean_waits = [ data_group_mean_waits[group] for group in sorted_group ]

    return sorted_group, data_mean_waits, sim_mean_waits


def rolling_window(job_history, job_to_metric, hours, window_hrs, data=False):
    if data:
        job_to_hour = lambda job: job.true_submit.replace(minute=0, second=0)
    else:
        job_to_hour = lambda job: job.submit.replace(minute=0, second=0)

    submit_hour_waits = defaultdict(list)
    for job in job_history:
        if job.ignore_in_eval:
            continue

        submit_hour_waits[job_to_hour(job)].append(job_to_metric(job))

    mean_wait_times_rolling_window = np.zeros(len(hours))
    mean_wait_times_rolling_window_err = np.zeros(len(hours))
    wait_times_rolling_window, wait_times_rolling_window_hour_lens = [], []
    for hr_num in range(window_hrs):
        wait_times_rolling_window += submit_hour_waits[hours[0] + timedelta(hours=hr_num)]
        wait_times_rolling_window_hour_lens.append(
            len(submit_hour_waits[hours[0] + timedelta(hours=hr_num)])
        )
    for i_hour, hour in enumerate(hours):
        if wait_times_rolling_window:
            mean_wait_times_rolling_window[i_hour] = np.mean(wait_times_rolling_window)
            mean_wait_times_rolling_window_err[i_hour] = np.std(wait_times_rolling_window)
        wait_times_rolling_window = (
            wait_times_rolling_window[wait_times_rolling_window_hour_lens.pop(0):]
        )
        wait_times_rolling_window += submit_hour_waits[hour + timedelta(hours=window_hrs)]
        wait_times_rolling_window_hour_lens.append(
            len(submit_hour_waits[hour + timedelta(hours=window_hrs)])
        )

    return mean_wait_times_rolling_window, mean_wait_times_rolling_window_err


def total_alloc_nodes(job_history):
    sim_max_end = max(job_history, key=lambda job: job.end).end
    sim_min_start = min(job_history, key=lambda job: job.start).start
    data_max_end_job = max(job_history, key=lambda job: job.true_job_start + job.runtime)
    data_max_end = data_max_end_job.true_job_start + data_max_end_job.runtime
    data_min_start = min(job_history, key=lambda job: job.true_job_start).true_job_start

    sim_alloc_nodes = np.zeros(int((sim_max_end - sim_min_start).total_seconds() / 60))
    data_alloc_nodes = np.zeros(int((data_max_end - data_min_start).total_seconds() / 60))

    for job in tqdm(job_history):
        l_mins = int((job.start - sim_min_start).total_seconds() / 60) + 1
        u_mins = int((job.end - sim_min_start).total_seconds() / 60)
        sim_alloc_nodes[l_mins:u_mins] += job.nodes

        l_mins = int((job.true_job_start - data_min_start).total_seconds() / 60) + 1
        u_mins = int((job.true_job_start + job.runtime - data_min_start).total_seconds() / 60)
        data_alloc_nodes[l_mins:u_mins] += job.nodes

    pad = 24 * 60 * 60
    print("Sim mean(max) allocations nodes = {} +- {} ({})".format(
        np.mean(sim_alloc_nodes[pad:-pad]), np.std(sim_alloc_nodes[pad:-pad]),
        np.max(sim_alloc_nodes)
    ))
    print("Data mean(max) allocations nodes = {} +- {} ({})".format(
        np.mean(data_alloc_nodes[pad:-pad]), np.std(data_alloc_nodes[pad:-pad]),
        np.max(data_alloc_nodes)
    ))

    data_minutes = [
        data_min_start + timedelta(minutes=min_num) for min_num in range(len(data_alloc_nodes))
    ]
    sim_minutes = [
        sim_min_start + timedelta(minutes=min_num) for min_num in range(len(sim_alloc_nodes))
    ]

    return data_alloc_nodes, data_minutes, sim_alloc_nodes, sim_minutes


def q_size(job_history):
    sim_min_submit = min(job_history, key=lambda job: job.submit).submit
    data_min_submit = min(job_history, key=lambda job: job.true_submit).true_submit
    sim_max_start = max(job_history, key=lambda job: job.start).start
    data_max_start = max(job_history, key=lambda job: job.true_job_start).true_job_start

    sim_q_length = np.zeros(int((sim_max_start - sim_min_submit).total_seconds() / 60))
    data_q_length = np.zeros(int((data_max_start - data_min_submit).total_seconds() / 60))
    sim_q_length_nodes = np.zeros(int((sim_max_start - sim_min_submit).total_seconds() / 60))
    data_q_length_nodes = np.zeros(int((data_max_start - data_min_submit).total_seconds() / 60))

    for job in tqdm(job_history):
        if job.ignore_in_eval:
            continue

        l_mins = int((job.submit - sim_min_submit).total_seconds() / 60) + 1
        u_mins = int((job.start - sim_min_submit).total_seconds() / 60)
        sim_q_length[l_mins:u_mins] += 1
        sim_q_length_nodes[l_mins:u_mins] += job.nodes

        l_mins = int((job.true_submit - data_min_submit).total_seconds() / 60) + 1
        u_mins = int((job.true_job_start - data_min_submit).total_seconds() / 60)
        data_q_length[l_mins:u_mins] += 1
        data_q_length_nodes[l_mins:u_mins] += job.nodes

    pad = 24 * 60 * 60
    print("Sim mean(max) queue size (jobs) = {} +- {} ({})".format(
        np.mean(sim_q_length[pad:-pad]), np.std(sim_q_length[pad:-pad]), np.max(sim_q_length)
    ))
    print("Data mean(max) queue size (jobs) = {} +- {} ({})".format(
        np.mean(data_q_length[pad:-pad]), np.std(data_q_length[pad:-pad]), np.max(data_q_length)
    ))
    print("Sim mean(max) queue size (nodes) = {} +- {} ({})".format(
        np.mean(sim_q_length_nodes[pad:-pad]), np.std(sim_q_length_nodes[pad:-pad]),
        np.max(sim_q_length_nodes)
    ))
    print("Data mean(max) queue size (nodes) = {} +- {} ({})".format(
        np.mean(data_q_length_nodes[pad:-pad]), np.std(data_q_length_nodes[pad:-pad]),
        np.max(data_q_length_nodes)
    ))

    data_minutes = [
        data_min_submit + timedelta(minutes=min_num) for min_num in range(len(data_q_length))
    ]
    sim_minutes = [
        sim_min_submit + timedelta(minutes=min_num) for min_num in range(len(sim_q_length))
    ]

    return (
        data_q_length, data_q_length_nodes, data_minutes, sim_q_length, sim_q_length_nodes,
        sim_minutes
    )


def mean_metrics(job_history, controller):
    data_bd_slowdowns = [
        max(
            (job.true_job_start + job.runtime - job.true_submit) / max(job.runtime, bd_threshold),
            1
        )
        for job in job_history
            if not job.ignore_in_eval
    ]
    sim_bd_slowdowns = [
        max((job.end - job.submit) / max(job.runtime, bd_threshold), 1)
        for job in job_history
            if not job.ignore_in_eval
    ]
    data_wait_times = [
        (job.true_job_start - job.true_submit).total_seconds() / 60 / 60
        for job in job_history 
            if not job.ignore_in_eval
    ]
    sim_wait_times = [
        (job.start - job.submit).total_seconds() / 60 / 60
        for job in job_history 
            if not job.ignore_in_eval
    ]

    print(
        "True mean bd slowdown={}+-{} (total = {})\n".format(
            np.mean(data_bd_slowdowns), np.std(data_bd_slowdowns), np.sum(data_bd_slowdowns)
        ) +
        "Sim mean bd slowdown={}+-{} (total = {})\n".format(
            np.mean(sim_bd_slowdowns), np.std(sim_bd_slowdowns), np.sum(sim_bd_slowdowns)
        ) +
        "True mean wait time={}+-{} hrs (total = {} hrs)\n".format(
            np.mean(data_wait_times), np.std(data_wait_times), np.sum(data_wait_times)
        ) +
        "Sim mean wait time={}+-{} hrs (total = {} hrs)\n".format(
            np.mean(sim_wait_times), np.std(sim_wait_times), np.sum(sim_wait_times)
        )
    )

    return data_bd_slowdowns, data_wait_times, sim_bd_slowdowns, sim_wait_times


def main(args):
    PLOT_DIR = os.path.join(
        args.plot_dir, "-".join(os.path.basename(sim).split(".")[0] for sim in args.sim)
    )
    mkdir_p(PLOT_DIR)

    # TODO Do I still want a FIFO baseline to compare with?

    controllers = []
    for sim in args.sim:
        with open(sim, "rb") as f:
            controllers.append(pickle.load(f))

    max_submit = max(controllers[0].job_history, key=lambda job: job.true_submit).true_submit

    job_histories = [
        [
            job
            for job in controller.job_history
                if (
                    controller.init_time + timedelta(days=args.days_ignore) < job.true_submit <
                    max_submit - timedelta(days=args.days_ignore)
                )
        ]
        for controller in controllers
    ]

    # Some things (including truth data) should be the same across all controllers so want a single
    # one to reference for this stuff
    controller, job_history = controllers[0], job_histories[0]

    job_history = [
        job for job in controller.job_history if (
            controller.init_time + timedelta(days=4) < job.true_submit <
            max_submit - timedelta(days=4)
        )
    ]

    print(
        "Ignoring {} out of {} jobs in evaluation\n".format(
            sum(1 for job in job_history if job.ignore_in_eval), len(job_history)
        )
    )

    assoc_tree = FairTree(
        controller.config.assocs_dump, timedelta(minutes=1), timedelta(minutes=1),
        controller.init_time, set(), 0, controller.partitions
    )

    data_bd_slowdowns, data_wait_times, sim_bd_slowdowns, sim_wait_times = mean_metrics(
        job_history, controller
    )

    if "bdslowdowns_hist2d" in args.plots:
        job_to_bdslowdown_sim = lambda job: (
            max((job.end - job.submit) / max(job.runtime, bd_threshold), 1)
        )
        job_to_bdslowdown_data = lambda job: (
            max(
                (
                    (job.true_job_start + job.runtime - job.true_submit) /
                    max(job.runtime, bd_threshold)
                ),
                1
            )
        )

        h_data, h_sim, bins_allocnodes, bins_bdslowdowns = metric_property_hist2d(
            job_history, job_to_bdslowdown_sim, job_to_bdslowdown_data, "nodes", "bdslowdown"
        )

        fig, ax = plt.subplots(1, 2, figsize=(16, 8))

        ax[0].pcolormesh(bins_allocnodes, bins_bdslowdowns, h_data.T, vmin=0.0, vmax=1.0)
        ax[1].pcolormesh(bins_allocnodes, bins_bdslowdowns, h_sim.T, vmin=0.0, vmax=1.0)

        ax[0].set_yscale("log")
        ax[0].set_xscale("log")
        ax[0].set_xlabel("Nodes")
        ax[0].set_ylabel("Bounded Slowdown")
        ax[0].set_title("Data")
        ax[1].set_yscale("log")
        ax[1].set_xscale("log")
        ax[1].set_xlabel("Nodes")
        ax[1].set_ylabel("Bounded Slowdown")
        ax[1].set_title("Sim")

        fig.savefig(
            os.path.join(PLOT_DIR, "allocnodes_bdslowdowns_hist2d{}.pdf".format(args.save_suffix))
        )
        to_plot_or_not_to_plot(args.batch)

        h_data, h_sim, bins_reqtime, bins_bdslowdowns= metric_property_hist2d(
            job_history, job_to_bdslowdown_sim, job_to_bdslowdown_data, "reqtime", "bdslowdown"
        )

        fig, ax = plt.subplots(1, 2, figsize=(16, 8))

        ax[0].pcolormesh(bins_reqtime, bins_bdslowdowns, h_data.T, vmin=0.0, vmax=1.0)
        ax[1].pcolormesh(bins_reqtime, bins_bdslowdowns, h_sim.T, vmin=0.0, vmax=1.0)

        ax[0].set_yscale("log")
        ax[0].set_xscale("log")
        ax[0].set_xlabel("Req Time (mins)")
        ax[0].set_ylabel("Bounded Slowdown")
        ax[0].set_title("Data")
        ax[1].set_yscale("log")
        ax[1].set_xscale("log")
        ax[1].set_xlabel("Req Time (mins)")
        ax[1].set_ylabel("Bounded Slowdown")
        ax[1].set_title("Sim")

        fig.savefig(
            os.path.join(PLOT_DIR, "reqtime_bdslowdowns_hist2d{}.pdf".format(args.save_suffix))
        )
        to_plot_or_not_to_plot(args.batch)

    if "wait_times_hist2d" in args.plots:
        job_to_wait_sim = lambda job: (job.start - job.submit).total_seconds() / 60
        job_to_wait_data = lambda job: (job.true_job_start - job.true_submit).total_seconds() / 60

        h_data, h_sim, bins_allocnodes, bins_wait_times = metric_property_hist2d(
            job_history, job_to_wait_sim, job_to_wait_data, "nodes", "wait_time"
        )

        fig, ax = plt.subplots(1, 2, figsize=(16, 8))

        ax[0].pcolormesh(bins_allocnodes, bins_wait_times, h_data.T, vmin=0.0, vmax=1.0)
        ax[1].pcolormesh(bins_allocnodes, bins_wait_times, h_sim.T, vmin=0.0, vmax=1.0)

        ax[0].set_yscale("log")
        ax[0].set_xscale("log")
        ax[0].set_xlabel("Nodes")
        ax[0].set_ylabel("Wait (mins)")
        ax[0].set_title("Data")
        ax[1].set_yscale("log")
        ax[1].set_xscale("log")
        ax[1].set_xlabel("Nodes")
        ax[1].set_ylabel("Wait (mins)")
        ax[1].set_title("Sim")

        fig.savefig(
            os.path.join(PLOT_DIR, "allocnodes_wait_time_hist2d{}.pdf".format(args.save_suffix))
        )
        to_plot_or_not_to_plot(args.batch)

        h_data, h_sim, bins_reqtime, bins_wait_times = metric_property_hist2d(
            job_history, job_to_wait_sim, job_to_wait_data, "reqtime", "wait_time"
        )

        fig, ax = plt.subplots(1, 2, figsize=(16, 8))

        ax[0].pcolormesh(bins_reqtime, bins_wait_times, h_data.T, vmin=0.0, vmax=1.0
        )
        ax[1].pcolormesh(bins_reqtime, bins_wait_times, h_sim.T, vmin=0.0, vmax=1.0
        )

        ax[0].set_yscale("log")
        ax[0].set_xscale("log")
        ax[0].set_xlabel("Req Time (mins)")
        ax[0].set_ylabel("Wait (mins)")
        ax[0].set_title("Data")
        ax[1].set_yscale("log")
        ax[1].set_xscale("log")
        ax[1].set_xlabel("Req Time (mins)")
        ax[1].set_ylabel("Wait (mins)")
        ax[1].set_title("Sim")

        fig.savefig(
            os.path.join(PLOT_DIR, "reqtime_wait_time_hist2d{}.pdf".format(args.save_suffix))
        )
        to_plot_or_not_to_plot(args.batch)

    if "top_projs" in args.plots:
        job_to_proj = lambda job: assoc_tree.assocs[job.assoc].parent.parent.name
        top_projs, sim_mean_waits, data_mean_waits = top_assoc_waits(job_history, job_to_proj, 15)
        x = np.arange(len(top_projs))

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        sim_bars = ax.bar(x - 2 * 0.2 / 3, sim_mean_waits, 0.2, label="Sim")
        data_bars = ax.bar(x + 2 * 0.2 / 3, data_mean_waits, 0.2, label="Data")
        ax.set_ylabel("Mean Wait Time (hrs)", fontsize=18)
        ax.set_xticks(x, top_projs)
        ax.legend()
        ax.bar_label(sim_bars, padding=3, fmt="%.1f")
        ax.bar_label(data_bars, padding=3, fmt="%.1f")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "top_projs_mean_waits{}.pdf".format(args.save_suffix)))
        to_plot_or_not_to_plot(args.batch)

    if "top_accounts" in args.plots:
        job_to_acc = lambda job: assoc_tree.assocs[job.assoc].parent.name
        top_accs, sim_mean_waits, data_mean_waits = top_assoc_waits(job_history, job_to_acc, 15)
        x = np.arange(len(top_accs))

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        sim_bars = ax.bar(x - 2 * 0.2 / 3, sim_mean_waits, 0.2, label="Sim")
        data_bars = ax.bar(x + 2 * 0.2 / 3, data_mean_waits, 0.2, label="Data")
        ax.set_ylabel("Mean Wait Time (hrs)", fontsize=18)
        ax.set_xticks(x, top_accs)
        ax.legend()
        ax.bar_label(sim_bars, padding=3, fmt="%.1f")
        ax.bar_label(data_bars, padding=3, fmt="%.1f")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "top_accs_mean_waits{}.pdf".format(args.save_suffix)))
        to_plot_or_not_to_plot(args.batch)

    if "top_users" in args.plots:
        job_to_usr = lambda job: assoc_tree.assocs[job.assoc].name
        top_usr, sim_mean_waits, data_mean_waits = top_assoc_waits(job_history, job_to_usr, 15)
        x = np.arange(len(top_usr))

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        sim_bars = ax.bar(x - 2 * 0.2 / 3, sim_mean_waits, 0.2, label="Sim")
        data_bars = ax.bar(x + 2 * 0.2 / 3, data_mean_waits, 0.2, label="Data")
        ax.set_ylabel("Mean Wait Time (hrs)", fontsize=18)
        ax.set_xticks(x, top_usr)
        ax.legend()
        ax.bar_label(sim_bars, padding=3, fmt="%.1f")
        ax.bar_label(data_bars, padding=3, fmt="%.1f")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "top_usr_mean_waits{}.pdf".format(args.save_suffix)))
        to_plot_or_not_to_plot(args.batch)

    if "qos_waits" in args.plots:
        job_to_qos = lambda job: job.qos.name
        sorted_qos, data_mean_waits, sim_mean_waits = group_waits(job_history, job_to_qos)

        print(
            "\nlargescale jobs "
            "(id - nodes - submit - elapsed - reqtime - sim wait - true wait - user - account)"
        )
        for job in job_history:
            if job.ignore_in_eval:
                continue

            if job.qos.name == "largescale":
                print(
                    job.id, job.nodes, job.true_submit,
                    job.runtime, job.reqtime, (job.start - job.submit).round(freq="S"),
                    job.true_job_start - job.true_submit, job.user, job.account,
                    sep=" - "
                )

        x = np.arange(len(sim_mean_waits))

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        sim_bars = ax.bar(x - 2 * 0.2 / 3, sim_mean_waits, 0.2, label="Sim")
        data_bars = ax.bar(x + 2 * 0.2 / 3, data_mean_waits, 0.2, label="Data")
        ax.set_ylabel("Mean Wait Time (hrs)", fontsize=18)
        ax.set_xticks(x, sorted_qos)
        ax.legend()
        ax.bar_label(sim_bars, padding=3, fmt="%.1f")
        ax.bar_label(data_bars, padding=3, fmt="%.1f")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "qos_mean_waits{}.pdf".format(args.save_suffix)))
        to_plot_or_not_to_plot(args.batch)

    if "partition_waits" in args.plots:
        job_to_partition = lambda job: job.partition.name
        sorted_partition, data_mean_waits, sim_mean_waits = group_waits(job_history, job_to_qos)

        x = np.arange(len(sim_mean_waits))

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        sim_bars = ax.bar(x - 2 * 0.2 / 3, sim_mean_waits, 0.2, label="Sim")
        data_bars = ax.bar(x + 2 * 0.2 / 3, data_mean_waits, 0.2, label="Data")
        ax.set_ylabel("Mean Wait Time (hrs)", fontsize=18)
        ax.set_xticks(x, sorted_partition)
        ax.legend()
        ax.bar_label(sim_bars, padding=3, fmt="%.1f")
        ax.bar_label(data_bars, padding=3, fmt="%.1f")
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "partition_mean_waits{}.pdf".format(args.save_suffix)))
        to_plot_or_not_to_plot(args.batch)

    if "rolling_window" in args.plots or "rolling_window_qos" in args.plots:
        hours = [
            controllers[0].init_time.replace(minute=0, second=0) + timedelta(hours=hr)
            for hr in range(
                int(
                    (
                        max_submit - timedelta(days=args.rolling_window_days) -
                        controllers[0].init_time
                    ).total_seconds() /
                    (60 * 60)
                )
            )
        ]
        window_hrs = int(args.rolling_window_days * 24)

        # Rolling window mean wait time
        job_to_wait_sim = lambda job: (job.start - job.submit).total_seconds() / 60 / 60

        sims_mean_wait_times_rolling_window, sims_mean_wait_times_rolling_window_err = [], []

        for job_history in job_histories:
            means, errs = rolling_window(job_history, job_to_wait_sim, hours, window_hrs)
            sims_mean_wait_times_rolling_window.append(means)
            sims_mean_wait_times_rolling_window_err.append(errs)

        if not args.no_data_comparison:
            job_to_wait_data = lambda job: (
                (job.true_job_start - job.true_submit).total_seconds() / 60 / 60
            )

            means, errs = rolling_window(
                job_history, job_to_wait_data, hours, window_hrs, data=True
            )
            data_mean_wait_times_rolling_window = means
            data_mean_wait_times_rolling_window_err = errs

        hour_dates = matplotlib.dates.date2num([ hour + timedelta(days=7) for hour in hours ])

        # The plot with the error band will be horrible for multiple experiments at once
        if len(sims_mean_wait_times_rolling_window) == 1:
            fig = plt.figure(1, figsize=(12, 8))

            sim_mean_wait_times_rolling_window_err = sims_mean_wait_times_rolling_window_err[0]
            sim_mean_wait_times_rolling_window = sims_mean_wait_times_rolling_window[0]

            ax_big = fig.add_axes((.1, .32, .8, .58))
            ax_big.plot_date(
                hour_dates, sim_mean_wait_times_rolling_window, 'b', label="Sim", linewidth=1
            )
            ax_big.plot_date(
                hour_dates, data_mean_wait_times_rolling_window, 'k', label="Data", linewidth=1
            )

            plt.legend()

            ax_small = fig.add_axes((.1, .1, .8, .2))
            ax_small.plot_date(
                hour_dates, sim_mean_wait_times_rolling_window_err, 'b', label="_", linewidth=1
            )
            ax_small.plot_date(
                hour_dates, data_mean_wait_times_rolling_window_err, 'k', label="_", linewidth=1
            )

            ax_big.set_ylabel("Mean Wait Time")
            ax_big.set_xticklabels([])
            ax_big.set_ylim(bottom=0.0)
            ax_small.set_xlabel("Middle Hour of Rolling Window")
            ax_small.set_ylabel("Std dev Wait Time")
            ax_small.set_ylim(bottom=0.0)

            fig.tight_layout()
            fig.savefig(
                os.path.join(PLOT_DIR, "wait_times_rolling_window{}.pdf".format(args.save_suffix))
            )
            to_plot_or_not_to_plot(args.batch)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        for sim_mean_wait_times_rolling_window, label in zip(
            sims_mean_wait_times_rolling_window, args.labels
        ):
            ax.plot_date(
                hour_dates, sim_mean_wait_times_rolling_window, "-", label=label, linewidth=1.0
            )
        if not args.no_data_comparison:
            ax.plot_date(
                hour_dates, data_mean_wait_times_rolling_window, "k-", label="Data",
                linewidth=2.0
            )

        ax.set_ylabel("Mean Wait Time in Window", fontsize=18)
        ax.set_xlabel("Middle Hour of 2 Week Rolling Window", fontsize=18)
        ax.grid(axis="x")
        plt.legend()

        fig.tight_layout()
        fig.savefig(
            os.path.join(
                PLOT_DIR, "wait_times_rolling_window_noerr{}.pdf".format(args.save_suffix)
            )
        )
        to_plot_or_not_to_plot(args.batch)

        # Rolling window mean bd slowdown
        job_to_bdslowdown_sim = lambda job: (
            max((job.end - job.submit) / max(job.runtime, bd_threshold), 1)
        )

        sims_mean_bdslowdowns_rolling_window, sims_mean_bdslowdowns_rolling_window_err = [], []

        for job_history in job_histories:
            means, errs = rolling_window(job_history, job_to_bdslowdown_sim, hours, window_hrs)
            sims_mean_bdslowdowns_rolling_window.append(means)
            sims_mean_bdslowdowns_rolling_window_err.append(errs)

        if not args.no_data_comparison:
            job_to_bdslowdown_data = lambda job: (
                max(
                    (
                        (job.true_job_start + job.runtime - job.true_submit) /
                        max(job.runtime, bd_threshold)
                    ),
                    1
                )
            )

            means, errs = rolling_window(
                job_history, job_to_bdslowdown_data, hours, window_hrs, data=True
            )
            data_mean_bdslowdowns_rolling_window = means
            data_mean_bdslowdowns_rolling_window_err = errs

        # The plot with the error band will be horrible for multiple experiments at once
        if len(sims_mean_bdslowdowns_rolling_window) == 1:
            fig = plt.figure(1, figsize=(12, 8))

            sim_mean_bdslowdowns_rolling_window_err = sims_mean_bdslowdowns_rolling_window_err[0]
            sim_mean_bdslowdowns_rolling_window = sims_mean_bdslowdowns_rolling_window[0]

            ax_big = fig.add_axes((.1, .3, .8, .6))
            ax_big.plot_date(
                hour_dates, sim_mean_bdslowdowns_rolling_window, 'b', label="Sim", linewidth=1
            )
            ax_big.plot_date(
                hour_dates, data_mean_bdslowdowns_rolling_window, 'k', label="Data", linewidth=1
            )

            plt.legend()

            ax_small = fig.add_axes((.1, .1, .8, .2))
            ax_small.plot_date(
                hour_dates, sim_mean_bdslowdowns_rolling_window_err, 'b', label="_", linewidth=1
            )
            ax_small.plot_date(
                hour_dates, data_mean_bdslowdowns_rolling_window_err, 'k', label="_", linewidth=1
            )

            ax_big.set_ylabel("Mean Bounded Slowdown")
            ax_big.set_xticklabels([])
            ax_big.set_ylim(bottom=1.0)
            ax_small.set_xlabel("Middle Hour of Rolling Window")
            ax_small.set_ylabel("Std dev Bounded Slowdown")
            ax_small.set_ylim(bottom=0.0)

            fig.tight_layout()
            fig.savefig(
                os.path.join(PLOT_DIR, "bd_slowdowns_rolling_window{}.pdf".format(args.save_suffix))
            )
            to_plot_or_not_to_plot(args.batch)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        for sim_mean_bdslowdowns_rolling_window, label in zip(
            sims_mean_bdslowdowns_rolling_window, args.labels
        ):
            ax.plot_date(
                hour_dates, sim_mean_bdslowdowns_rolling_window, "-", label=label, linewidth=1.0
            )
        if not args.no_data_comparison:
            ax.plot_date(
                hour_dates, data_mean_bdslowdowns_rolling_window, "k-", label="Data",
                linewidth=2.0
            )

        ax.set_ylabel("Mean Bounded Slowdown in Window", fontsize=18)
        ax.set_xlabel("Middle Hour of 2 Week Rolling Window", fontsize=18)
        ax.grid(axis="x")

        plt.legend()
        fig.tight_layout()
        fig.savefig(
            os.path.join(
                PLOT_DIR, "bd_slowdowns_rolling_window_noerr{}.pdf".format(args.save_suffix)
            )
        )
        to_plot_or_not_to_plot(args.batch)

    if "rolling_window_qos" in args.plots:
        qos_sim_mean_wait_times_rolling_window = {
            "all" : sims_mean_wait_times_rolling_window[0]
        }
        qos_data_mean_wait_times_rolling_window = {
            "all" : data_mean_wait_times_rolling_window
        }
        qos_sim_mean_bdslowdowns_rolling_window = {
            "all" : sims_mean_bdslowdowns_rolling_window[0]
        }
        qos_data_mean_bdslowdowns_rolling_window = {
            "all" : data_mean_bdslowdowns_rolling_window
        }

        qos_job_history = defaultdict(list)

        for job in job_history:
            qos_job_history[job.qos.name].append(job)

        for qos, job_history in qos_job_history.items():
            # short goes through instantly and there are too few largescale
            if qos == "largescale" or qos == "short":
                continue

            means, _ = rolling_window(job_history, job_to_wait_sim, hours, window_hrs)
            qos_sim_mean_wait_times_rolling_window[qos] = means
            means, _ = rolling_window(job_history, job_to_wait_data, hours, window_hrs, data=True)
            qos_data_mean_wait_times_rolling_window[qos] = means

            means, _ = rolling_window(job_history, job_to_bdslowdown_sim, hours, window_hrs)
            qos_sim_mean_bdslowdowns_rolling_window[qos] = means
            means, _ = rolling_window(
                job_history, job_to_bdslowdown_data, hours, window_hrs, data=True
            )
            qos_data_mean_bdslowdowns_rolling_window[qos] = means

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        my_cycler = (
            cycler(color=[
                "r", "r", "g", "g", "b", "b", "c", "c", "m", "m", "y", "y", "tab:orange",
                "tab:orange"
            ])
        )

        # ax.plot_date(
        #     hour_dates, qos_sim_mean_wait_times_rolling_window.pop("all"), fmt="-k", label="all"
        # )
        # ax.plot_date(
        #     hour_dates, qos_data_mean_wait_times_rolling_window.pop("all"), fmt="--k", label="_"
        # )
        ax.set_prop_cycle(my_cycler)
        for qos in qos_sim_mean_wait_times_rolling_window:
            sim_mean_wait_times_rolling_window = qos_sim_mean_wait_times_rolling_window[qos]
            data_mean_wait_times_rolling_window = qos_data_mean_wait_times_rolling_window[qos]

            ax.plot_date(
                hour_dates, qos_sim_mean_wait_times_rolling_window[qos], fmt="-", label=qos
            )
            ax.plot_date(
                hour_dates, qos_data_mean_wait_times_rolling_window[qos], fmt="--", label="_"
            )

        ax.set_ylabel("Mean Wait Time", fontsize=18)
        ax.set_xlabel("Middle Hour of Rolling Window", fontsize=18)
        ax.set_yscale("log")
        plt.legend()

        fig.tight_layout()
        fig.savefig(
            os.path.join(
                PLOT_DIR, "wait_times_rolling_window_byqos{}.pdf".format(args.save_suffix)
            )
        )
        to_plot_or_not_to_plot(args.batch)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # ax.plot_date(
        #     hour_dates, qos_sim_mean_bdslowdowns_rolling_window.pop("all"), fmt="-k", label="all"
        # )
        # ax.plot_date(
        #     hour_dates, qos_data_mean_bdslowdowns_rolling_window.pop("all"), fmt="--k", label="_"
        # )
        ax.set_prop_cycle(my_cycler)
        for qos in qos_sim_mean_bdslowdowns_rolling_window:
            sim_mean_bdslowdowns_rolling_window = qos_sim_mean_bdslowdowns_rolling_window[qos]
            data_mean_bdslowdowns_rolling_window = qos_data_mean_bdslowdowns_rolling_window[qos]

            ax.plot_date(
                hour_dates, qos_sim_mean_bdslowdowns_rolling_window[qos], fmt="-", label=qos
            )
            ax.plot_date(
                hour_dates, qos_data_mean_bdslowdowns_rolling_window[qos], fmt="--", label="_"
            )

        ax.set_ylabel("Mean Bounded Slowdown", fontsize=18)
        ax.set_xlabel("Middle Hour of Rolling Window", fontsize=18)
        ax.set_yscale("log")
        plt.legend()

        fig.tight_layout()
        fig.savefig(
            os.path.join(
                PLOT_DIR, "bd_slowdowns_rolling_window_byqos{}.pdf".format(args.save_suffix)
            )
        )
        to_plot_or_not_to_plot(args.batch)

    if "total_allocnodes_timeseries" in args.plots:
        data_alloc_nodes, data_minutes, sim_alloc_nodes, sim_minutes = total_alloc_nodes(
            job_history
        )

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        ax.plot_date(sim_minutes, sim_alloc_nodes, 'g', label="Sim", linewidth=0.75, alpha=0.8)
        ax.plot_date(data_minutes, data_alloc_nodes, 'r', label="Data", linewidth=0.75, alpha=0.8)

        ax.set_xlabel("Date (minute resolution)", fontsize=18)
        ax.set_ylabel("Number of Allocated Nodes", fontsize=18)
        ax.set_ylim(
            max(data_alloc_nodes) * 0.5 if max(data_alloc_nodes) > 2000 else 0,
            len(controller.partitions.nodes)
        )
        ax.grid(axis="y")
        plt.legend()

        fig.tight_layout()
        fig.savefig(
            os.path.join(PLOT_DIR, "total_allocnodes_bytime{}.pdf".format(args.save_suffix))
        )
        to_plot_or_not_to_plot(args.batch)

    if "queue_size_timeseries" in args.plots:
        ret = q_size(job_history)
        data_q_length, data_q_length_nodes, data_minutes = ret[0], ret[1], ret[2]
        sim_q_length, sim_q_length_nodes, sim_minutes = ret[3], ret[4], ret[5]

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        ax.plot_date(sim_minutes, sim_q_length, 'g', label="Sim", linewidth=0.5)
        ax.plot_date(data_minutes, data_q_length, 'r', label="Data", linewidth=0.5)

        ax.set_xlabel("Date (minute resolution)", fontsize=18)
        ax.set_ylabel("Queue Size (Jobs)", fontsize=18)
        plt.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "queue_size_jobs{}.pdf".format(args.save_suffix)))
        to_plot_or_not_to_plot(args.batch)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        ax.plot_date(sim_minutes, sim_q_length_nodes, 'g', label="Sim", linewidth=0.5)
        ax.plot_date(data_minutes, data_q_length_nodes, 'r', label="Data", linewidth=0.5)

        ax.set_xlabel("Date (minute resolution)", fontsize=18)
        ax.set_ylabel("Queue Size (Nodes)", fontsize=18)
        plt.legend()

        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "queue_size_nodes{}.pdf".format(args.save_suffix)))
        to_plot_or_not_to_plot(args.batch)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "sim", type=lambda sims: [ sim for sim in sims.split(',') ],
        help="Experiment to plot, can be comma delimited list to plot multiple experiments"
    )

    parser.add_argument(
        "--labels", default=["sim"], type=lambda plots: [ plot for plot in plots.split(',') ],
        help="Labels to use when plotting multiple experiments"
    )
    parser.add_argument(
        "--plots", default=[], type=lambda plots: [ plot for plot in plots.split(',') ],
        help=(
            "comma delimited list or plots to plot\n"
            "(bdslowdowns_hist2d|wait_times_hist2d|top_projs|top_qccounts|top_users|qos_waits|"
            "partition_waits|rolling_window|rolling_window_qos|cumulative_throughput|"
            "total_allocnodes_timeseries|queue_size_timeseries)"
        )
    )

    parser.add_argument("--batch", action="store_true", help="Dont draw plots, just save")
    parser.add_argument(
        "--save_suffix", type=str, default="", help="Optional suffix to add to name of saved plots"
    )
    parser.add_argument(
        "--no_data_comparison", action="store_true", help="Dont plot the data with the sim"
    )
    parser.add_argument(
        "--plot_dir", type=str, default="/work/y02/y02/awilkins/data/plots/archer2_jobdata_plots",
        help="Override ARCHER2 plot dir"
    )
    parser.add_argument(
        "--days_ignore", type=float, default=4.0,
        help="ovveride the default ignore period (4 days) at the start and end of job data"
    )
    parser.add_argument("--rolling_window_days", type=float, default=14.0)

    args = parser.parse_args()

    if len(args.sim) != len(args.labels):
        parser.error("Need a label for each experiment being plotted")

    if len(args.sim) > 1:
        print("NOTE: Not all plots have been implemented to plot multiple experiments")

    return args


if __name__ == "__main__":
    main(parse_arguments())


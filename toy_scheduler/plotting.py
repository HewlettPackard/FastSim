import os, argparse
from datetime import timedelta
import dill as pickle
from collections import defaultdict

import matplotlib.dates
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import numpy as np
from tqdm import tqdm

from controller import Controller
from fairshare import FairTree
from helpers import mkdir_p


# TODO
# - Total power usage plots


def to_plot_or_not_to_plot(batch):
    if batch:
        plt.close()
    else:
        plt.show()

def bdslowdowns_allocnodes_hist2d_true_sim(
    true_bd_slowdowns, true_allocnodes, mf_bd_slowdowns, mf_allocnodes
):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

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
                zip(mf_allocnodes, mf_bd_slowdowns)
            )
        ])
    )

    ax[0].hist2d(
        true_allocnodes, true_bd_slowdowns, bins=[bins_allocnodes, bins_bd_slowdowns], cmap='jet',
        norm=matplotlib.colors.LogNorm(vmin=1, vmax=vmax)
    )
    ax[0].set_title("ARCHER2 Data", fontsize=22)
    ax[0].set_ylabel("Job Bounded Slowdown", fontsize=20)
    ax[0].set_xlabel("Job Num Nodes", fontsize=20)

    h = ax[1].hist2d(
        mf_allocnodes, mf_bd_slowdowns, bins=[bins_allocnodes, bins_bd_slowdowns], cmap='jet',
        norm=matplotlib.colors.LogNorm(vmin=1, vmax=vmax)
    )
    ax[1].set_title("Sim", fontsize=22)
    ax[1].set_xlabel("Job Num Nodes", fontsize=20)

    for a in ax:
        a.set_xscale("log")
        a.set_yscale("log")

    fig.tight_layout()

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.03, 0.7])
    fig.colorbar(h[3], cax=cbar_ax)

    return fig, ax


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
                    controller.init_time + timedelta(days=4) < job.true_submit <
                    max_submit - timedelta(days=4)
                )
        ]
        for controller in controllers
    ]

    # NOTE Keeping implementation for a single experiment so other plots don't break
    with open(args.sim[0], "rb") as f:
        controller = pickle.load(f)

    job_history = [
        job for job in controller.job_history if (
            controller.init_time + timedelta(days=4) < job.true_submit <
            max_submit - timedelta(days=4)
        )
    ]

    print(
        "Ignoring {} out of {} jobs in evaulation\n".format(
            sum(1 for job in job_history if job.ignore_in_eval), len(job_history)
        )
    )

    assoc_tree = FairTree(
        controller.config.assocs_dump, timedelta(minutes=1), timedelta(minutes=1),
        controller.init_time, set(), 0, controller.partitions
    )

    data_bd_slowdowns = [
        max(
            (
                (job.true_job_start + job.runtime - job.true_submit) /
                max(job.runtime, controller.config.bd_threshold)
            ),
            1
        )
        for job in job_history if not job.ignore_in_eval
    ]
    sim_bd_slowdowns = [
        max(
            (
                (job.start + job.runtime - job.submit) /
                max(job.runtime, controller.config.bd_threshold)
            ),
            1
        )
        for job in job_history if not job.ignore_in_eval
    ]
    data_wait_times = [
        (job.true_job_start + job.runtime - job.true_submit).total_seconds() / 60 / 60
        for job in job_history if not job.ignore_in_eval
    ]
    sim_wait_times = [
        (job.start + job.runtime - job.submit).total_seconds() / 60 / 60
        for job in job_history if not job.ignore_in_eval
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

    if "allocnodes_bdslowdowns_hist" in args.plots:
        data_allocnodes = [ job.nodes for job in job_history if not job.ignore_in_eval ]
        fig, ax = bdslowdowns_allocnodes_hist2d_true_sim(
            data_bd_slowdowns, data_allocnodes, sim_bd_slowdowns, data_allocnodes,
        )
        fig.savefig(
            os.path.join(PLOT_DIR, "allocnodes_bdslowdowns_hist2d{}.pdf".format(args.save_suffix))
        )
        to_plot_or_not_to_plot(args.batch)

    if "top_projs" in args.plots:
        proj_sim_wait, proj_data_wait = defaultdict(list), defaultdict(list)
        proj_nodehours = defaultdict(float)
        for job in job_history:
            proj = assoc_tree.assocs[job.assoc].parent.parent.name
            sim_wait = (job.start - job.submit).total_seconds() / 60 / 60
            data_wait = (job.true_job_start - job.true_submit).total_seconds() / 60 / 60

            proj_nodehours[proj] += job.nodes * job.runtime.total_seconds() / 60 / 60
            proj_sim_wait[proj].append(sim_wait)
            proj_data_wait[proj].append(data_wait)

        top_projs = [
            proj for proj, _ in (
                sorted(proj_nodehours.items(), key=lambda keyval: keyval[1], reverse=True)[:15]
            )
        ]

        proj_sim_wait_mean = { proj : np.mean(waits) for proj, waits in proj_sim_wait.items() }
        proj_data_wait_mean = { proj : np.mean(waits) for proj, waits in proj_data_wait.items() }
        proj_sim_wait_err = { proj : np.std(waits) for proj, waits in proj_sim_wait.items() }
        proj_data_wait_err = { proj : np.std(waits) for proj, waits in proj_data_wait.items() }
        sorted_sim_wait = [
            (proj, wait_mean, proj_sim_wait_err[proj], len(proj_sim_wait[proj]))
            for proj, wait_mean in sorted(
                proj_sim_wait_mean.items(), key=lambda proj_wait: proj_wait[1], reverse=True
            )
                if proj in top_projs
        ]
        sorted_data_wait = [
            (proj, wait_mean, proj_data_wait_err[proj], len(proj_data_wait[proj]))
            for proj, wait_mean in sorted(
                proj_data_wait_mean.items(), key=lambda proj_wait: proj_wait[1], reverse=True
            )
                if proj in top_projs
        ]

        print(
            "Sim top projects by mean wait times:\n" +
            "\n".join(
                "{}.\t{}\t- {} += {} ({} jobs)".format(
                    i + 1, proj_wait[0], proj_wait[1], proj_wait[2], proj_wait[3]
                ) for i, proj_wait in enumerate(sorted_sim_wait)
            ) +
            "\n"
        )
        print(
            "True top projects by mean wait times:\n" +
            "\n".join(
                "{}.\t{}\t- {} += {} ({} jobs)".format(
                    i + 1, proj_wait[0], proj_wait[1], proj_wait[2], proj_wait[3]
                ) for i, proj_wait in enumerate(sorted_data_wait)
            ) +
            "\n"
        )

        top_projs.sort(key=lambda proj: proj_data_wait_mean[proj], reverse=True)

        sim_mean_waits = [ proj_sim_wait_mean[proj] for proj in top_projs ]
        data_mean_waits = [ proj_data_wait_mean[proj] for proj in top_projs ]
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

    if "qos_waits" in args.plots:
        sim_qos_waits, data_qos_waits = defaultdict(list), defaultdict(list)
        print("\nlargescale jobs (id - nodes - submit - sim wait - true wait")
        for job in job_history:
            sim_qos_waits[job.qos.name].append((job.start - job.submit).total_seconds() / 60 / 60)
            data_qos_waits[job.qos.name].append(
                (job.true_job_start - job.true_submit).total_seconds() / 60 / 60
            )

            if job.qos.name == "largescale":
                print(job.id, job.nodes, job.true_submit, job.start - job.submit, job.true_job_start - job.true_submit, sep=" - ")
        print()

        print("Num Jobs by QOS:")
        print(
            " | ".join("{} - {}".format(qos, len(waits)) for qos, waits in sim_qos_waits.items())
        )

        sim_qos_mean_waits = { qos : np.mean(waits) for qos, waits in sim_qos_waits.items() }
        data_qos_mean_waits = { qos : np.mean(waits) for qos, waits in data_qos_waits.items() }
        sorted_qos = [
            qos for qos, _ in sorted(
                data_qos_mean_waits.items(), key=lambda qos_wait: qos_wait[1], reverse=True
            )
        ]
        sim_mean_waits = [ sim_qos_mean_waits[qos] for qos in sorted_qos ]
        data_mean_waits = [ data_qos_mean_waits[qos] for qos in sorted_qos ]
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

    if "rolling_window" in args.plots:
        hours = [
            controllers[0].init_time.replace(minute=0, second=0) + timedelta(hours=hr)
            for hr in range(
                int(
                    (max_submit - timedelta(days=14) - controllers[0].init_time).total_seconds() /
                    (60 * 60)
                )
            )
        ]

        # Rolling window mean wait time
        sims_submit_hour_waits = []
        for job_history in job_histories:
            sim_submit_hour_waits = defaultdict(list)
            for job in job_history:
                sim_submit_hour_waits[job.submit.replace(minute=0, second=0)].append(
                    (job.start - job.submit).total_seconds() / 60 / 60
                )
            sims_submit_hour_waits.append(sim_submit_hour_waits)

        sims_mean_wait_times_rolling_window, sims_mean_wait_times_rolling_window_err = [], []
        for sim_submit_hour_waits in sims_submit_hour_waits:
            sim_mean_wait_times_rolling_window = np.zeros(len(hours))
            sim_mean_wait_times_rolling_window_err = np.zeros(len(hours))
            wait_times_rolling_window, wait_times_rolling_window_hour_lens = [], []
            for hr_num in range(336): # 2 weeks
                wait_times_rolling_window += (
                    sim_submit_hour_waits[hours[0] + timedelta(hours=hr_num)]
                )
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

            sims_mean_wait_times_rolling_window.append(sim_mean_wait_times_rolling_window)
            sims_mean_wait_times_rolling_window_err.append(sim_mean_wait_times_rolling_window_err)

        if not args.no_data_comparison:
            data_submit_hour_waits = defaultdict(list)
            for job in job_history:
                data_submit_hour_waits[job.submit.replace(minute=0, second=0)].append(
                    (job.true_job_start - job.true_submit).total_seconds() / 60 / 60
                )

            data_mean_wait_times_rolling_window = np.zeros(len(hours))
            data_mean_wait_times_rolling_window_err = np.zeros(len(hours))
            wait_times_rolling_window, wait_times_rolling_window_hour_lens = [], []
            for hr_num in range(336):
                wait_times_rolling_window += (
                    data_submit_hour_waits[hours[0] + timedelta(hours=hr_num)]
                )
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
        # The plot with the error band will be horrible for multiple experiments at once
        if len(sims_mean_wait_times_rolling_window) == 1:
            sim_mean_wait_times_rolling_window_err = sims_mean_wait_times_rolling_window_err[0]
            sim_mean_wait_times_rolling_window = sims_mean_wait_times_rolling_window[0]
            ax.plot_date(
                hour_dates, sim_mean_wait_times_rolling_window, 'g', label="Sim", linewidth=0.6
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
        sims_submit_hour_bdslowdowns = []
        for job_history in job_histories:
            sim_submit_hour_bdslowdowns = defaultdict(list)
            for job in job_history:
                sim_submit_hour_bdslowdowns[job.submit.replace(minute=0, second=0)].append(
                    max(
                        (job.end - job.submit) / max(job.runtime, controller.config.bd_threshold),
                        1
                    )
                )
            sims_submit_hour_bdslowdowns.append(sim_submit_hour_bdslowdowns)

        sims_mean_bdslowdowns_rolling_window, sims_mean_bdslowdowns_rolling_window_err = [], []
        for sim_submit_hour_bdslowdowns in sims_submit_hour_bdslowdowns:
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
                bdslowdowns_rolling_window += (
                    sim_submit_hour_bdslowdowns[hour + timedelta(hours=336)]
                )
                bdslowdowns_rolling_window_hour_lens.append(
                    len(sim_submit_hour_bdslowdowns[hour + timedelta(hours=336)])
                )

            sims_mean_bdslowdowns_rolling_window.append(sim_mean_bdslowdowns_rolling_window)
            sims_mean_bdslowdowns_rolling_window_err.append(
                sim_mean_bdslowdowns_rolling_window_err
            )

        if not args.no_data_comparison:
            data_submit_hour_bdslowdowns = defaultdict(list)
            for job in job_history:
                data_submit_hour_bdslowdowns[job.submit.replace(minute=0, second=0)].append(
                    max(
                        (
                            (job.true_job_start + job.runtime - job.true_submit) /
                            max(job.runtime, controller.config.bd_threshold)
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
                data_mean_bdslowdowns_rolling_window_err[i_hour] = (
                    np.std(bdslowdowns_rolling_window)
                )
                bdslowdowns_rolling_window = (
                    bdslowdowns_rolling_window[bdslowdowns_rolling_window_hour_lens.pop(0):]
                )
                bdslowdowns_rolling_window += (
                    data_submit_hour_bdslowdowns[hour + timedelta(hours=336)]
                )
                bdslowdowns_rolling_window_hour_lens.append(
                    len(data_submit_hour_bdslowdowns[hour + timedelta(hours=336)])
                )

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        # The plot with the error band will be horrible for multiple experiments at once
        if len(sims_mean_bdslowdowns_rolling_window) == 1:
            ax.plot_date(
                hour_dates, sim_mean_bdslowdowns_rolling_window, 'g', label="Sim",
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
            fig.savefig(
                os.path.join(
                    PLOT_DIR, "bd_slowdowns_rolling_window{}.pdf".format(args.save_suffix)
                )
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

    if "cumulative_throughput" in args.plots:
        hours_t0_to_tf = [
            controller.init_time.replace(minute=0, second=0) + timedelta(hours=hr)
            for hr in range(int((max_submit - controller.init_time).total_seconds() / 60 / 60))
        ]

        sim_end_hour_cnt = defaultdict(int)
        for job in job_history:
            sim_end_hour_cnt[job.end.replace(minute=0, second=0, microsecond=0)] += 1

        sim_throughput_cumulative = [sim_end_hour_cnt[hours_t0_to_tf[0]]]
        for hour in hours_t0_to_tf[1:]:
            sim_throughput_cumulative.append(
                sim_throughput_cumulative[-1] + sim_end_hour_cnt[hour]
            )

        data_end_hour_cnt = defaultdict(int)
        for job in job_history:
            data_end_hour_cnt[
                (job.true_job_start + job.runtime).replace(minute=0, second=0, microsecond=0)
            ] += 1

        data_throughput_cumulative = [data_end_hour_cnt[hours_t0_to_tf[0]]]
        for hour in hours_t0_to_tf[1:]:
            data_throughput_cumulative.append(
                data_throughput_cumulative[-1] + data_end_hour_cnt[hour]
            )

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        hour_dates_start_hour = matplotlib.dates.date2num([ hour for hour in hours_t0_to_tf ])
        ax.plot_date(
            hour_dates_start_hour, sim_throughput_cumulative, 'g', label="Sim",
            linewidth=0.6
        )
        ax.plot_date(
            hour_dates_start_hour, data_throughput_cumulative, 'r', label="Data", linewidth=0.6
        )
        plt.legend()
        print("Need to set the zoomed in box manually if wanted")
        # axins = zoomed_inset_axes(ax, 12, loc=4)
        # axins.plot_date(
        #     hour_dates_start_hour, sim_throughput_cumulative, 'g', label="Sim",
        #     linewidth=0.6
        # )
        # axins.plot_date(
        #     hour_dates_start_hour, data_throughput_cumulative, 'r', label="Data", linewidth=0.6
        # )
        # mark_inset(ax, axins, loc1=2, loc2=1, fc="none", ec="0.5")
        plt.xticks(visible=False)
        plt.yticks(visible=False)
        ax.set_ylabel("Cumulative Throughput", fontsize=18)
        ax.set_xlabel("Date (hour resolution)", fontsize=18)
        ax.grid(axis="x")
        # axins.set_xlim(hour_dates_start_hour[-550], hour_dates_start_hour[-490])
        # axins.set_ylim(245000, 255000)
        plt.draw()
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "throughput_cumulative{}.pdf".format(args.save_suffix)))
        to_plot_or_not_to_plot(args.batch)

    if "total_allocnodes_timeseries" in args.plots:
        sim_max_end = max(job_history, key=lambda job: job.end).end
        sim_min_start = min(job_history, key=lambda job: job.start).start
        data_max_end_job = max(job_history, key=lambda job: job.true_job_start + job.runtime)
        data_max_end = data_max_end_job.true_job_start + data_max_end_job.runtime
        data_min_start = min(job_history, key=lambda job: job.true_job_start).true_job_start

        sim_allocated_nodes = np.zeros(int((sim_max_end - sim_min_start).total_seconds() / 60))
        data_allocated_nodes = np.zeros(int((data_max_end - data_min_start).total_seconds() / 60))

        for job in tqdm(job_history):
            l_mins = int((job.start - sim_min_start).total_seconds() / 60) + 1
            u_mins = int((job.end - sim_min_start).total_seconds() / 60)
            sim_allocated_nodes[l_mins:u_mins] += job.nodes

            l_mins = int((job.true_job_start - data_min_start).total_seconds() / 60) + 1
            u_mins = int((job.true_job_start + job.runtime - data_min_start).total_seconds() / 60)
            data_allocated_nodes[l_mins:u_mins] += job.nodes

        print("Sim mean(max) allocations nodes = {} +- {} ({})".format(
            np.mean(sim_allocated_nodes[2880:-2880]), np.std(sim_allocated_nodes[2880:-2880]),
            np.max(sim_allocated_nodes)
        ))
        print("Data mean(max) allocations nodes = {} +- {} ({})".format(
            np.mean(data_allocated_nodes[2880:-2880]), np.std(data_allocated_nodes[2880:-2880]),
            np.max(data_allocated_nodes)
        ))

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        data_minutes = [
            data_min_start + timedelta(minutes=min_num)
            for min_num in range(len(data_allocated_nodes))
        ]
        sim_minutes = [
            sim_min_start + timedelta(minutes=min_num)
            for min_num in range(len(sim_allocated_nodes))
        ]
        ax.plot_date(sim_minutes, sim_allocated_nodes, 'g', label="Sim", linewidth=0.75)
        ax.plot_date(data_minutes, data_allocated_nodes, 'r', label="Data", linewidth=0.75)
        ax.set_xlabel("Date (minute resolution)", fontsize=18)
        ax.set_ylabel("Number of Allocated Nodes", fontsize=18)
        ax.set_ylim(3000, 6000)
        ax.grid(axis="y")
        plt.legend()
        fig.tight_layout()
        fig.savefig(
            os.path.join(PLOT_DIR, "total_allocnodes_bytime{}.pdf".format(args.save_suffix))
        )
        to_plot_or_not_to_plot(args.batch)

    if "queue_size_timeseries" in args.plots:
        sim_min_submit = min(job_history, key=lambda job: job.submit).submit
        data_min_submit = min(job_history, key=lambda job: job.true_submit).true_submit
        sim_max_start = max(job_history, key=lambda job: job.start).start
        data_max_start = max(job_history, key=lambda job: job.true_job_start).true_job_start

        sim_queue_length = np.zeros(int((sim_max_start - sim_min_submit).total_seconds() / 60))
        data_queue_length = np.zeros(int((data_max_start - data_min_submit).total_seconds() / 60))
        sim_queue_length_nodes = np.zeros(
            int((sim_max_start - sim_min_submit).total_seconds() / 60)
        )
        data_queue_length_nodes = np.zeros(
            int((data_max_start - data_min_submit).total_seconds() / 60)
        )

        for job in tqdm(job_history):
            l_mins = int((job.submit - sim_min_submit).total_seconds() / 60) + 1
            u_mins = int((job.start - sim_min_submit).total_seconds() / 60)
            sim_queue_length[l_mins:u_mins] += 1
            sim_queue_length_nodes[l_mins:u_mins] += job.nodes

            l_mins = int((job.true_submit - data_min_submit).total_seconds() / 60) + 1
            u_mins = int((job.true_job_start - data_min_submit).total_seconds() / 60)
            data_queue_length[l_mins:u_mins] += 1
            data_queue_length_nodes[l_mins:u_mins] += job.nodes

        print("Sim mean(max) queue size (jobs) = {} +- {} ({})".format(
            np.mean(sim_queue_length[2880:-2880]), np.std(sim_queue_length[2880:-2880]),
            np.max(sim_queue_length)
        ))
        print("Data mean(max) queue size (jobs) = {} +- {} ({})".format(
            np.mean(data_queue_length[2880:-2880]), np.std(data_queue_length[2880:-2880]),
            np.max(data_queue_length)
        ))
        print("Sim mean(max) queue size (nodes) = {} +- {} ({})".format(
            np.mean(sim_queue_length_nodes[2880:-2880]),
            np.std(sim_queue_length_nodes[2880:-2880]), np.max(sim_queue_length_nodes)
        ))
        print("Data mean(max) queue size (nodes) = {} +- {} ({})".format(
            np.mean(data_queue_length_nodes[2880:-2880]),
            np.std(data_queue_length_nodes[2880:-2880]), np.max(data_queue_length_nodes)
        ))

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        data_minutes = [
            data_min_submit + timedelta(minutes=min_num)
            for min_num in range(len(data_queue_length))
        ]
        sim_minutes = [
            sim_min_submit + timedelta(minutes=min_num)
            for min_num in range(len(sim_queue_length))
        ]
        ax.plot_date(sim_minutes, sim_queue_length, 'g', label="Sim", linewidth=0.5)
        ax.plot_date(data_minutes, data_queue_length, 'r', label="Data", linewidth=0.5)
        ax.set_xlabel("Date (minute resolution)", fontsize=18)
        ax.set_ylabel("Queue Size (Jobs)", fontsize=18)
        plt.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(PLOT_DIR, "queue_size_jobs{}.pdf".format(args.save_suffix)))
        to_plot_or_not_to_plot(args.batch)

        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.plot_date(sim_minutes, sim_queue_length_nodes, 'g', label="Sim", linewidth=0.5)
        ax.plot_date(data_minutes, data_queue_length_nodes, 'r', label="Data", linewidth=0.5)
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
            "(allocnodes_bdslowdowns_hist|top_projs|qos_waits|rolling_window|"
            "cumulative_throughput|total_allocnodes_timeseries|queue_size_timeseries)"
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

    args = parser.parse_args()

    if len(args.sim) != len(args.labels):
        parser.error("Need a label for each experiment being plotted")

    if len(args.sim) > 1:
        print("NOTE: Not all plots have been implemented to plot multiple experiments")

    return args


if __name__ == "__main__":
    main(parse_arguments())


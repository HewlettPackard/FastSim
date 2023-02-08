import os, argparse
from datetime import timedelta
import dill as pickle

import matplotlib.dates
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
import numpy as np

from controller import Controller
from fairshare import FairTree


def mkdir_p(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        return dir
    return False


def main(args):
    PLOT_DIR = os.path.join(
        "/work/y02/y02/awilkins/archer2_jobdata/plots", os.path.basename(args.sim).split(".")[0]
    )
    mkdir_p(PLOT_DIR)

    # TODO Redo this baseline fifo to not count the ignore in eval jobs, this will require
    # turning off partitions, QOS, and dependencies. This will be easier once a lot of this
    # is moved in a consts global. Wait to do this so I dont have to hack away at code
    # print("Reading baseline fifo sim results")
    # with open("/work/y02/y02/awilkins/pandas_cache/toy_scheduler/fifo_baseline.pkl", "rb") as (
    #     f
    # ):
    #     data = pickle.load(f)
    # archer_fifo = data["archer"][0]

    # XXX Temporary until I get time to redo this plotting script
    with open(args.sim, "rb") as f:
        sim_controller = pickle.load(f)
    archer = { 0 : sim_controller }

    max_submit = max(archer[0].job_history, key=lambda job: job.submit).true_submit
    job_history = [
        job for job in archer[0].job_history if (
            archer[0].init_time + timedelta(days=4) < job.true_submit <
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
            (job.true_job_start + job.runtime - job.true_submit) / max(job.runtime, archer[0].config.bd_threshold),
            1
        ) for job in job_history if not job.ignore_in_eval
    ]
    sim_bd_slowdowns = [
        max(
            (job.start + job.runtime - job.submit) / max(job.runtime, archer[0].config.bd_threshold), 1
        ) for job in job_history if not job.ignore_in_eval
    ]
    # no_eval_ids = [ job.id for job in job_history if job.ignore_in_eval ]
    # fifo_bd_slowdowns = [
    #     max(
    #         (job.start + job.runtime - job.submit) / max(job.runtime, archer[0].config.bd_threshold), 1
    #     ) for job in archer_fifo.job_history if job.id not in no_eval_ids
    # ]
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
    # fifo_wait_times = [
    #     (
    #         (job.start + job.runtime - job.submit).total_seconds() / 60 / 60
    #     ) for job in archer_fifo.job_history if job.id not in no_eval_ids
    # ]
    print(
        "True starts mean bd slowdown={}+-{} (total = {})\n".format(
            np.mean(data_bd_slowdowns), np.std(data_bd_slowdowns), np.sum(data_bd_slowdowns)
        ) +
        "Scheduling sim mean bd slowdown={}+-{} (total = {})\n".format(
            np.mean(sim_bd_slowdowns), np.std(sim_bd_slowdowns), np.sum(sim_bd_slowdowns)
        ) +
        # "FIFO baseline sim mean bd slowdown={}+-{}\n".format(
        #     np.mean(fifo_bd_slowdowns), np.std(fifo_bd_slowdowns)
        # ) +
        "True starts mean wait time={}+-{} hrs (total = {} hrs)\n".format(
            np.mean(data_wait_times), np.std(data_wait_times), np.sum(data_wait_times)
        ) +
        "Scheduling sim mean wait time={}+-{}hrs (total = {} hrs)\n".format(
            np.mean(sim_wait_times), np.std(sim_wait_times), np.sum(sim_wait_times)
        )
        # "FIFO baseline sim mean wait time={}+-{}hr\n".format(
        #     np.mean(fifo_wait_times), np.std(fifo_wait_times)
        # )
    )

    # data_allocnodes = [ job.nodes for job in job_history if not job.ignore_in_eval ]
    # fifo_allocnodes = [
    #     job.nodes for job in archer_fifo.job_history if job.id not in no_eval_ids
    # ]
    # fig, ax = bdslowdowns_allocnodes_hist2d_true_fifo_mf_noclass(
    #     data_bd_slowdowns, data_allocnodes, sim_bd_slowdowns, data_allocnodes,
    #     fifo_bd_slowdowns, fifo_allocnodes
    # )
    # fig.savefig(os.path.join(
    #     PLOT_DIR,
    #     "toyscheduler_test_refactor_withfifobaseline_bdslowdowns_allocnodes{}.pdf".format(
    #         save_suffix
    #     )
    # ))
    # if batch:
    #     plt.close()
    # else:
    #     plt.show()

    start_time = archer[0].init_time + timedelta(days=20)


    assoc_tree = FairTree(
        sim_controller.config.assocs_dump, timedelta(minutes=1), timedelta(minutes=1),
        archer[0].init_time
    )

    # print("Jobs from proj-e761 or proj-e697:")
    # for job in job_history:
    #     if job.ignore_in_eval:
    #         continue

    #     # proj-e761, proj-e697
    #     proj = assoc_tree.uniq_users[job.account][job.user].parent.parent.name
    #     if proj in ["proj-e761", "proj-e697"]:
    #         print(
    #             proj, job.user, job.submit, job.true_submit, job.start, job.true_job_start,
    #             job.id, job.nodes, job.name, job.partition, job.qos.name,
    #             job.dependency.conditions if job.dependency else None, job.reason
    #         )
    # print()

    proj_sim_wait, proj_data_wait = defaultdict(list), defaultdict(list)
    proj_sim_nolowprio_wait, proj_data_nolowprio_wait = defaultdict(list), defaultdict(list)
    proj_nodehours = defaultdict(float)
    for job in job_history:
        proj = assoc_tree.uniq_users[job.account][job.user].parent.parent.name
        sim_wait = (job.start - job.submit).total_seconds() / 60 / 60
        data_wait = (job.true_job_start - job.true_submit).total_seconds() / 60 / 60

        proj_nodehours[proj] += job.nodes * job.runtime.total_seconds() / 60 / 60
        proj_sim_wait[proj].append(sim_wait)
        proj_data_wait[proj].append(data_wait)
        if job.qos.name != "lowpriority":
            proj_sim_nolowprio_wait[proj].append(sim_wait)
            proj_data_nolowprio_wait[proj].append(data_wait)

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

    proj_sim_nolowprio_wait_mean = {
        proj : np.mean(waits) for proj, waits in proj_sim_nolowprio_wait.items()
    }
    proj_data_nolowprio_wait_mean = {
        proj : np.mean(waits) for proj, waits in proj_data_nolowprio_wait.items()
    }
    proj_sim_nolowprio_wait_err = {
        proj : np.std(waits) for proj, waits in proj_sim_nolowprio_wait.items()
    }
    proj_data_nolowprio_wait_err = {
        proj : np.std(waits) for proj, waits in proj_data_nolowprio_wait.items()
    }
    sorted_sim_nolowprio_wait = [
        (proj, wait_mean, proj_sim_nolowprio_wait_err[proj]) for proj, wait_mean in (
            sorted(
                proj_sim_nolowprio_wait_mean.items(), key=lambda proj_wait: proj_wait[1],
                reverse=True
            )
        ) if proj in top_projs
    ]
    sorted_data_nolowprio_wait = [
        (proj, wait_mean, proj_data_nolowprio_wait_err[proj]) for proj, wait_mean in (
            sorted(
                proj_data_nolowprio_wait_mean.items(), key=lambda proj_wait: proj_wait[1],
                reverse=True
            )
        ) if proj in top_projs
    ]

    print(
        "Scheduling sim top projects by mean wait times:\n" +
        "\n".join(
            "{}.\t{}\t- {} += {}".format(
                i + 1, proj_wait[0], proj_wait[1], proj_wait[2]
            ) for i, proj_wait in enumerate(sorted_sim_wait)
        ) +
        "\n"
    )
    print(
        "True starts top projects by mean wait times:\n" +
        "\n".join(
            "{}.\t{}\t- {} += {}".format(
                i + 1, proj_wait[0], proj_wait[1], proj_wait[2]
            ) for i, proj_wait in enumerate(sorted_data_wait)
        ) +
        "\n"
    )
    print(
        "Scheduling sim top projects by mean wait times excluding lowpriority jobs:\n" +
        "\n".join(
            "{}.\t{}\t- {} += {}".format(
                i + 1, proj_wait[0], proj_wait[1], proj_wait[2]
            ) for i, proj_wait in enumerate(sorted_sim_nolowprio_wait)
        ) +
        "\n"
    )
    print(
        "True starts top projects by mean wait times excluding lowpriority jobs:\n" +
        "\n".join(
            "{}.\t{}\t- {} += {}".format(
                i + 1, proj_wait[0], proj_wait[1], proj_wait[2]
            ) for i, proj_wait in enumerate(sorted_data_nolowprio_wait)
        ) +
        "\n"
    )

    top_projs.sort(key=lambda proj: proj_data_wait_mean[proj], reverse=True)

    sim_mean_waits = [ proj_sim_wait_mean[proj] for proj in top_projs ]
    data_mean_waits = [ proj_data_wait_mean[proj] for proj in top_projs ]
    x = np.arange(len(top_projs))

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    sim_bars = ax.bar(x - 2 * 0.2 / 3, sim_mean_waits, 0.2, label="Scheduling Sim")
    data_bars = ax.bar(x + 2 * 0.2 / 3, data_mean_waits, 0.2, label="Data")
    ax.set_ylabel("Mean Wait Time (hrs)", fontsize=18)
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

    sim_nolowprio_mean_waits = [ proj_sim_nolowprio_wait_mean[proj] for proj in top_projs ]
    data_nolowprio_mean_waits = [ proj_data_nolowprio_wait_mean[proj] for proj in top_projs ]
    x = np.arange(len(top_projs))

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    sim_bars = ax.bar(x - 2 * 0.2 / 3, sim_nolowprio_mean_waits, 0.2, label="Scheduling Sim")
    data_bars = ax.bar(x + 2 * 0.2 / 3, data_nolowprio_mean_waits, 0.2, label="Data")
    ax.set_ylabel("Mean Wait Time (hrs)", fontsize=18)
    ax.set_xticks(x, top_projs)
    ax.legend()
    ax.bar_label(sim_bars, padding=3, fmt="%.1f")
    ax.bar_label(data_bars, padding=3, fmt="%.1f")
    fig.tight_layout()
    fig.savefig(os.path.join(
        PLOT_DIR, "test_refactor_top_projs_nolowprio_mean_waits{}.pdf".format(save_suffix)
    ))
    if batch:
        plt.close()
    else:
        plt.show()

    sim_qos_waits, data_qos_waits = defaultdict(list), defaultdict(list)
    for job in job_history:
        sim_qos_waits[job.qos.name].append((job.start - job.submit).total_seconds() / 60 / 60)
        data_qos_waits[job.qos.name].append(
            (job.true_job_start - job.true_submit).total_seconds() / 60 / 60
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
    sim_bars = ax.bar(x - 2 * 0.2 / 3, sim_mean_waits, 0.2, label="Scheduling Sim")
    data_bars = ax.bar(x + 2 * 0.2 / 3, data_mean_waits, 0.2, label="Data")
    ax.set_ylabel("Mean Wait Time (hrs)", fontsize=18)
    ax.set_xticks(x, sorted_qos)
    ax.legend()
    ax.bar_label(sim_bars, padding=3, fmt="%.1f")
    ax.bar_label(data_bars, padding=3, fmt="%.1f")
    fig.tight_layout()
    fig.savefig(os.path.join(
        PLOT_DIR, "test_refactor_qos_mean_waits{}.pdf".format(save_suffix)
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
    ax.set_ylabel("Mean Wait Time in Window", fontsize=18)
    ax.set_xlabel("Middle Hour of 2 Week Rolling Window", fontsize=18)
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
            max((job.end - job.submit) / max(job.runtime, archer[0].config.bd_threshold), 1)
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
                    max(job.runtime, archer[0].config.bd_threshold)
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
    ax.set_ylabel("Mean Bounded Slowdown in Window", fontsize=18)
    ax.set_xlabel("Middle Hour of 2 Week Rolling Window", fontsize=18)
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

    hours_t0_to_tf = [
        archer[0].init_time.replace(minute=0, second=0) + timedelta(hours=hr) for hr in (
            range(int(
                (
                    max_submit - archer[0].init_time
                ).total_seconds() / 60 / 60
            ))
        )
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
        hour_dates_start_hour, sim_throughput_cumulative, 'g', label="Scheduling sim",
        linewidth=0.6
    )
    ax.plot_date(
        hour_dates_start_hour, data_throughput_cumulative, 'r', label="Data", linewidth=0.6
    )
    plt.legend()
    axins = zoomed_inset_axes(ax, 12, loc=4)
    axins.plot_date(
        hour_dates_start_hour, sim_throughput_cumulative, 'g', label="Scheduling sim",
        linewidth=0.6
    )
    axins.plot_date(
        hour_dates_start_hour, data_throughput_cumulative, 'r', label="Data", linewidth=0.6
    )
    mark_inset(ax, axins, loc1=2, loc2=1, fc="none", ec="0.5")
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    ax.set_ylabel("Cumulative Throughput", fontsize=18)
    ax.set_xlabel("Date (hour resolution)", fontsize=18)
    ax.grid(axis="x")
    axins.set_xlim(hour_dates_start_hour[-550], hour_dates_start_hour[-490])
    axins.set_ylim(245000, 255000)
    plt.draw()
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
    ax.set_xlabel("Date (minute resolution)", fontsize=18)
    ax.set_ylabel("Number of Allocated Nodes", fontsize=18)
    ax.set_ylim(3000, 6000)
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

    print("Scheduling sim mean(max) queue size (jobs) = {} +- {} ({})".format(
        np.mean(sim_queue_length[2880:-2880]), np.std(sim_queue_length[2880:-2880]),
        np.max(sim_queue_length)
    ))
    print("Data mean(max) queue size (jobs) = {} +- {} ({})".format(
        np.mean(data_queue_length[2880:-2880]), np.std(data_queue_length[2880:-2880]),
        np.max(data_queue_length)
    ))
    print("Scheduling sim mean(max) queue size (nodes) = {} +- {} ({})".format(
        np.mean(sim_queue_length_nodes[2880:-2880]),
        np.std(sim_queue_length_nodes[2880:-2880]), np.max(sim_queue_length_nodes)
    ))
    print("Data mean(max) queue size (nodes) = {} +- {} ({})".format(
        np.mean(data_queue_length_nodes[2880:-2880]),
        np.std(data_queue_length_nodes[2880:-2880]), np.max(data_queue_length_nodes)
    ))

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    data_minutes = [
        data_min_submit + timedelta(minutes=min_num) for min_num in (
            range(len(data_queue_length))
        )
    ]
    sim_minutes = [
        sim_min_submit + timedelta(minutes=min_num) for min_num in (
            range(len(sim_queue_length))
        )
    ]
    ax.plot_date(sim_minutes, sim_queue_length, 'g', label="Scheduling sim", linewidth=0.5)
    ax.plot_date(data_minutes, data_queue_length, 'r', label="Data", linewidth=0.5)
    ax.set_xlabel("Date (minute resolution)", fontsize=18)
    ax.set_ylabel("Queue Size (Jobs)", fontsize=18)
    plt.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(
        PLOT_DIR, "test_refactor_queue_size_jobs{}.pdf".format(save_suffix)
    ))
    if batch:
        plt.close()
    else:
        plt.show()

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.plot_date(
        sim_minutes, sim_queue_length_nodes, 'g', label="Scheduling sim", linewidth=0.5
    )
    ax.plot_date(data_minutes, data_queue_length_nodes, 'r', label="Data", linewidth=0.5)
    ax.set_xlabel("Date (minute resolution)", fontsize=18)
    ax.set_ylabel("Queue Size (Nodes)", fontsize=18)
    plt.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(
        PLOT_DIR, "test_refactor_queue_size_nodes{}.pdf".format(save_suffix)
    ))
    if batch:
        plt.close()
    else:
        plt.show()


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


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("sim", type=str)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    main(parse_arguments())


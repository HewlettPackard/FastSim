import argparse
from datetime import timedelta
import dill as pickle

import numpy as np

from controller import Controller


def main(args):
    controller = Controller(args.config_file)

    controller.run_sim(max_steps=args.max_steps)

    print_sim_result(controller)

    if args.dump_sim_to:
        with open(args.dump_sim_to, "wb") as f:
            pickle.dump(controller, f)


def print_sim_result(controller):
    max_submit = max(controller.job_history, key=lambda job: job.true_submit).true_submit
    job_history = [
        job for job in controller.job_history if (
            controller.init_time + timedelta(days=4) < job.true_submit <
            max_submit - timedelta(days=4)
        )
    ]
    data_bd_slowdowns = [
        max(
            (
                (job.true_job_start + job.runtime - job.true_submit) /
                max(job.runtime, controller.config.bd_threshold)
            ),
            1
        ) for job in job_history if not job.ignore_in_eval
    ]
    sim_bd_slowdowns = [
        max(
            (
                (job.start + job.runtime - job.submit) /
                max(job.runtime, controller.config.bd_threshold)
            ),
            1
        ) for job in job_history if not job.ignore_in_eval
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
    print(
        "True starts mean bd slowdown={}+-{} (total = {})\n".format(
            np.mean(data_bd_slowdowns), np.std(data_bd_slowdowns), np.sum(data_bd_slowdowns)
        ) +
        "Scheduling sim mean bd slowdown={}+-{} (total = {})\n".format(
            np.mean(sim_bd_slowdowns), np.std(sim_bd_slowdowns), np.sum(sim_bd_slowdowns)
        ) +
        "True starts mean wait time={}+-{} hrs (total = {} hrs)\n".format(
            np.mean(data_wait_times), np.std(data_wait_times), np.sum(data_wait_times)
        ) +
        "Scheduling sim mean wait time={}+-{}hrs (total = {} hrs)\n".format(
            np.mean(sim_wait_times), np.std(sim_wait_times), np.sum(sim_wait_times)
        )
    )


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("config_file", type=str)

    parser.add_argument("--dump_sim_to", type=str, default="", help="Pickle Controller after sim")

    parser.add_argument("--max_steps", type=int, default=0, help="Terminate sim after max steps")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    main(parse_arguments())


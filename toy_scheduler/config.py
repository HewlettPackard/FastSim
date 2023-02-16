import os
from datetime import timedelta
from collections import namedtuple

import yaml


defaults = {
    "bd_threshold" : 60, "defer" : False, "default_queue_depth" : 100, "sched_interval" : 60,
    "sched_min_interval" : 2000000, "PriorityCalcPeriod" : 5, "bf_resolution" : 60,
    "bf_max_job_test" : 500, "bf_window" : 1440, "bf_interval" : 30, "bf_max_time" : 30,
    "bf_yield_interval" : 2000000, "bf_yield_sleep" : 500000, "bf_continue" : False,
    "slowdown_with_queuesize" : False, "sched_interval_perpendingjob" : 0.028,
    "bf_time_perpriorityjob" : 0.1125, "hpe_restrictlongjobs_sliding_reservations" : "",
    "bf_max_resvs" : 500
}

vals_us = ["sched_min_interval", "bf_yield_interval", "bf_yield_sleep"]
vals_s=[
    "sched_interval", "bf_resolution", "bf_interval", "bf_max_time",
    "sched_interval_perpendingjob", "bf_time_perpriorityjob"
]
vals_min=["bd_threshold", "PriorityCalcPeriod", "bf_window"]

# TODO Include node/partition information dump once setup to read this
mandatory_fields = set(
    ["plot_dir", "assocs_dump", "node_events_dump", "reservations_dump", "job_dump"]
)


def get_config(config_file):
    print("Reading config from {}".format(config_file))

    with open(config_file) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    missing_fields = mandatory_fields - set(config_dict.keys())
    if missing_fields:
        raise ValueError(
            "Missing mandatory fields {} in config file at {}".format(missing_fields, config_file)
        )

    for option in set(defaults.keys()) - set(config_dict.keys()):
        config_dict[option] = defaults[option]

    for option in vals_us:
        config_dict[option] = timedelta(microseconds=config_dict[option])
    for option in vals_s:
        config_dict[option] = timedelta(seconds=config_dict[option])
    for option in vals_min:
        config_dict[option] = timedelta(minutes=config_dict[option])

    config_namedtuple = namedtuple("config", config_dict)
    config = config_namedtuple(**config_dict)

    return config


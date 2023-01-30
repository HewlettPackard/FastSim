import os
from datetime import timedelta
from collections import namedtuple

import yaml


defaults = {
    "bd_threshold" : timedelta(hours=1), "defer" : False, "sched_interval" : timedelta(seconds=60),
    "sched_min_interval" : timedelta(microseconds=2), "PriorityCalcPeriod" : timedelta(minutes=5),
    "bf_resolution" : timedelta(seconds=60), "bf_max_job_test" : 500,
    "bf_window" : timedelta(minutes=1440), "bf_interval" : timedelta(seconds=30),
    "bf_max_time" : timedelta(seconds=30), "bf_yield_interval" : timedelta(seconds=2),
    "bf_yield_sleep" : timedelta(seconds=0.5), "bf_continue" : False,
    "slowdown_with_queuesize" : False, "sched_interval_perpendingjob" : timedelta(seconds=0.028),
    "bf_time_perpriorityjob" : timedelta(seconds=0.1125)
}

# TODO Include node/partition information dump once setup to read this
mandatory_fields = set(
    ["plot_dir", "assocs_dump", "node_events_dump", "reservations_dump", "job_dump" ]
)


def get_config(self, config_file):
    print("Reading config from {}".format(config_file))

    with open(config_file) as f:
        config_dict = yaml.load(f, Loader=yaml.FullLoader)

    missing_fields = mandatory_fields - set(config_dict.keys()):
    if missing_fields:
        raise ValueError(
            "Missing mandatory fields {} in config file at {}".format(missing_fields, config_file)
        )

    for option in set(defaults.keys()) - set(config_dict.keys()):
        config_dict[option] = defaults[option]

    config_namedtuple = namedtuple("config", config_dict)
    config = config_namedtuple(**config_dict)

    return config


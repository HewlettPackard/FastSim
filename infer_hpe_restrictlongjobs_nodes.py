import argparse, sys, os
from collections import defaultdict
from datetime import timedelta
import dill as pickle

import pandas as pd
from tqdm import tqdm

sys.path.append("./toy_scheduler")
from helpers import convert_nodelist_to_node_nums, timelimit_str_to_timedelta


def main(args):
    df_jobs = pd.read_csv(
        args.sacct_dump, delimiter='|', lineterminator='\n', header=0,
        usecols=["Start", "End", "NodeList", "QOS", "SubmitLine", "Partition", "Timelimit"]
    )

    df_jobs = df_jobs.loc[
        (df_jobs.Start != "Unknown") & (df_jobs.Start.notna()) & (df_jobs.End != "Unknown") &
        (df_jobs.End.notna()) & (df_jobs.NodeList.notna()) & (df_jobs.NodeList != "") &
        (df_jobs.NodeList != "None assigned") & (df_jobs.Partition != "serial")
    ]
    df_jobs.Start = pd.to_datetime(df_jobs.Start, format="%Y-%m-%dT%H:%M:%S")
    df_jobs.End = pd.to_datetime(df_jobs.End, format="%Y-%m-%dT%H:%M:%S")
    df_jobs.Timelimit = df_jobs.Timelimit.apply(lambda row: timelimit_str_to_timedelta(row))

    df_jobs_short = df_jobs.loc[(df_jobs.QOS == "short")]
    exempt_nids = {
        nid for _, job_row in df_jobs_short.iterrows() for nid in (
            convert_nodelist_to_node_nums(job_row.NodeList)
        )
    }
    print("{} nids exempt due to being part of short reservation".format(len(exempt_nids)))

    start_endlim_nodes = [
        (
            job_row.Start, job_row.Start + job_row.Timelimit,
            convert_nodelist_to_node_nums(job_row.NodeList)
        ) for _, job_row in df_jobs.iterrows()
    ]

    nid_schedule = defaultdict(list)
    for entry in tqdm(start_endlim_nodes):
        for nid in entry[2]:
            if nid in exempt_nids:
                continue
            nid_schedule[nid].append(entry[:2])

    nid_in_res = defaultdict(list)
    res_start, res_end = None, None
    for nid, schedule in tqdm(nid_schedule.items()):
        schedule.sort(key=lambda start_endlim: start_endlim[0])
        cnt = 0
        for start_endlim in schedule:
            if start_endlim[1] - start_endlim[0] > timedelta(hours=1, minutes=5):
                if res_end:
                    if res_end - res_start >= timedelta(hours=6):
                        nid_in_res[nid].append((res_start, res_end))
                    res_start, res_end = None, None
                cnt = 0
                continue

            if (
                start_endlim[1].replace(minute=0, second=0) + timedelta(minutes=55) <
                start_endlim[1]
            ):
                earliest_start = (
                    start_endlim[1].replace(minute=0, second=0) + timedelta(minutes=55)
                )
            else:
                earliest_start = start_endlim[1].replace(minute=0, second=0) - timedelta(minutes=5)
            if start_endlim[0] < earliest_start:
                if res_end:
                    if res_end - res_start >= timedelta(hours=6):
                        nid_in_res[nid].append((res_start, res_end))
                    res_start, res_end = None, None
                cnt = 0
                continue

            cnt += 1
            if cnt == 1:
                res_start = earliest_start
            if cnt >= args.n:
                res_end = earliest_start + timedelta(hours=1)

        if cnt >= args.n:
            if res_end:
                if res_end - res_start >= timedelta(hours=6):
                    nid_in_res[nid].append((res_start, res_end))

    sliding_res = defaultdict(list)
    for nid, schedules in nid_in_res.items():
        for schedule in schedules:
            submitted = schedule[0]
            while submitted < schedule[1]:
                sliding_res[submitted].append(nid)
                submitted += timedelta(hours=1)

    for submitted in sorted(sliding_res):
        print("{}: {}".format(submitted, len(sliding_res[submitted])))

    with open(
        os.path.join(
            args.out_dir,
            (
                os.path.basename(args.sacct_dump).split(".")[0] +
                "_hperestrictlongjobs_slidingres_n{}.pkl".format(args.n)
            )
        ),
        "wb"
    ) as file:
        pickle.dump(sliding_res, file)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("sacct_dump", type=str)
    parser.add_argument("out_dir", type=str)

    parser.add_argument(
        "-n", type=int, default=2, help="Number of occurences to assume part of reservation"
    )

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main(parse_arguments())


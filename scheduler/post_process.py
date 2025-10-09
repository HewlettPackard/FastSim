# MIT License
#
# Copyright (c) 2023-2025 Hewlett Packard Enterprise Development LP 
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import re
import os
import sys
import ast
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd

# module_path = os.path.abspath(os.path.join('..', 'modules'))
# if module_path not in sys.path:
#    sys.path.append(module_path)
    
# import controller  # noqa: F401  (kept for drop-in parity)

# ----------------------------
# Helpers
# ----------------------------
def explode_shrunk_jobs(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every job whose `node_timeline` has >1 entries, create extra rows
    (one per shrink segment) and restore the original job to its initial node count.
    """
    extra_rows = []
    restore_map = {}  # idx to (original_jid, original_node_count, original_timeline)

    for idx, row in df.iterrows():
        timeline = row.get("node_timeline", None)
        if not timeline or len(timeline) == 1:
            continue

        # Parse timeline as list of (datetime, int)
        timeline = [(pd.to_datetime(t), int(n)) for (t, n) in timeline]
        timeline = sorted(timeline, key=lambda t: t[0])

        # Record original job info to restore later
        restore_map[idx] = (f"orig_{row['jid']}", timeline[0][1], timeline)

        for seg_idx, (seg_start, seg_nodes) in enumerate(timeline):
            seg_end = (
                timeline[seg_idx + 1][0]
                if seg_idx + 1 < len(timeline)
                else pd.to_datetime(row["end"])
            )

            seg = row.copy(deep=True)
            seg["jid"] = f"resub_shrunk_{row['jid']}_{seg_idx}"
            seg["nodes"] = seg_nodes
            seg["true_job_start"] = seg["start"] = seg_start
            seg["end"] = seg_end
            seg["runtime"] = seg["reqtime"] = seg_end - seg_start
            seg["node_timeline"] = [(seg_start, seg_nodes)]
            extra_rows.append(seg)

    # Apply renaming and restoration to the original jobs
    for idx, (new_jid, original_nodes, original_timeline) in restore_map.items():
        df.at[idx, "jid"] = new_jid
        df.at[idx, "nodes"] = original_nodes
        df.at[idx, "node_timeline"] = original_timeline

    if not extra_rows:
        return df

    return (
        pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)
        .sort_values(
            ["submit", "jid"],
            key=lambda col: pd.to_datetime(col, errors="coerce")
            if col.name == "submit"
            else col,
        )
        .reset_index(drop=True)
    )


def _ensure_dt(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns and not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], errors="coerce")


def _ensure_td(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns and not pd.api.types.is_timedelta64_dtype(df[c]):
            df[c] = pd.to_timedelta(df[c], errors="coerce")


_BASE_ID = re.compile(r"(\d+(?:_\d+)?)")  # "12345" or "12345_7"


def root_jid(j):
    """Strip any number of orig_/resub_/resub_shrunk_ prefixes; return base job id (incl. array suffix)."""
    s = str(j)
    while True:
        if s.startswith("orig_"):
            s = s[len("orig_") :]
            continue
        if s.startswith("resub_shrunk_"):
            s = s[len("resub_shrunk_") :]
            continue
        if s.startswith("resub_"):
            s = s[len("resub_") :]
            continue
        break
    m = _BASE_ID.search(s)
    return m.group(1) if m else s


# ----------------------------
# Builders
# ----------------------------
def build_true_jobs_df(jobs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only *original* jobs:
      - include base rows and 'orig_*' rows,
      - exclude any 'resub_*' and 'resub_shrunk_*' rows.
    """
    df = jobs_df.copy()

    _ensure_dt(df, ["true_submit", "true_job_start"])
    _ensure_td(df, ["runtime", "reqtime"])

    jid_str = df["jid"].astype(str)
    mask_original = ~jid_str.str.startswith("resub_") & ~jid_str.str.startswith(
        "resub_shrunk_"
    )
    df = df[mask_original].copy()

    wanted = [
        "jid",
        "user",
        "account",
        "reqtime",
        "nodes",
        "true_submit",
        "true_job_start",
        "runtime",
        "partition",
        "qos",
        "reservation",
        "node_power",
        "reason",
    ]
    keep = [c for c in wanted if c in df.columns]
    out = df[keep].copy().rename(
        columns={"true_submit": "submit", "true_job_start": "start"}
    )

    out["end"] = out["start"] + out["runtime"]
    out["wait_time"] = (out["start"] - out["submit"]).dt.total_seconds()
    out = out.sort_values("submit").set_index("submit")
    out["submit"] = out.index
    return out


def build_sim_jobs_df(jobs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Treat each simulated piece independently:
      - include base, resub_*, resub_shrunk_* rows,
      - exclude only 'orig_*' rows (those are synthetic originals).
    """
    df = jobs_df.copy()

    _ensure_dt(df, ["submit", "start", "end"])
    _ensure_td(df, ["runtime", "reqtime"])

    mask_sim_piece = ~df["jid"].astype(str).str.startswith("orig_")
    df = df[mask_sim_piece].copy()

    wanted = [
        "jid",
        "user",
        "account",
        "reqtime",
        "nodes",
        "submit",
        "start",
        "end",
        "runtime",
        "partition",
        "qos",
        "reservation",
        "node_power",
        "reason",
    ]
    keep = [c for c in wanted if c in df.columns]
    out = df[keep].copy()

    out["runtime_orig"] = out.get("runtime", pd.NaT)
    out["runtime"] = (out["end"] - out["start"])
    out["wait_time"] = (out["start"] - out["submit"]).dt.total_seconds()

    out = out.sort_values("submit").set_index("submit")
    out["submit"] = out.index
    return out


def build_compare_df_first_attempt(jobs_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each original job (root_jid), pick the FIRST simulated attempt (earliest sim start),
    and compare it to the original job's true timings. Ignore later resubmissions.
    """
    df = jobs_df.copy()

    _ensure_dt(df, ["true_submit", "true_job_start", "submit", "start", "end"])
    _ensure_td(df, ["runtime", "reqtime"])

    df["root_jid"] = df["jid"].apply(root_jid)

    jid_str = df["jid"].astype(str)
    mask_original = ~jid_str.str.startswith("resub_") & ~jid_str.str.startswith(
        "resub_shrunk_"
    )
    df_orig = (
        df[mask_original]
        .sort_values(["true_submit", "true_job_start"], na_position="last")
        .drop_duplicates("root_jid", keep="first")
    )

    mask_sim_piece = ~jid_str.str.startswith("orig_")
    df_sim = df[mask_sim_piece].copy()
    df_sim = df_sim.sort_values(["start", "submit"], na_position="last")
    first_sim = df_sim.drop_duplicates("root_jid", keep="first")

    comp = first_sim.merge(
        df_orig[
            [
                "root_jid",
                "user",
                "account",
                "partition",
                "qos",
                "reservation",
                "reqtime",
                "nodes",
                "true_submit",
                "true_job_start",
                "runtime",
            ]
        ],
        on="root_jid",
        suffixes=("", "_true"),
        how="inner",
    )

    comp["true_job_end"] = comp["true_job_start"] + comp["runtime"]
    comp["true_wait_time"] = (
        comp["true_job_start"] - comp["true_submit"]
    ).dt.total_seconds()
    comp["sim_wait_time"] = (comp["start"] - comp["submit"]).dt.total_seconds()

    cols = [
        "jid",
        "user",
        "account",
        "reqtime",
        "nodes",
        "true_submit",
        "true_job_start",
        "submit",
        "start",
        "end",
        "runtime",
        "partition",
        "reservation",
        # "dependency",
        # "is_dependency_target",
        "qos",
        "submit_priority",
        "true_job_end",
        "true_wait_time",
        "sim_wait_time",
    ]
    comp = comp[[c for c in cols if c in comp.columns]]
    return comp.sort_values(["true_submit", "submit"])


# ----------------------------
# Orchestrator
# ----------------------------
def run(sim_start: pd.Timestamp, sim_end: pd.Timestamp, results_fp: Path) -> Path:
    # --- load
    jobs_df = pd.read_pickle(results_fp)

    # Parse node_timeline if serialized
    if "node_timeline" in jobs_df.columns:
        jobs_df["node_timeline"] = jobs_df["node_timeline"].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )

    # Explode shrunk jobs (if timeline is present)
    if "node_timeline" in jobs_df.columns and "end" in jobs_df.columns:
        jobs_df = explode_shrunk_jobs(jobs_df)

    # Ensure dtypes used below
    _ensure_dt(jobs_df, ["true_submit", "true_job_start", "submit", "start", "end"])
    _ensure_td(jobs_df, ["runtime", "reqtime"])

    # --- fix initial state injections
    m1 = jobs_df["true_submit"] == (sim_start - pd.Timedelta(seconds=1))  # running at sim start
    m2 = jobs_df["true_submit"] == (sim_start + pd.Timedelta(seconds=5))  # queued at sim start

    # Set true_submit & submit from submit_priority for both cases
    if "submit_priority" in jobs_df.columns:
        jobs_df.loc[m1 | m2, "true_submit"] = jobs_df.loc[m1 | m2, "submit_priority"].values
        jobs_df.loc[m1 | m2, "submit"] = jobs_df.loc[m1 | m2, "submit_priority"].values

    # For the "running at sim_start" case, also shift start and extend runtime
    if "true_job_start" in jobs_df.columns:
        jobs_df.loc[m1, "start"] = jobs_df.loc[m1, "true_job_start"].values
        jobs_df.loc[m1, "runtime"] = jobs_df.loc[m1, "runtime"] + (
            sim_start - jobs_df.loc[m1, "true_job_start"]
        )

    # --- Build outputs
    true_jobs_df = build_true_jobs_df(jobs_df)
    sim_jobs_df = build_sim_jobs_df(jobs_df)
    compare_df = build_compare_df_first_attempt(jobs_df)

    # --- Save to timestamped results dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("../processed_results") / f"{results_fp.stem}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs_df.to_pickle(out_dir / "jobs_df_processed.pkl")
    true_jobs_df.to_pickle(out_dir / "true_jobs_df.pkl")
    sim_jobs_df.to_pickle(out_dir / "sim_jobs_df.pkl")
    compare_df.to_pickle(out_dir / "compare_df.pkl")

    print(f"Saved dataframes to: {out_dir.resolve()}")
    return out_dir


# ----------------------------
# CLI
# ----------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Post-process simulation results")
    p.add_argument("--sim-start", required=True, help="ISO datetime, e.g. 2024-09-01T00:00:00")
    p.add_argument("--sim-end", required=True, help="ISO datetime, e.g. 2024-09-15T00:00:00")
    p.add_argument("--results-fp", required=True, help="Path to results .pkl (input)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    sim_start = pd.to_datetime(args.sim_start)
    sim_end = pd.to_datetime(args.sim_end)
    results_fp = Path(args.results_fp)

    if not results_fp.exists():
        raise FileNotFoundError(f"Input results file not found: {results_fp}")

    run(sim_start=sim_start, sim_end=sim_end, results_fp=results_fp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

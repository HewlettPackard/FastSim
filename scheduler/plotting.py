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

"""
Plot driver script

It loads pickled DataFrames produced by the post-processing step:
  true_jobs_df.pkl, sim_jobs_df.pkl, compare_df.pkl
and generates/saves all figures to ../figures/<timestamp>/.
"""

import re
import argparse
from pathlib import Path
from datetime import datetime
from math import floor, log10
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns


def compute_wait_event_df(df, by=None):
    """
    Compute average wait over time using your original algorithm.
    If `by` is provided (e.g., 'partition' or 'qos'), returns a long DF with
    columns ['group', 'time', 'avg_wait'] containing one series per group.
    Otherwise, returns the classic two-column DF ['time', 'avg_wait'].
    """

    def _to_hours_since_epoch(series):
        # Ensure datetime64[ns] then convert to hours since epoch
        s = pd.to_datetime(series, errors='coerce')
        return s.astype('int64') // 10**9 / 3600  # ns -> s -> hours

    def _one_group(group_df):
        # Duplicate rows by 'nodes'
        new_rows = []
        for _, row in group_df.iterrows():
            num_duplicates = int(row['nodes'])
            for _ in range(num_duplicates):
                new_rows.append(row.to_dict())

        new_df = pd.DataFrame(new_rows).reset_index(drop=True)
        if new_df.empty:
            return pd.DataFrame({'time': [], 'avg_wait': []})

        # Convert datetimes to hours since epoch (keep your original math)
        submit_hr = _to_hours_since_epoch(new_df['submit'])
        start_hr  = _to_hours_since_epoch(new_df['start'])

        arrivals = pd.DataFrame({
            'time': submit_hr,
            'delta_count': 1,
            'delta_sum': submit_hr,
            'order': 1
        })

        departures = pd.DataFrame({
            'time': start_hr,
            'delta_count': -1,
            'delta_sum': -submit_hr,
            'order': 0
        })

        events = pd.concat([arrivals, departures], ignore_index=True)
        events.sort_values(by=['time', 'order'], inplace=True)
        events.reset_index(drop=True, inplace=True)

        events['cum_count'] = events['delta_count'].cumsum()
        events['cum_sum']   = events['delta_sum'].cumsum()

        times = events['time'].values
        unique_times, first_idx, counts = np.unique(times, return_index=True, return_counts=True)

        avg_wait_list = []
        prev_count = 0.0
        prev_sum = 0.0
        orders     = events['order'].values
        cum_counts = events['cum_count'].values
        cum_sums   = events['cum_sum'].values

        for t, i, cnt in zip(unique_times, first_idx, counts):
            group_orders     = orders[i:i+cnt]
            group_cum_counts = cum_counts[i:i+cnt]
            group_cum_sums   = cum_sums[i:i+cnt]

            if group_orders[0] == 0:
                num_dep    = (group_orders == 0).sum()
                curr_count = group_cum_counts[num_dep - 1]
                curr_sum   = group_cum_sums[num_dep - 1]
            else:
                curr_count = prev_count
                curr_sum   = prev_sum

            if curr_count > 0:
                avg_wait = t - (curr_sum / curr_count)
            else:
                avg_wait = np.nan
            avg_wait_list.append(avg_wait)

            prev_count = group_cum_counts[-1]
            prev_sum   = group_cum_sums[-1]

        return pd.DataFrame({'time': unique_times, 'avg_wait': avg_wait_list})

    # Overall curve (original behavior)
    if by is None:
        return _one_group(df)

    # Per-group curves
    if by not in df.columns:
        raise KeyError(f"Column '{by}' not found in dataframe.")

    pieces = []
    for g, sub in df.groupby(by, dropna=False):
        edf = _one_group(sub)
        if not edf.empty:
            edf['group'] = g
            pieces.append(edf)

    if not pieces:
        return pd.DataFrame(columns=['group', 'time', 'avg_wait'])

    out = pd.concat(pieces, ignore_index=True)
    return out.sort_values(['group', 'time'])


def _ymax_in_range(event_df, sim_start, sim_end):
    """Return max(avg_wait) within [sim_start, sim_end] or None if no points."""
    if event_df is None or event_df.empty or "time" not in event_df or "avg_wait" not in event_df:
        return None
    t = pd.to_datetime(event_df["time"] * 3600 * 1e9)  # hours → ns → datetime
    s0 = pd.to_datetime(sim_start)
    s1 = pd.to_datetime(sim_end)
    m = (t >= s0) & (t <= s1)
    if not m.any():
        return None
    vals = pd.to_numeric(event_df.loc[m, "avg_wait"], errors="coerce")
    vals = vals.replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.max()) if not vals.empty else None


def plot_wait_time_over_time(true_event_df, sim_event_df, sim_start, sim_end, title=None, savepath=None):
    # Convert X to datetime for plotting
    true_t = pd.to_datetime(true_event_df.time * 3600 * 1e9) if not true_event_df.empty else pd.Series([], dtype='datetime64[ns]')
    sim_t  = pd.to_datetime(sim_event_df.time  * 3600 * 1e9) if not sim_event_df.empty  else pd.Series([], dtype='datetime64[ns]')

    plt.figure(figsize=(12, 5), dpi=300)

    # Plot Ground Truth and Simulation (your style)
    if not true_event_df.empty:
        plt.plot(
            true_t, true_event_df.avg_wait,
            label='Ground Truth', color='#3976A3', linewidth=2.5
        )
    if not sim_event_df.empty:
        plt.plot(
            sim_t, sim_event_df.avg_wait,
            label='FastSim', color='#FFA245', linewidth=2.5, alpha=0.9
        )

    # X-axis formatting
    s0 = pd.to_datetime(sim_start)
    s1 = pd.to_datetime(sim_end)
    plt.xlim([s0, s1])
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.xticks(rotation=45)

    # Y-axis: clamp using ONLY points within [sim_start, sim_end]
    ymax_true = _ymax_in_range(true_event_df, s0, s1)
    ymax_sim  = _ymax_in_range(sim_event_df,  s0, s1)
    candidates = [v for v in (ymax_true, ymax_sim) if v is not None]
    if candidates:
        y_max = max(candidates)
        plt.ylim(bottom=0, top=(y_max * 1.05 if y_max > 0 else 1.0))
    else:
        plt.ylim(bottom=0)  # fall back to autoscale for top

    # Grid, legend, and styling
    plt.ylabel('Average Wait Time (hours)', fontsize=14)
    plt.grid(visible=True, which='major', linestyle='--', linewidth=0.5, alpha=0.6)
    plt.legend(fontsize=12, loc='upper right')
    if title:
        plt.title(title, fontsize=16)
    plt.tight_layout()

    if savepath:
        plt.savefig(savepath, bbox_inches='tight')
    #  


def plot_wait_time_by(true_df, sim_df, by, sim_start, sim_end, groups=None, save_dir=None, fig_num=None, fig_dir=None):
    """
    Plot wait-time curves for each value in `by` (e.g., 'partition' or 'qos').
    Uses the updated y-lim logic so max is computed only from data inside [sim_start, sim_end].
    Assumes `compute_wait_event_df(df, by=...)` from the previous cell is available.
    """
    true_events = compute_wait_event_df(true_df, by=by)
    sim_events  = compute_wait_event_df(sim_df,  by=by)

    # Determine groups to plot
    true_groups = set(true_events['group'].dropna().unique()) if not true_events.empty else set()
    sim_groups  = set(sim_events['group'].dropna().unique())  if not sim_events.empty  else set()
    all_groups  = sorted(true_groups.union(sim_groups), key=lambda x: str(x))

    if groups is not None:
        wanted = set(groups)
        all_groups = [g for g in all_groups if g in wanted]

    for g in all_groups:
        tdf = true_events[true_events['group'] == g][['time','avg_wait']]
        sdf = sim_events[sim_events['group'] == g][['time','avg_wait']]
        if tdf.empty and sdf.empty:
            continue

        # title = f"Average Wait Time Over Time —"
        # savepath = None
        # if save_dir:
        #     safe_g = str(g).replace(' ', '_')
        #     savepath = f"{save_dir}/wait_time_{by}_{safe_g}.png"
        _title = fig_title(f"Average Wait Time Over Time — {by} = {g}", fig_num)
        plot_wait_time_over_time(
            tdf, sdf, sim_start, sim_end,
            title=_title
        )
        save_fig(fig_dir, fig_num, title_text=_title)
        plt.close()


def _mk_bucket(wait_h: float):
    """Return (mag_exp, label) for an order-of-magnitude bucket based on wait_h (hours)."""
    if not np.isfinite(wait_h) or wait_h < 0:
        return (-999, "invalid")
    if wait_h < 1:
        return (-1, "<1 h")
    p = int(floor(log10(wait_h)))  # 1–9 -> 10^0, 10–99 -> 10^1, etc
    lo = int(10 ** p)
    hi = int(10 ** (p + 1))
    return (p, f"{lo}–{hi} h")

def _agg_mean_wait_from_compare(comp_df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    From compare_df: compute mean true and sim wait (hours) per group_col,
    and return long-form with columns: [group_col, wait_h, data]
    """
    if comp_df.empty:
        return pd.DataFrame(columns=[group_col, "wait_h", "data"])

    comp_df = comp_df.copy()
    for c in ("true_wait_time", "sim_wait_time"):
        comp_df[c] = pd.to_numeric(comp_df[c], errors="coerce")

    g = comp_df.groupby(group_col, dropna=False)
    mean_true = (g["true_wait_time"].mean() / 3600.0).rename("Ground Truth")
    mean_sim  = (g["sim_wait_time"].mean()  / 3600.0).rename("FastSim")

    wide = pd.concat([mean_true, mean_sim], axis=1)
    wide.index = wide.index.astype(str)  # stable categorical labels
    long = (
        wide.reset_index()
            .melt(id_vars=[group_col], var_name="data", value_name="wait_h")
            .dropna(subset=["wait_h"])
    )
    return long

# ---------------------------
# Main plotting function
# ---------------------------
def plot_mean_wait_by_orders_one_figure(
    compare_df: pd.DataFrame,
    group_col: str,                   # "user" | "account" | "partition"
    *,
    top_n: int | None = None,         # keep N most frequent groups (by count in compare_df)
    min_jobs: int | None = None,      # require at least this many compare_df rows per group
    title_prefix: str = "Average Wait Time (Apples-to-Apples)",
    save_path: str | Path | None = None,
    annotate: bool = False,
    per_bar_in: float = 0.40,         # width contribution (inches) per bar in a panel
    panel_height_in: float = 6.0,     # height (inches) for each subplot
    dpi: int = 300,
    palette: dict | None = None,
):
    """
    Build mean waits from compare_df per `group_col`, bucket by order-of-magnitude,
    and render ONE figure with side-by-side subplots (one per bucket). Each subplot's
    width scales with its number of bars so bar width is visually constant across panels.
    """
    if palette is None:
        palette = {"Ground Truth": "#3976A3", "FastSim": "#FFA245"}

    # Choose groups by frequency in compare_df (applies to apples-to-apples set)
    counts = compare_df[group_col].astype(str).value_counts(dropna=False)
    if min_jobs is not None:
        counts = counts[counts >= min_jobs]
    if top_n is not None:
        counts = counts.head(top_n)
    groups = set(counts.index.astype(str))
    if not groups:
        groups = set(compare_df[group_col].astype(str).unique())

    # Aggregate mean waits (hours) and keep selected groups
    data = _agg_mean_wait_from_compare(
        compare_df[compare_df[group_col].astype(str).isin(groups)].copy(),
        group_col=group_col
    )
    if data.empty:
        print(f"[plot] No data to plot for '{group_col}'.")
        return None, None

    # Assign bucket by OOM using MAX of (GT, FastSim) per group so both bars land together
    max_by_group = (data.groupby(group_col, as_index=False)["wait_h"].max()
                        .rename(columns={"wait_h": "wait_h_max"}))
    max_by_group[["mag_exp", "bucket_label"]] = max_by_group["wait_h_max"].apply(
        lambda x: pd.Series(_mk_bucket(x))
    )
    data = data.merge(max_by_group[[group_col, "mag_exp", "bucket_label"]],
                      on=group_col, how="left")

    # Bucket ordering & panel widths (proportional to #categories in that bucket)
    bucket_tbl = (data[[group_col, "mag_exp", "bucket_label"]]
                    .drop_duplicates()
                    .groupby(["mag_exp", "bucket_label"], as_index=False)
                    .size()
                    .rename(columns={"size": "n_groups"}))
    # Drop 'invalid'
    bucket_tbl = bucket_tbl[bucket_tbl["bucket_label"] != "invalid"]
    if bucket_tbl.empty:
        print("[plot] All means fell in invalid bucket.")
        return None, None

    bucket_tbl = bucket_tbl.sort_values(["mag_exp", "bucket_label"])
    width_ratios = bucket_tbl["n_groups"].tolist()
    n_panels = len(width_ratios)

    # Figure width scales with total number of bars (i.e., sum of groups across buckets)
    total_groups = sum(max(1, n) for n in width_ratios)
    # Pad a bit for margins/legends
    fig_w = max(8.0, total_groups * per_bar_in + 1.5 + 0.5 * (n_panels - 1))
    fig_h = panel_height_in

    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(fig_w, fig_h),
        dpi=dpi,
        gridspec_kw={"width_ratios": width_ratios}
    )
    if n_panels == 1:
        axes = [axes]

    # Plot each bucket in its own axis
    for ax, (_, row) in zip(axes, bucket_tbl.iterrows()):
        mag_exp = row["mag_exp"]
        label   = row["bucket_label"]

        data_b = data[data["bucket_label"] == label].copy()
        # Order categories by descending max wait within the bucket
        order_df = (data_b.groupby(group_col, as_index=False)["wait_h"].max()
                         .sort_values("wait_h", ascending=False))
        x_order = order_df[group_col].tolist()

        sns.barplot(
            data=data_b,
            x=group_col, y="wait_h", hue="data",
            order=x_order, hue_order=["Ground Truth", "FastSim"],
            palette=palette, ax=ax, width=0.8
        )
        ax.set_xlabel("")
        ax.set_ylabel("Mean Wait (hours)")
        ax.set_title(label)
        ax.set_ylim(bottom=0)
        # Rotate labels a bit for long names
        ax.tick_params(axis='x', labelrotation=(25 if group_col == "user" else 45))

        if annotate:
            for cont in ax.containers:
                ax.bar_label(cont, fmt="%.1f", rotation=60, padding=2)

        # Put legend only on the first panel to save space
        if ax is axes[0]:
            ax.legend(title="", loc="best")
        else:
            ax.legend_.remove()

    # Common title
    suptitle = f"{title_prefix} — by {group_col}"
    fig.suptitle(suptitle, y=1.02, fontsize=14)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

     
    return fig, axes


def _queue_series(
    df: pd.DataFrame,
    *,
    kind: str,                       # 'node_hours' | 'nodes' | 'jobs'
    exclude_shrunk: bool = False,
    runtime_col: str = "runtime",    # used only when kind == 'node_hours'
    submit_col: str = "submit",
    start_col: str = "start"
) -> pd.DataFrame:
    """
    Build a stepwise time series of the queued quantity via +arrival/-departure events.

    Returns a DataFrame indexed by 'time' with a single column 'queued' (cumsum of deltas).
    """
    if df.empty:
        return pd.DataFrame(columns=["queued"]).set_index(pd.to_datetime([]))

    work = df.copy()
    if exclude_shrunk and "jid" in work.columns:
        work = work[~work["jid"].astype(str).str.contains("shrunk", na=False)]

    # Ensure dtypes
    for c in (submit_col, start_col):
        if c in work.columns and not pd.api.types.is_datetime64_any_dtype(work[c]):
            work[c] = pd.to_datetime(work[c], errors="coerce")
    if kind == "node_hours":
        if runtime_col in work.columns and not pd.api.types.is_timedelta64_dtype(work[runtime_col]):
            work[runtime_col] = pd.to_timedelta(work[runtime_col], errors="coerce")

    # Weight per job
    if kind == "node_hours":
        hours = work[runtime_col].dt.total_seconds() / 3600.0
        weight = work["nodes"] * hours
    elif kind == "nodes":
        weight = work["nodes"]
    elif kind == "jobs":
        weight = pd.Series(1, index=work.index, dtype=float)
    else:
        raise ValueError("kind must be one of {'node_hours','nodes','jobs'}")

    # Build + / - event table
    arrivals = pd.DataFrame({"time": work[submit_col], "delta": weight})
    departures = pd.DataFrame({"time": work[start_col],  "delta": -weight})

    events = pd.concat([arrivals, departures], ignore_index=True)
    events = events.dropna(subset=["time", "delta"]).sort_values("time")
    events["queued"] = events["delta"].cumsum()

    out = events[["time", "queued"]].copy()
    out = out.drop_duplicates(subset="time", keep="last")  # one row per time
    out = out.set_index("time").sort_index()
    return out

# -----------------------------
# Plotter
# -----------------------------
def plot_queue_series(
    true_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    *,
    kind: str,                         # 'node_hours' | 'nodes' | 'jobs'
    sim_start: pd.Timestamp,
    sim_end: pd.Timestamp,
    title: str | None = None,
    ylabel: str | None = None,
    use_runtime_orig_for_sim: bool = True,
    exclude_shrunk_for_sim: bool = True,
    color_true: str = "#3976A3",
    color_sim: str = "#FFA245",
    alpha_sim: float = 0.85,
    linewidth: float = 2.0,
    figsize=(12, 6),
    dpi: int = 300,
):
    """
    Plot Ground Truth vs Sim stepwise series for the requested 'kind'.
    """
    # Series
    true_series = _queue_series(
        true_df, kind=kind,
        exclude_shrunk=False,
        runtime_col="runtime"  # True uses the actual runtime
    )

    sim_runtime_col = "runtime_orig" if (kind == "node_hours" and use_runtime_orig_for_sim) else "runtime"
    sim_series = _queue_series(
        sim_df, kind=kind,
        exclude_shrunk=exclude_shrunk_for_sim,
        runtime_col=sim_runtime_col
    )

    # Labels
    if title is None:
        title = {
            "node_hours": "Node-Hours on Queue Over Time",
            "nodes":      "Nodes on Queue Over Time",
            "jobs":       "Jobs on Queue Over Time",
        }[kind]
    if ylabel is None:
        ylabel = {
            "node_hours": "# of Node-Hours on Queue",
            "nodes":      "# of Nodes on Queue",
            "jobs":       "# of Jobs on Queue",
        }[kind]

    # Plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.plot(true_series.index, true_series["queued"], label="Ground Truth",
            color=color_true, linewidth=linewidth)
    ax.plot(sim_series.index,  sim_series["queued"],  label="Sim",
            color=color_sim, linewidth=linewidth, alpha=alpha_sim)

    # Style
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Date")
    ax.set_xlim([sim_start, sim_end])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(visible=True, which="major", linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend()
    plt.tight_layout()
    #  
    return fig, ax




# ---------- colors / style ----------
GT_COLOR  = '#3976A3'
SIM_COLOR = '#FFA245'
date_format = mdates.DateFormatter('%b %d')

# ---------- figure numbering + saving ----------
def make_fig_dir(root: Path = Path("../figures")) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = root / ts
    out.mkdir(parents=True, exist_ok=True)
    return out

_fig_id = defaultdict(int)

def fig_title(text: str, fig_num: int) -> str:
    if _fig_id[fig_num] == 0:
        _fig_id[fig_num] = 'A'
    s = f"Figure {fig_num}{_fig_id[fig_num]}: {text}"
    return s

def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_").lower()

def save_fig(fig_dir: Path, fig_num: int, fig=None, title_text=None, extra_suffix=None, dpi=300):
    if fig is None:
        fig = plt.gcf()
    num = _fig_id["n"] - 1
    if title_text is None:
        ax = plt.gca()
        title_text = (fig._suptitle.get_text() if fig._suptitle is not None else ax.get_title()) or f"figure_{num:02d}"
    base = f"{_slug(title_text)}"
    if extra_suffix:
        base += f"_{_slug(extra_suffix)}"
    out_fp = fig_dir / f"{base}.png"
    fig.savefig(out_fp, dpi=dpi, bbox_inches="tight")
    _fig_id[fig_num] = chr(ord(_fig_id[fig_num]) + 1)
    # print(f"Saved: {out_fp}")

# ---------- core run ----------
def run_all_plots(true_jobs_df: pd.DataFrame,
                  sim_jobs_df: pd.DataFrame,
                  compare_df: pd.DataFrame,
                  sim_start: pd.Timestamp,
                  sim_end: pd.Timestamp,
                  fig_dir: Path):
    """Generate and save the full plot suite to fig_dir."""
    print(" 1) Average Wait Time Over Time — Overall")
    true_event_df = compute_wait_event_df(true_jobs_df)
    sim_event_df  = compute_wait_event_df(sim_jobs_df)
    _title = fig_title("Average Wait Time Over Time — Overall", 1)
    plot_wait_time_over_time(true_event_df, sim_event_df, sim_start, sim_end, title=_title)
    save_fig(fig_dir, 1, title_text=_title)
    plt.close()

    print(" 2) Average Wait Time Over Time — By Partition")
    _title = fig_title("Average Wait Time Over Time — By Partition", 2)
    _ = plot_wait_time_by(true_jobs_df, sim_jobs_df, by='partition', sim_start=sim_start, sim_end=sim_end, fig_num=2, fig_dir=fig_dir)
    # plt.gcf().suptitle(_title, y=1.02)
    # plt.tight_layout()
    # save_fig(fig_dir, 2, title_text=_title)
    # plt.close()

    print(" 3) Average Wait Time Over Time — By QOS")
    _title = fig_title("Average Wait Time Over Time — By QOS", 3)
    _ = plot_wait_time_by(true_jobs_df, sim_jobs_df, by='qos', sim_start=sim_start, sim_end=sim_end, fig_num=3, fig_dir=fig_dir)
    # plt.gcf().suptitle(_title, y=1.02)
    # plt.tight_layout()
    # save_fig(fig_dir, 3, title_text=_title)
    # plt.close()

    print(" 4) Mean Wait (Users)")
    _title = fig_title("Average Wait Time for Top 15 Users (Bucketed by Magnitude)", 4)
    fig, axes = plot_mean_wait_by_orders_one_figure(
        compare_df, group_col="user", top_n=15, min_jobs=None,
        title_prefix=_title, save_path=None, annotate=False,
        per_bar_in=0.42, panel_height_in=6, dpi=300
    )
    save_fig(fig_dir, 4, fig=fig, title_text=_title)
    plt.close()

    print(" 5) Mean Wait (Accounts)")
    _title = fig_title("Average Wait Time for Top 15 Accounts (Bucketed by Magnitude)", 5)
    fig, axes = plot_mean_wait_by_orders_one_figure(
        compare_df, group_col="account", top_n=15, min_jobs=None,
        title_prefix=_title, save_path=None, annotate=False,
        per_bar_in=0.42, panel_height_in=6, dpi=300
    )
    save_fig(fig_dir, 5, fig=fig, title_text=_title)
    plt.close()

    print(" 6) Mean Wait (Partitions)")
    _title = fig_title("Average Wait Time for Top 15 Partitions (Bucketed by Magnitude)", 6)
    fig, axes = plot_mean_wait_by_orders_one_figure(
        compare_df, group_col="partition", top_n=15, min_jobs=None,
        title_prefix=_title, save_path=None, annotate=False,
        per_bar_in=0.42, panel_height_in=6, dpi=300
    )
    save_fig(fig_dir, 6, fig=fig, title_text=_title)
    plt.close()

    print(" 7) Wait Time Distribution (global)")
    plt.figure(figsize=(12,5), dpi=300)
    (true_jobs_df['wait_time'] / 3600).hist(
        bins=np.logspace(-4, 3.5, 100), grid=False, label='Ground Truth', color=GT_COLOR
    )
    (sim_jobs_df['wait_time'] / 3600).hist(
        bins=np.logspace(-4, 3.5, 100), grid=False, alpha=0.7, label='Simulator', color=SIM_COLOR
    )
    _title = fig_title('Wait Time Distribution (All Jobs)', 7)
    plt.title(_title, fontsize=16)
    plt.xlabel('Wait Time (hours)', fontsize=14)
    plt.ylabel('Job Count', fontsize=14)
    plt.legend(fontsize=12)
    plt.xscale('log')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    save_fig(fig_dir, 7, title_text=_title)
    plt.close()
     

    print(" 8) Allocated Nodes Over Time (GT vs. Sim)")
    node_changes = pd.concat([
        pd.DataFrame({'time': true_jobs_df['start'], 'node_change':  true_jobs_df['nodes']}),
        pd.DataFrame({'time': true_jobs_df['end'],   'node_change': -true_jobs_df['nodes']})
    ]).sort_values('time')
    node_changes['allocated_nodes'] = node_changes['node_change'].cumsum()
    node_changes = node_changes.drop_duplicates(subset=['time'], keep='last').set_index('time')

    node_changes_sim = pd.concat([
        pd.DataFrame({'time': sim_jobs_df['start'], 'node_change':  sim_jobs_df['nodes']}),
        pd.DataFrame({'time': sim_jobs_df['end'],   'node_change': -sim_jobs_df['nodes']})
    ]).sort_values('time')
    node_changes_sim['allocated_nodes'] = node_changes_sim['node_change'].cumsum()
    node_changes_sim = node_changes_sim.drop_duplicates(subset=['time'], keep='last').set_index('time')

    fig, ax = plt.subplots(figsize=(12,6), dpi=300)
    _title = fig_title('Allocated Nodes Over Time', 8)
    plt.title(_title)
    plt.ylabel('# of Nodes Allocated')
    plt.xlabel('Date')
    ax.plot(node_changes.index,     node_changes.allocated_nodes,     linewidth=.7, label='Ground Truth', color=GT_COLOR)
    ax.plot(node_changes_sim.index, node_changes_sim.allocated_nodes, linewidth=.7, alpha=.9, label='Sim', color=SIM_COLOR)
    plt.xlim([sim_start, sim_end])
    ax.xaxis.set_major_formatter(date_format)
    plt.legend()
    plt.tight_layout()
    save_fig(fig_dir, 8, fig=fig, title_text=_title)
    plt.close()
     

    print(" 9) Allocated Nodes by Partition (GT & Sim)")
    for (df_src, title_side) in ((true_jobs_df, "Ground Truth"), (sim_jobs_df, "Simulator")):
        df_start = df_src[['start','nodes','partition']].copy().rename(columns={'start': 'time', 'nodes': 'change'})
        df_end   = df_src[['end','nodes','partition']].copy().rename(  columns={'end':   'time', 'nodes': 'change'})
        df_end['change'] *= -1
        events = (pd.concat([df_start, df_end], ignore_index=True)
                    .sort_values('time')
                    .groupby(['time','partition'], as_index=False)
                    .agg({'change': 'sum'}))
        events_pivot = events.pivot(index='time', columns='partition', values='change').fillna(0)
        cumulative = events_pivot.cumsum().clip(lower=0)

        fig, ax = plt.subplots(figsize=(12,6), dpi=300)
        cumulative.plot(kind='area', stacked=True, colormap='tab20', linewidth=0, ax=ax)
        _title = fig_title(f'Allocated Nodes by Partition ({title_side})', 9)
        plt.title(_title)
        plt.xlim([sim_start, sim_end])
        ax.xaxis.set_major_formatter(date_format)
        plt.legend(loc='lower right', ncol=2, fontsize=9)
        plt.tight_layout()
        save_fig(fig_dir, 9, fig=fig, title_text=_title)
        plt.close()
         

    print(" 10) Queued Node-Hours by Partition (GT & Sim)")
    # GT
    df_submit = true_jobs_df[['submit','nodes','runtime','partition']].copy().rename(columns={'submit': 'time', 'nodes': 'change'})
    df_start  = true_jobs_df[['start','nodes','runtime','partition']].copy().rename( columns={'start':  'time', 'nodes': 'change'})
    df_start['change']  *= -1
    df_submit['change'] *= df_submit['runtime'].dt.total_seconds() / 3600
    df_start['change']  *= df_start['runtime'].dt.total_seconds() / 3600
    events = (pd.concat([df_submit, df_start], ignore_index=True)
                .sort_values('time')
                .groupby(['time','partition'], as_index=False)
                .agg({'change': 'sum'}))
    events_pivot = events.pivot(index='time', columns='partition', values='change').fillna(0)
    cumulative = events_pivot.cumsum().clip(lower=0)

    fig, ax = plt.subplots(figsize=(12,6), dpi=300)
    cumulative.plot(kind='area', stacked=True, colormap='tab20', linewidth=0, ax=ax)
    _title = fig_title('Queued Node-Hours by Partition (Ground Truth)', 10)
    plt.title(_title)
    plt.xlim([sim_start, sim_end])
    ax.xaxis.set_major_formatter(date_format)
    ymin, ymax = plt.ylim()
    plt.tight_layout()
    save_fig(fig_dir, 10, fig=fig, title_text=_title)
    plt.close()
     

    # Sim (uses runtime_orig where available & excludes 'shrunk' if present)
    if 'runtime_orig' in sim_jobs_df.columns:
        sim_base = sim_jobs_df[~sim_jobs_df.get('jid','').astype(str).str.contains('shrunk')]
        df_submit = sim_base[['submit','nodes','runtime_orig','partition']].copy().rename(columns={'submit': 'time', 'nodes': 'change'})
        df_start  = sim_base[['start','nodes','runtime_orig','partition']].copy().rename( columns={'start':  'time', 'nodes': 'change'})
        df_start['change']  *= -1
        df_submit['change'] *= df_submit['runtime_orig'].dt.total_seconds() / 3600
        df_start['change']  *= df_start['runtime_orig'].dt.total_seconds() / 3600
    else:
        df_submit = sim_jobs_df[['submit','nodes','runtime','partition']].copy().rename(columns={'submit': 'time', 'nodes': 'change'})
        df_start  = sim_jobs_df[['start','nodes','runtime','partition']].copy().rename( columns={'start':  'time', 'nodes': 'change'})
        df_start['change']  *= -1
        df_submit['change'] *= df_submit['runtime'].dt.total_seconds() / 3600
        df_start['change']  *= df_start['runtime'].dt.total_seconds() / 3600

    events = (pd.concat([df_submit, df_start], ignore_index=True)
                .sort_values('time')
                .groupby(['time','partition'], as_index=False)
                .agg({'change': 'sum'}))
    events_pivot = events.pivot(index='time', columns='partition', values='change').fillna(0)
    cumulative = events_pivot.cumsum().clip(lower=0)

    fig, ax = plt.subplots(figsize=(12,6), dpi=300)
    cumulative.plot(kind='area', stacked=True, colormap='tab20', linewidth=0, ax=ax)
    _title = fig_title('Queued Node-Hours by Partition (Simulator)', 11)
    plt.title(_title)
    plt.xlim([sim_start, sim_end])
    ax.xaxis.set_major_formatter(date_format)
    plt.ylim([ymin, ymax])
    plt.tight_layout()
    save_fig(fig_dir, 10, fig=fig, title_text=_title)
    plt.close()
     

    print(" 11) Queue Series (Node-Hours / Nodes / Jobs vs Time)")
    _title = fig_title("Node-Hours on Queue Over Time", 12)
    plot_queue_series(
        true_jobs_df, sim_jobs_df,
        kind="node_hours",
        sim_start=sim_start, sim_end=sim_end,
        title=_title,
        ylabel="# of Node-Hours on Queue",
        use_runtime_orig_for_sim=True,
        exclude_shrunk_for_sim=True
    )
    save_fig(fig_dir, 12, title_text=_title)
    plt.close()

    _title = fig_title("Nodes on Queue Over Time", 13)
    plot_queue_series(
        true_jobs_df, sim_jobs_df,
        kind="nodes",
        sim_start=sim_start, sim_end=sim_end,
        title=_title,
        ylabel="# of Nodes on Queue",
        exclude_shrunk_for_sim=True
    )
    save_fig(fig_dir, 13, title_text=_title)
    plt.close()

    _title = fig_title("Jobs on Queue Over Time", 14)
    plot_queue_series(
        true_jobs_df, sim_jobs_df,
        kind="jobs",
        sim_start=sim_start, sim_end=sim_end,
        title=_title,
        ylabel="# of Jobs on Queue",
        exclude_shrunk_for_sim=True
    )
    save_fig(fig_dir, 14, title_text=_title)
    plt.close()

    print(" 12) Per-Partition Line Plots")
    for partition in sorted(true_jobs_df['partition'].dropna().unique()):
        t_part = true_jobs_df[true_jobs_df['partition'] == partition].copy()
        s_part = sim_jobs_df[sim_jobs_df['partition'] == partition].copy()
        if t_part.empty and s_part.empty:
            continue

        # Allocated Nodes
        t_alloc = (pd.concat([
            pd.DataFrame({'time': t_part['start'], 'delta':  t_part['nodes']}),
            pd.DataFrame({'time': t_part['end'],   'delta': -t_part['nodes']}),
        ], ignore_index=True).dropna(subset=['time']).sort_values('time'))
        t_alloc['allocated_nodes'] = t_alloc['delta'].cumsum().astype(float)
        t_alloc = t_alloc.set_index('time')

        s_alloc = (pd.concat([
            pd.DataFrame({'time': s_part['start'], 'delta':  s_part['nodes']}),
            pd.DataFrame({'time': s_part['end'],   'delta': -s_part['nodes']}),
        ], ignore_index=True).dropna(subset=['time']).sort_values('time'))
        s_alloc['allocated_nodes'] = s_alloc['delta'].cumsum().astype(float)
        s_alloc = s_alloc.set_index('time')

        plt.figure(figsize=(12,6), dpi=150)
        _ptitle = fig_title(f'Allocated Nodes — Partition: {partition}', 15)
        plt.title(_ptitle)
        plt.ylabel('# of Allocated Nodes'); plt.xlabel('Date')
        if not t_alloc.empty:
            plt.plot(t_alloc.index, t_alloc['allocated_nodes'], linewidth=1, label='Ground Truth', color=GT_COLOR)
        if not s_alloc.empty:
            plt.plot(s_alloc.index, s_alloc['allocated_nodes'], linewidth=1, alpha=.85, label='Sim', color=SIM_COLOR)
        plt.xlim([sim_start, sim_end]); plt.gca().xaxis.set_major_formatter(date_format)
        plt.legend(); plt.tight_layout()
        save_fig(fig_dir, 15, title_text=_ptitle, extra_suffix=partition)
        plt.close()
        

        # Node-Hours on Queue
        t_hours = (pd.concat([
            pd.DataFrame({'time': t_part['submit'], 'delta':  t_part['nodes'] * t_part['runtime'].dt.total_seconds() / 3600.0}),
            pd.DataFrame({'time': t_part['start'],  'delta': -t_part['nodes'] * t_part['runtime'].dt.total_seconds() / 3600.0}),
        ], ignore_index=True).dropna(subset=['time']).sort_values('time'))
        t_hours['node_hours'] = t_hours['delta'].cumsum()
        t_hours = t_hours.set_index('time')

        s_hours = (pd.concat([
            pd.DataFrame({'time': s_part['submit'], 'delta':  s_part['nodes'] * s_part['runtime'].dt.total_seconds() / 3600.0}),
            pd.DataFrame({'time': s_part['start'],  'delta': -s_part['nodes'] * s_part['runtime'].dt.total_seconds() / 3600.0}),
        ], ignore_index=True).dropna(subset=['time']).sort_values('time'))
        s_hours['node_hours'] = s_hours['delta'].cumsum()
        s_hours = s_hours.set_index('time')

        plt.figure(figsize=(12,6), dpi=300)
        _ptitle = fig_title(f'Node-Hours on Queue — Partition: {partition}', 16)
        plt.title(_ptitle)
        plt.ylabel('# of Node-Hours on Queue'); plt.xlabel('Date')
        if not t_hours.empty:
            plt.plot(t_hours.index, t_hours['node_hours'], linewidth=1, label='Ground Truth', color=GT_COLOR)
        if not s_hours.empty:
            plt.plot(s_hours.index, s_hours['node_hours'], linewidth=1, alpha=.85, label='Sim', color=SIM_COLOR)
        plt.xlim([sim_start, sim_end]); plt.gca().xaxis.set_major_formatter(date_format)
        plt.legend(); plt.tight_layout()
        save_fig(fig_dir, 16, title_text=_ptitle, extra_suffix=partition)
        plt.close()
         



    print(" 13) Cluster Power Usage Over Time (MW)")
    # Build event streams (+power at job start, −power at job end), then cumulative sum.
    # We pick the most reliable per-node power column available:
    #   prefer 'true_node_power', else 'node_power', else 'predicted_power'.
    def _pick_power_col(df: pd.DataFrame) -> str:
        for c in ("true_node_power", "node_power", "predicted_power"):
            if c in df.columns:
                return c
        raise KeyError("No power column found in DataFrame (looked for true_node_power/node_power/predicted_power).")

    p_true = _pick_power_col(true_jobs_df)
    p_sim  = _pick_power_col(sim_jobs_df)

    # --- Ground Truth power series
    gt_power_events = pd.concat([
        pd.DataFrame({
            "time":  true_jobs_df["start"],
            "delta": true_jobs_df["nodes"] * true_jobs_df[p_true]
        }),
        pd.DataFrame({
            "time":  true_jobs_df["end"],
            "delta": -true_jobs_df["nodes"] * true_jobs_df[p_true]
        }),
    ], ignore_index=True).dropna(subset=["time"])
    gt_power_events = gt_power_events.sort_values("time")
    gt_power_series = gt_power_events.assign(
        allocated_node_power=lambda d: d["delta"].cumsum()
    ).set_index("time")["allocated_node_power"]

    # --- Simulator power series
    sim_power_events = pd.concat([
        pd.DataFrame({
            "time":  sim_jobs_df["start"],
            "delta": sim_jobs_df["nodes"] * sim_jobs_df[p_sim]
        }),
        pd.DataFrame({
            "time":  sim_jobs_df["end"],
            "delta": -sim_jobs_df["nodes"] * sim_jobs_df[p_sim]
        }),
    ], ignore_index=True).dropna(subset=["time"])
    sim_power_events = sim_power_events.sort_values("time")
    sim_power_series = sim_power_events.assign(
        allocated_node_power=lambda d: d["delta"].cumsum()
    ).set_index("time")["allocated_node_power"]

    # Plot (match your style; divide by 1e6 to get MW)
    plt.figure(figsize=(12, 5), dpi=300)
    _title = fig_title("Power Usage Over Time", 17)
    plt.title(_title, fontsize=16)
    plt.ylabel("Power Usage (MW)", fontsize=14)

    plt.plot(
        gt_power_series.index,
        gt_power_series.values / 1e6,
        linewidth=1.5,
        label="Ground Truth",
        color=GT_COLOR
    )
    plt.plot(
        sim_power_series.index,
        sim_power_series.values / 1e6,
        linewidth=1.5,
        alpha=0.9,
        label="Simulator",
        color=SIM_COLOR
    )

    # X-axis formatting
    plt.xlim([sim_start, sim_end])
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.xticks(rotation=45)

    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    save_fig(fig_dir, 17, title_text=_title)
    plt.close()
     


# ---------- loader ----------
def load_pickled_results(results_dir: Path):
    """Load the three dataframes saved by post-processing."""
    true_fp    = results_dir / "true_jobs_df.pkl"
    sim_fp     = results_dir / "sim_jobs_df.pkl"
    compare_fp = results_dir / "compare_df.pkl"
    if not true_fp.exists() or not sim_fp.exists() or not compare_fp.exists():
        raise FileNotFoundError(
            f"Expected pickles in {results_dir}:\n"
            f"  - true_jobs_df.pkl (exists={true_fp.exists()})\n"
            f"  - sim_jobs_df.pkl  (exists={sim_fp.exists()})\n"
            f"  - compare_df.pkl   (exists={compare_fp.exists()})"
        )
    true_jobs_df = pd.read_pickle(true_fp)
    sim_jobs_df  = pd.read_pickle(sim_fp)
    compare_df   = pd.read_pickle(compare_fp)
    return true_jobs_df, sim_jobs_df, compare_df

# ---------- main ----------
def main():
    parser = argparse.ArgumentParser(description="Generate plot suite from processed results.")
    parser.add_argument("--results-dir", required=True, help="Directory containing true_jobs_df.pkl, sim_jobs_df.pkl, compare_df.pkl")
    parser.add_argument("--sim-start",   required=True, help="ISO datetime (e.g. 2024-09-01T00:00:00)")
    parser.add_argument("--sim-end",     required=True, help="ISO datetime (e.g. 2024-09-15T00:00:00)")
    parser.add_argument("--fig-root",    default="../figures", help="Root directory to store timestamped figures folder")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    sim_start   = pd.to_datetime(args.sim_start)
    sim_end     = pd.to_datetime(args.sim_end)
    fig_root    = Path(args.fig_root)

    # Load inputs
    true_jobs_df, sim_jobs_df, compare_df = load_pickled_results(results_dir)

    # Output dir
    fig_dir = Path("../figures") / results_dir.stem
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # Run plots
    run_all_plots(true_jobs_df, sim_jobs_df, compare_df, sim_start, sim_end, fig_dir)

    print(f"All figures saved to: {fig_dir.resolve()}")

if __name__ == "__main__":
    main()

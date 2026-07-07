# Determinism Check: Baseline Verification Before Jobs-Data Input Feature

- **Date:** 2026-07-06 22:52
- **Status:** Complete
- **Scope:** Simulation reproducibility; baseline for `feature/jobs-data-input`

## Background

We are starting work on changing FastSim's input path from raw SLURM dumps to
a jobs-data format (branch `feature/jobs-data-input`). The regression-testing
strategy for that refactor (per CONTRIBUTING.md §4) relies on FastSim's
determinism claim: identical inputs must produce identical job histories, so
any behavioral drift introduced by the input change will show up as a diff
against a baseline run. Before relying on that property, we verified it
actually holds on the current codebase with a modern environment
(Python 3.12, pandas 3.0.3, numpy 2.5.1 — installed via uv into `.venv`).

## What was run

All runs used `configs/kestrel_conf.yaml` (Kestrel, "No GPU" partition list,
power-aware inputs enabled: `predicted_power.pkl` / `predicted_runtime.pkl`,
`power_model: "regression"`), executed from the repo root:

```
.venv/bin/python scheduler/main.py <config> --output <out.pkl>
```

1. **Smoke test (full window):** sim 2024-09-01 → 2024-09-15. Completed in
   18 min, 155,626 jobs loaded, output `test_run.pkl` (51 MB). Benchmark:
   true mean bounded slowdown 1.101 ± 0.315 vs simulated 1.001 ± 0.002;
   true mean wait 0.157 hr vs simulated 0.003 hr. The sim is optimistic
   relative to history on this lightly-loaded window — a calibration gap to
   keep in mind, but not a blocker for A/B comparisons between sim runs.
2. **Determinism pair (short window):** two runs of an identical config with
   `sim_end` moved to 2024-09-04 (3 days), run concurrently as separate
   processes with separate output paths. Each completed in ~3 min and
   produced 39,620 jobs × 37 columns.

The short-window runs print `nan` benchmark stats — no jobs qualify for the
evaluation filter in a 3-day window. Expected; irrelevant to determinism.

## Comparison methodology

Naive comparisons produce false positives, so the check was done in two
stages:

- **Byte-level:** `cmp` on the two pickles → files are the same size but NOT
  byte-identical. This is expected and is not a determinism failure:
  - `assigned_nodes` values are Python `set`s; set iteration order differs
    across processes (per-process string hash randomization, PYTHONHASHSEED),
    so pickled element order differs even for equal sets.
  - `dependency` values are `job_queue.Dependency` objects whose `conditions`
    contain `Job` object references; reprs/pickles embed process-specific
    detail.
- **Canonicalized content-level:** a script that loads both pickles
  (needs `scheduler/` on `sys.path` to unpickle `job_queue.JobState` /
  `Dependency`) and canonicalizes every cell before comparing:
  sets → sorted tuples; `Job` objects → `("Job", jid)`; enums → names;
  other domain objects → recursive canonicalization of `__dict__`
  (depth-capped); then compares all 37 columns row by row.

**Result: DETERMINISTIC.** All 39,620 rows × 37 columns identical across the
two runs — start/end times, assigned node sets, wait histories, states, and
power fields all match exactly.

The comparison script currently lives in the session scratchpad
(`compare_runs.py`). It should be promoted into the repo (e.g.,
`scripts/compare_results.py` or a future `tests/`) as the baseline-comparison
tool the quality gate calls for — especially before the jobs-data input work
lands. Key implementation notes for that port: canonicalize sets, compare
`Job`s by `jid`, handle `enum.Enum` and `MappingProxyType`, and cap recursion
depth.

## Incidental findings

- **Log path quirk:** `scheduler/main.py:246` hardcodes
  `setup_run_logs(args.output, '../')`, so run logs land in the repo's
  *parent* directory (`~/Artifacts/logfile_<stem>.log` etc.) now that configs
  assume repo-root CWD (paths like `slurm_dump/...`). Should be fixed to
  default to `./logs` or derive from the output path.
- `test_run.pkl` (July, pandas 3.0.3) is coincidentally the same byte size as
  the January `results.pkl` but differs in content; not investigated further
  (same job set; likely pandas-version or set-order artifacts).
- Unpickling results requires `scheduler/` importable (`job_queue` classes
  are embedded in the pickle). Any downstream consumer of results pickles
  inherits this coupling — worth removing when we rework the output format.

## Next steps

- Design the jobs-data input format and reader path
  (`scheduler/data_reader.py:get_cleaned_job_df` is the main touchpoint).
- Capture a baseline results pickle for the chosen reference config before
  refactoring, and wire the canonicalized comparison into the workflow.

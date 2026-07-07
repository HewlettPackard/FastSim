# CLAUDE.md

Guidance for AI agents working on FastSim.

**FastSim** is a lightweight, discrete-event SLURM scheduling simulator that
replays historical workload traces (sacct/sacctmgr dumps) against a simplified
model of SLURM's scheduling behavior, for scheduling-strategy research
(including power-aware prioritization).

## Architecture

Data flow: YAML config → `SlurmDataReader` (cleans dumps) → `Controller`
(builds world state, runs event-driven loop) → job-history pickle
(pandas DataFrame) → post-processing/plots.

All source lives in `scheduler/`:

- `main.py` — CLI entry point. Offline mode (run to completion, pickle
  results) and ZMQ strict-lockstep server mode (`--serve`, for Digital Twin
  integration).
- `controller.py` — the engine. `Controller.run_sim` jumps time
  event-to-event (job submit/finish, node events, reservations, scheduler/
  backfill/fairshare intervals); `_step` runs the per-event pipeline:
  finish jobs → node events → reservations → queue step → fairshare →
  reservation scheduling → main scheduling → backfill.
- `config.py` — config loading. Precedence: YAML > `slurm.conf` > built-in
  defaults. Returns an immutable namedtuple.
- `data_reader.py` — loads/cleans the SLURM dump CSVs and `slurm.conf`. Only
  the job trace and `slurm.conf` are required; assocs (identical-shares
  fairshare tree), QOS (equal priority, no limits), node events (none), and
  reservations (none) are synthesized when their dumps are absent.
- `job_queue.py` — `Queue`, `Job`, `QOS`/`AssocLimit` (limits and holds),
  `Dependency` (after/afterok/singleton).
- `partition.py` — `Partitions`/`Partition`/`Node`; free-block tracking
  (`{reservation: {(start, end): set(nodes)}}`) is the node-availability model.
- `fairshare.py` — fairshare association tree with usage decay.
- `priority_sorters.py` — multi-factor priority (age/size/fairshare/
  partition/QOS) plus optional additive power-aware factor.
- `post_process.py`, `plotting.py` — analysis and figures.
- `interactive_shell.py` — Ctrl-C pause-time inspector of live sim state.
- `aux_funcs.py` — parsing utilities (nodelists, submit lines).

Other directories: `configs/` (YAML configs; `default_conf.yaml` is the
documented template, `kestrel_conf.yaml` the live power-aware example),
`scripts/` (SLURM dump collection), `docs/` (paper + diagram).

Run: `python scheduler/main.py <config.yaml> --output results.pkl`

Determinism matters: identical inputs must produce identical job histories
(tie-breaking uses stable unique IDs). Refactors are verified by comparing
results against a baseline run.

## Development Notes Convention

Development notes live in `dev/` and are named with a datetime tag:
`YYYYMMDDHHMM-descriptor.md` (e.g., `202603101739-project-inception.md`).

When the user says "add this to development notes", "make a dev note", or
similar:

1. Get the current time by running `date '+%Y%m%d%H%M'` — do NOT guess or
   invent timestamps
2. Create a new file in `dev/` with that datetime tag and a descriptive slug
3. The note should provide **comprehensive detail** — not a summary, but a
   thorough record that gives full context to anyone reading it later
4. Include relevant background, rationale, technical details, decisions made,
   alternatives considered, and next steps
5. Reference related dev notes, code, or external resources where applicable
6. Use the standard header format: title, date, status, scope

## Rules

- Follow [CONTRIBUTING.md](CONTRIBUTING.md) — workflow (issue → branch →
  signed-off commits → PR → squash-merge), quality gate, and code standards.
- **Never add Claude to commit tags.** No `Co-Authored-By: Claude` or
  "Generated with Claude" trailers in commit messages or PR descriptions.
- Commits require DCO sign-off: `git commit -s`.
- Never commit real site data (dumps, user/account names, results) — see
  CONTRIBUTING.md §5.

# Contributing

This document describes how we plan, build, and merge changes to FastSim. The
goal is a lightweight, traceable process that keeps `master` green and every
change tied to a discussed plan.

## TL;DR

Issue (with subtasks) → discuss → branch → small signed-off commits → verify →
PR → review → merge. One issue per unit of work; everything references the issue.

---

## 1. Development environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

There is no automated test suite yet. Verify changes by running the simulator
against a known config and comparing results (see §4).

---

## 2. Workflow

### 2.1 Open an issue

Every change starts with an issue that describes **what** is changing and
**why**. Break the work into **subtasks** as a GitHub task list so progress
tracks automatically:

```markdown
- [ ] Subtask A — …
- [ ] Subtask B — …
```

For large or ambiguous work (architecture changes, anything affecting
simulation fidelity or output schemas), include a short design note / RFC in
the issue *before* coding, so the approach — not just the intent — is agreed.

### 2.2 Discuss and get a go

The team reviews and discusses the plan on the issue. When the approach is
agreed, mark it ready to develop (a `ready-to-develop` label or an explicit
"go" comment) so it's unambiguous that branching can start.

### 2.3 Branch

Branch from up-to-date `master`, named by change type with the issue number:

```
fix/<topic>-<issue>      e.g. fix/backfill-window-8
feat/<topic>-<issue>     e.g. feat/jobs-data-input-4
chore/<topic>-<issue>
```

### 2.4 Commit

- Keep commits **small and focused** — ideally one subtask per commit.
- Reference the subtask(s) in the commit so history maps back to the plan.
- Write clear messages: a concise imperative summary line, then a body
  explaining the *what* and *why* when it isn't obvious.
- **Sign off every commit** (Developer Certificate of Origin, see `./DCO`):

  ```bash
  git commit -s -m "Your commit message"
  ```

  PRs with unsigned commits will not be merged.

### 2.5 Open a PR

Open a PR when development is finished, referencing the issue and using a
closing keyword so the merge auto-closes it:

```
Closes #8
```

The PR description is the running log of the change; you generally don't need a
comment-per-commit on the issue. Comment on the issue only for **decisions or
blockers** worth recording for watchers.

---

## 3. Merge strategy

**Squash-merge only.** Every PR lands as exactly one commit on `master`, with
the title taken from the PR title and the body from the PR description.

- **Reproducible `master`.** Every commit is one complete, reviewed change, so
  `git bisect` never lands on a half-finished intermediate state.
- **Transparency is preserved, not lost.** The per-subtask granularity —
  individual branch commits, the full diff, and the review discussion — lives
  on the PR forever and is linked from the squash commit's `(#NN)` suffix.
- **Low friction.** Squash absorbs varied commit hygiene (`wip`, `fix typo`)
  into one clean commit.

Because the squash commit message comes from the PR, **write PR titles as
Conventional Commits** (`fix:`, `feat:`, `chore:`, `docs:`) matching your
branch prefix (§2.3), and treat the PR description as the commit body you want
on `master`.

---

## 4. Quality gate (Definition of Done)

A change is "finished" (ready for PR) when **all** of the following hold:

- All issue subtasks are complete.
- Behavioral changes update the relevant docs (README, config comments).

There is no automated test suite yet (planned). Running a simulation is
expensive, so it is **not** a per-change requirement — reserve the baseline
comparison below for changes that plausibly affect simulation behavior
(scheduler/backfill logic, data cleaning, priority calculation), and rely on
review for everything else.

### Baseline comparison (for behavior-sensitive changes)

FastSim is deterministic — identical inputs must produce identical job
histories — so a refactor can be checked against a baseline run. The
reference config is `configs/kestrel_baseline_conf.yaml` (3-day window,
~3 min runtime):

  ```bash
  # once, at any known-good commit:
  python scheduler/main.py configs/kestrel_baseline_conf.yaml --output results/baseline.pkl
  # after your changes:
  python scheduler/main.py configs/kestrel_baseline_conf.yaml --output results/candidate.pkl
  python scripts/compare_results.py results/baseline.pkl results/candidate.pkl
  ```

  `compare_results.py` canonicalizes process-specific noise (set order,
  object identity), so any reported diff is a real behavioral change.

---

## 5. Code standards

- **No dead code or scaffolding** — every function is reachable from a
  user-facing entry point or a documented API. No placeholders or
  `NotImplementedError` stubs.
- **Single source of truth** — shared logic lives in one canonical module;
  don't duplicate across files.
- **Configuration integrity** — every config parameter must influence runtime
  behavior.
- **Proportional complexity** — match the solution's complexity to the problem.
- **No sensitive data** — never commit real site dumps, user/account names,
  hostnames, or credentials. SLURM dumps and results (`slurm_dump/`,
  `power_data/`, `*.csv`, `*.pkl`) are gitignored; use
  `scripts/slurm_dump_anon.sh` when sharing traces.
- **Known limitations** are documented in the README (Limitations section).

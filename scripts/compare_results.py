"""Compare two FastSim result pickles and report whether they are identical.

Used as the regression check for refactors: FastSim is deterministic, so two
runs of the same config at different commits must produce identical job
histories (see CONTRIBUTING.md section 4 and dev/202607062252-determinism-check.md).

Raw pickle bytes are NOT comparable across processes, so values are
canonicalized before comparison:
- sets/frozensets -> sorted tuples (set iteration order varies per process
  due to string hash randomization)
- Job objects -> ("Job", jid) (object identity is process-specific)
- enums -> (class name, member name)
- other scheduler objects (e.g. Dependency) -> recursive canonicalization
  of their attributes, depth-capped

Usage:
    python scripts/compare_results.py <baseline.pkl> <candidate.pkl>

Exits 0 if identical, 1 with a per-column diff summary otherwise.
"""
import argparse
import enum
import sys
from pathlib import Path
from types import MappingProxyType

# Result pickles reference classes from scheduler/ (JobState, Dependency),
# so it must be importable to unpickle them.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scheduler"))

import pandas as pd

MAX_DEPTH = 6
EXAMPLES_PER_COLUMN = 3


def canon(v, depth=0):
    """Reduce a value to a process-independent canonical form."""
    if depth > MAX_DEPTH:
        return repr(v)
    if isinstance(v, enum.Enum):
        return (type(v).__name__, v.name)
    if isinstance(v, type):
        return f"<class {v.__name__}>"
    if isinstance(v, MappingProxyType):
        v = dict(v)
    if isinstance(v, (set, frozenset)):
        return tuple(sorted((canon(x, depth + 1) for x in v), key=repr))
    if isinstance(v, dict):
        return tuple(sorted(((repr(k), canon(x, depth + 1)) for k, x in v.items()), key=repr))
    if isinstance(v, (list, tuple)):
        return tuple(canon(x, depth + 1) for x in v)
    cls = type(v)
    if cls.__module__ not in ("builtins", "datetime", "numpy", "pandas") and hasattr(v, "__dict__"):
        if hasattr(v, "jid"):  # Job object: identity is its job ID
            return ("Job", v.jid)
        return (cls.__name__,) + canon(v.__dict__, depth + 1)
    return v


def compare(fp1, fp2):
    df1 = pd.read_pickle(fp1)
    df2 = pd.read_pickle(fp2)

    print(f"baseline:  {df1.shape[0]} rows x {df1.shape[1]} cols ({fp1})")
    print(f"candidate: {df2.shape[0]} rows x {df2.shape[1]} cols ({fp2})")

    if list(df1.columns) != list(df2.columns):
        print("MISMATCH: column sets differ")
        print("  only in baseline: ", sorted(set(df1.columns) - set(df2.columns)))
        print("  only in candidate:", sorted(set(df2.columns) - set(df1.columns)))
        return False
    if len(df1) != len(df2):
        print("MISMATCH: row counts differ")
        return False

    ok = True
    for col in df1.columns:
        c1 = df1[col].map(lambda v: repr(canon(v))).values
        c2 = df2[col].map(lambda v: repr(canon(v))).values
        bad = (c1 != c2).nonzero()[0]
        if len(bad):
            ok = False
            print(f"MISMATCH {col}: {len(bad)} differing rows")
            for i in bad[:EXAMPLES_PER_COLUMN]:
                print(f"  row {i} (jid {df1['jid'].iloc[i]}):")
                print(f"    baseline:  {c1[i][:300]}")
                print(f"    candidate: {c2[i][:300]}")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("baseline", help="reference results pickle")
    parser.add_argument("candidate", help="results pickle to check against the baseline")
    args = parser.parse_args()

    if compare(args.baseline, args.candidate):
        print("IDENTICAL: all rows and columns match")
        sys.exit(0)
    sys.exit(1)


if __name__ == "__main__":
    main()

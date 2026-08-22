"""Localise where our LIF and the Brian2 oracle disagree.

Comparing totals only tells you *that* they differ. Splitting by role - the directly
stimulated neurons versus everything downstream - tells you whether the fault is in the
stimulation or in the synaptic dynamics, which is the difference between a one-line fix and
a real bug.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SUGAR_GRNS = {
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
}
MN9 = 720575940660219265


def summarise(label: str, ours: dict, theirs: dict, keys: set[str]) -> None:
    a = np.array([ours.get(k, 0.0) for k in sorted(keys)])
    b = np.array([theirs.get(k, 0.0) for k in sorted(keys)])
    if not len(a):
        print(f"  {label:<28} (none)")
        return
    delta = (a.sum() - b.sum()) / b.sum() * 100 if b.sum() else float("nan")
    print(
        f"  {label:<28} n={len(a):<6} ours={a.sum():>9.1f} Hz  "
        f"theirs={b.sum():>9.1f} Hz  {delta:+6.1f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", type=Path, default=Path("runs/shiu_jax_1trial.json"))
    parser.add_argument("--theirs", type=Path, default=Path("runs/shiu_brian2.json"))
    args = parser.parse_args()

    ours = json.loads(args.ours.read_text())
    theirs = json.loads(args.theirs.read_text())
    rates_a: dict[str, float] = ours["rates"]
    rates_b: dict[str, float] = theirs["rates"]

    print("=" * 78)
    print(f"ours   : {ours['model']}  {ours['spikes_per_trial']:,.0f} spikes/trial")
    print(f"theirs : {theirs['model']}  {theirs['spikes_per_trial']:,.0f} spikes/trial")
    print("=" * 78)

    sugar = {str(f) for f in SUGAR_GRNS}
    active = set(rates_a) | set(rates_b)
    downstream = active - sugar

    summarise("stimulated (sugar GRNs)", rates_a, rates_b, sugar)
    summarise("downstream", rates_a, rates_b, downstream)
    summarise("MN9", rates_a, rates_b, {str(MN9)})

    print("\n  per-GRN rate (ours vs theirs):")
    for key in sorted(sugar):
        print(f"    {key}  {rates_a.get(key, 0):>7.1f}  {rates_b.get(key, 0):>7.1f}")

    print("\n  largest downstream discrepancies:")
    rows = []
    for key in downstream:
        a, b = rates_a.get(key, 0.0), rates_b.get(key, 0.0)
        rows.append((abs(a - b), key, a, b))
    rows.sort(reverse=True)
    for _, key, a, b in rows[:15]:
        ratio = a / b if b else float("inf")
        print(f"    {key}  ours={a:>7.1f}  theirs={b:>7.1f}  x{ratio:.2f}")

    only_ours = sorted(set(rates_a) - set(rates_b))
    only_theirs = sorted(set(rates_b) - set(rates_a))
    print(f"\n  active only in ours  : {len(only_ours)}")
    print(f"  active only in theirs: {len(only_theirs)}")


if __name__ == "__main__":
    main()

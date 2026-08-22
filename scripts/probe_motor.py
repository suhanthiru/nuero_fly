"""Why did PSI -> DLMn come out empty? Probe the descending-to-motor hop directly."""

from __future__ import annotations

import pandas as pd

from data.cell_types import ids_for, matching_types
from data.loader import load_connectome


def ranked(connectome, indices, *, axis: str, top: int = 18) -> None:
    ann = connectome.annotations
    if indices.size == 0:
        print("  (no such neurons)")
        return
    sub = (
        connectome.weights[:, indices].tocoo()
        if axis == "post"
        else connectome.weights[indices].tocoo()
    )
    key = sub.row if axis == "post" else sub.col
    frame = pd.DataFrame({"other": key, "w": sub.data})
    agg = frame.groupby("other")["w"].sum().sort_values(key=abs, ascending=False).head(top)
    for other, weight in agg.items():
        row = ann.iloc[other]
        label = row["cell_type"] or "(untyped)"
        print(f"  {weight:>9,.0f}  {label:<24} {row['superclass']:<20} {row['instance']}")


def main() -> None:
    c = load_connectome("malecns-1.0")

    for pattern in ("PSI*", "*PSI*", "DLM*", "DVM*", "*TTM*"):
        print(f"types matching {pattern:<8}: {matching_types(c, pattern)}")

    print("\nTop DNp01 POSTsynaptic partners (what the giant fiber drives):")
    ranked(c, c.indices_of(ids_for(c, "DNp01")), axis="post")

    print("\nTop DLMn PREsynaptic partners (what drives the wing depressors):")
    ranked(c, c.indices_of(ids_for(c, "DLMn*")), axis="pre")

    print("\nTop TTMn PREsynaptic partners (what drives the leg extensor MN):")
    ranked(c, c.indices_of(ids_for(c, "TTMn")), axis="pre")


if __name__ == "__main__":
    main()

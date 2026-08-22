"""Phase 0.5 reconnaissance - what escape-pathway cells does MaleCNS actually contain,
and which bodies count as neurons?

The build spec supplied a candidate list (LC4, LC6, LC22, LPLC2, GF/DNp01, TTM, DNp02,
DNp04, DNp11). The spec said to verify it against the annotations rather than trust it.
This does that, and reports what is really there rather than what we hoped for.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pyarrow.feather as feather

RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "malecns"
pd.set_option("display.width", 200)


def main() -> None:
    ann = feather.read_table(RAW / "body-annotations.feather").to_pandas()
    print(f"annotation rows: {len(ann)}")

    print("\n--- status ---")
    print(ann["status"].value_counts(dropna=False).to_string())
    print("\n--- statusLabel ---")
    print(ann["statusLabel"].value_counts(dropna=False).head(12).to_string())

    print("\n--- how many bodies have a type? ---")
    print(f"  non-null type:         {ann['type'].notna().sum()}")
    print(f"  non-null superclass:   {ann['superclass'].notna().sum()}")
    print(f"  non-null flywireType:  {ann['flywireType'].notna().sum()}")
    print(f"  non-null mancType:     {ann['mancType'].notna().sum()}")

    # --- escape pathway candidates from the spec ---
    print("\n--- spec candidates: exact type matches ---")
    wanted = ["LC4", "LC6", "LC22", "LPLC2", "DNp01", "DNp02", "DNp04", "DNp11"]
    for name in wanted:
        hits = ann[ann["type"] == name]
        sides = hits["somaSide"].value_counts().to_dict() if len(hits) else {}
        print(f"  {name:<8} n={len(hits):<5} sides={sides}")

    # --- TTM: the spec calls it 'TTM motor neurons'. Find what it is really called. ---
    print("\n--- TTM search across every annotation column ---")
    pattern = re.compile(r"ttm", re.IGNORECASE)
    for col in ("type", "instance", "mancType", "hemibrainType", "synonyms", "flywireType"):
        series = ann[col].dropna().astype(str)
        hits = series[series.str.contains(pattern, na=False)]
        if len(hits):
            print(f"  [{col}] {len(hits)} hits, distinct: {sorted(hits.unique())[:15]}")

    # --- motor neurons in general ---
    print("\n--- vnc_motor / cb_motor types (top 25 by count) ---")
    motor = ann[ann["superclass"].isin(["vnc_motor", "cb_motor"])]
    print(f"  total motor bodies: {len(motor)}")
    print(motor["type"].value_counts().head(25).to_string())

    # --- all descending neurons, to check the spec's DN list against reality ---
    print("\n--- descending neurons: DNp* types present ---")
    dn = ann[ann["superclass"] == "descending_neuron"]
    dnp = sorted({t for t in dn["type"].dropna().unique() if str(t).startswith("DNp")})
    print(f"  {len(dn)} descending bodies, {dn['type'].nunique()} distinct types")
    print(f"  DNp types: {dnp}")

    # --- LC / LPLC visual projection inventory ---
    print("\n--- visual projection LC*/LPLC* types ---")
    vp = ann[ann["superclass"] == "visual_projection"]
    lc = sorted({t for t in vp["type"].dropna().unique() if re.match(r"^L(C|PLC|PC)\d", str(t))})
    print(f"  {len(vp)} visual_projection bodies; LC-family types: {lc}")


if __name__ == "__main__":
    main()

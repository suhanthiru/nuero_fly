"""Does the ours/theirs firing-rate ratio correlate with inhibitory input?

The single-synapse comparison matched Brian2 exactly, including spike times, and the
synaptic delay has been ruled out. The one qualitative feature present in the full network
but absent from that toy is inhibition, which carries 40% of the edges. If our inhibition
were weaker, the neurons we over-drive would be the ones with the most inhibitory input.

Pure analysis over data already on disk - no simulation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

REF = Path(__file__).resolve().parent.parent / "data" / "raw" / "shiu"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", type=Path, default=Path("runs/shiu_jax_1trial.json"))
    parser.add_argument("--theirs", type=Path, default=Path("runs/shiu_brian2_3.json"))
    args = parser.parse_args()

    completeness = pd.read_csv(REF / "2023_03_23_completeness_630_final.csv", index_col=0)
    flywire_ids = np.asarray(completeness.index, dtype=np.uint64)
    index_of = {int(f): i for i, f in enumerate(flywire_ids)}
    n = len(flywire_ids)

    table = pq.read_table(
        REF / "2023_03_23_connectivity_630_final.parquet",
        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
    )
    pre = table.column("Presynaptic_Index").to_numpy()
    post = table.column("Postsynaptic_Index").to_numpy()
    signed = table.column("Excitatory x Connectivity").to_numpy().astype(np.float64)

    print(f"parquet rows        {len(signed):,}")
    weights = sp.coo_matrix((signed, (post, pre)), shape=(n, n)).tocsr()
    weights.sum_duplicates()
    print(f"unique (pre,post)   {weights.nnz:,}")
    print(f"duplicate rows      {len(signed) - weights.nnz:,}")
    print(f"sum of signed w     {signed.sum():,.0f}")

    excitatory = sp.coo_matrix(
        (np.maximum(signed, 0), (post, pre)), shape=(n, n)
    ).tocsr().sum(axis=1).A1
    inhibitory = sp.coo_matrix(
        (np.minimum(signed, 0), (post, pre)), shape=(n, n)
    ).tocsr().sum(axis=1).A1

    ours = json.loads(args.ours.read_text())["rates"]
    theirs = json.loads(args.theirs.read_text())["rates"]

    rows = []
    for key, our_rate in ours.items():
        their_rate = theirs.get(key, 0.0)
        if their_rate < 5.0:  # ratios on near-silent neurons are meaningless
            continue
        i = index_of[int(key)]
        rows.append(
            {
                "flywire_id": key,
                "ours": our_rate,
                "theirs": their_rate,
                "ratio": our_rate / their_rate,
                "exc_in": excitatory[i],
                "inh_in": -inhibitory[i],
            }
        )
    frame = pd.DataFrame(rows)
    frame["inh_fraction"] = frame["inh_in"] / (frame["exc_in"] + frame["inh_in"]).clip(lower=1)

    print(f"\nneurons compared: {len(frame)}")
    print(f"mean ratio ours/theirs: {frame['ratio'].mean():.3f}")

    print("\n--- ratio, bucketed by inhibitory fraction of input ---")
    frame["bucket"] = pd.cut(frame["inh_fraction"], [-0.01, 0.05, 0.15, 0.3, 0.5, 1.01])
    summary = frame.groupby("bucket", observed=True).agg(
        n=("ratio", "size"),
        mean_ratio=("ratio", "mean"),
        mean_inh=("inh_in", "mean"),
        mean_exc=("exc_in", "mean"),
    )
    print(summary.to_string())

    correlation = frame["inh_fraction"].corr(frame["ratio"])
    print(f"\ncorrelation(inhibitory fraction, ratio) = {correlation:+.3f}")
    if correlation > 0.25:
        print("  -> we over-drive precisely the neurons with the most inhibition:")
        print("     our inhibition is too weak.")
    elif correlation < -0.25:
        print("  -> the excess is concentrated on weakly inhibited neurons.")
    else:
        print("  -> no relationship. The excess is not about inhibition.")

    print("\n--- neurons with essentially no inhibitory input ---")
    clean = frame[frame["inh_in"] < 1]
    print(f"  n={len(clean)}, mean ratio {clean['ratio'].mean():.3f}")
    print("  If these already run hot, the cause is upstream of inhibition entirely.")


if __name__ == "__main__":
    main()

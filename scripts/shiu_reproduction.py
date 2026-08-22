"""Phase 1 gate: reproduce the Shiu et al. sugar -> proboscis result.

Stimulating sugar-sensing gustatory receptor neurons should drive the proboscis motor
neuron MN9, through the intervening pathway they report. If we cannot reproduce a published
result on the same data with the same model, every number this project produces afterwards
is worthless - so this runs before anything else is believed.

The comparison is deliberately staged so that a mismatch is unambiguous:

1. Our JAX LIF on *their* preprocessed graph. Isolates the neuron model.
2. Their Brian2 implementation on the same graph. The oracle.
3. Our loader's FlyWire 783 graph compared against their tables structurally.

Their graph is used exactly as published, including the fact that it carries all 15.1M
edges with no synapse-count threshold - unlike our own loader, which floors at 5.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp

from sim.lif import LIF, LIFParams
from sim.neuron import StimulusSpec

REF = Path(__file__).resolve().parent.parent / "data" / "raw" / "shiu"

# The 21 sugar-sensing neurons stimulated in the reference notebook's `neu_sugar`.
SUGAR_GRNS = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]

# The readout. Their notebook: `id_mn9 = 720575940660219265`.
MN9 = 720575940660219265


# FlyWire root ids do not survive across releases, and the published `neu_sugar` list is
# made of 630 ids - one of them is simply absent from 783. Release 630 is also what the
# paper itself ran, so it is the default here and 783 is available for comparison.
RELEASES = {
    "630": ("2023_03_23_completeness_630_final.csv", "2023_03_23_connectivity_630_final.parquet"),
    "783": ("Completeness_783.csv", "Connectivity_783.parquet"),
}


def load_reference_graph(release: str) -> tuple[sp.csr_matrix, np.ndarray, dict[int, int]]:
    """Their connectivity, as a signed ``[post, pre]`` CSR matrix."""
    completeness_file, connectivity_file = RELEASES[release]
    completeness = pd.read_csv(REF / completeness_file, index_col=0)
    flywire_ids = np.asarray(completeness.index, dtype=np.uint64)
    n = len(flywire_ids)

    table = pq.read_table(
        REF / connectivity_file,
        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
    )
    pre = table.column("Presynaptic_Index").to_numpy()
    post = table.column("Postsynaptic_Index").to_numpy()
    signed = table.column("Excitatory x Connectivity").to_numpy().astype(np.float32)

    weights = sp.coo_matrix((signed, (post, pre)), shape=(n, n), dtype=np.float32).tocsr()
    weights.sum_duplicates()
    return weights, flywire_ids, {int(f): i for i, f in enumerate(flywire_ids)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", choices=sorted(RELEASES), default="630")
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--duration", type=float, default=1000.0, help="ms per trial")
    parser.add_argument("--rate", type=float, default=150.0, help="Poisson rate, Hz")
    parser.add_argument("--delay", type=float, default=1.8, help="synaptic delay, ms")
    parser.add_argument("--out", type=Path, default=Path("runs/shiu_jax.json"))
    args = parser.parse_args()

    print(f"loading the reference graph (FlyWire {args.release}) ...")
    weights, flywire_ids, index_of = load_reference_graph(args.release)
    print(f"  neurons  {weights.shape[0]:,}")
    print(f"  edges    {weights.nnz:,}")
    print(f"  synapses {np.abs(weights.data).sum():,.0f}")
    print(f"  excitatory share {(weights.data > 0).mean():.1%}")

    missing = [f for f in SUGAR_GRNS if f not in index_of]
    if missing:
        raise SystemExit(f"sugar neurons absent from the reference graph: {missing}")
    stim = np.array([index_of[f] for f in SUGAR_GRNS], dtype=np.int64)
    mn9 = index_of[MN9]
    print(f"  stimulating {len(stim)} sugar GRNs; reading out MN9 at index {mn9}")

    model = LIF(LIFParams(poisson_rate_hz=args.rate, t_delay=args.delay))
    totals = np.zeros(weights.shape[0], dtype=np.float64)

    for trial in range(args.trials):
        result = model.simulate(
            weights,
            StimulusSpec(poisson_targets=stim, rate_hz=args.rate),
            duration_ms=args.duration,
            seed=trial,
            record=np.array([mn9]),
        )
        totals += result.spike_counts
        print(
            f"  trial {trial}: {int(result.spike_counts.sum()):>8,} spikes, "
            f"{int(result.spike_counts[mn9]):>4} from MN9, "
            f"{int((result.spike_counts > 0).sum()):>6,} neurons active"
        )

    rates = totals / args.trials / (args.duration / 1000.0)

    print("\n" + "=" * 78)
    print("MOST ACTIVE NEURONS")
    print("=" * 78)
    order = np.argsort(rates)[::-1][:25]
    sugar_set = set(SUGAR_GRNS)
    for rank, i in enumerate(order, 1):
        flywire = int(flywire_ids[i])
        tag = "  <- sugar GRN" if flywire in sugar_set else ("  <- MN9" if i == mn9 else "")
        print(f"  {rank:>2}. {flywire}  {rates[i]:>8.1f} Hz{tag}")

    print("\n" + "=" * 78)
    print("READOUT")
    print("=" * 78)
    print(f"  MN9 ({MN9}): {rates[mn9]:.1f} Hz")
    print(f"  neurons active: {int((rates > 0).sum()):,} of {len(rates):,}")
    print(f"  total spikes per trial: {totals.sum() / args.trials:,.0f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "source": f"shiu-{args.release}-reference-graph",
                "model": "jax-lif",
                "trials": args.trials,
                "duration_ms": args.duration,
                "rate_hz": args.rate,
                "mn9_rate_hz": float(rates[mn9]),
                "n_active": int((rates > 0).sum()),
                "spikes_per_trial": float(totals.sum() / args.trials),
                "rates": {
                    str(int(flywire_ids[i])): float(rates[i])
                    for i in np.flatnonzero(rates > 0)
                },
                "top": [
                    {"flywire_id": int(flywire_ids[i]), "rate_hz": float(rates[i])}
                    for i in order
                ],
            },
            indent=2,
        )
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

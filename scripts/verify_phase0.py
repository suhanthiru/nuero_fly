"""Phase 0 exit criterion - compare against published figures.

The rule from the build spec: any discrepancy greater than 10% is a loader bug, to be
found rather than explained away. This script makes the comparisons that are actually
apples-to-apples and is explicit about the ones that are not.
"""

from __future__ import annotations

import numpy as np

from data.cell_types import ids_for
from data.loader import load_connectome


def block(connectome, pre_ids, post_ids):
    """Signed weight and edge count between two explicit id sets."""
    pre = connectome.indices_of(pre_ids)
    post = connectome.indices_of(post_ids)
    if pre.size == 0 or post.size == 0:
        return 0.0, 0
    sub = connectome.weights[post][:, pre]
    return float(sub.sum()), int(sub.nnz)


def compare(label: str, ours: float, published: dict[str, float]) -> None:
    parts = []
    for name, value in published.items():
        delta = (ours - value) / value * 100 if value else float("nan")
        parts.append(f"{name}={value:g} ({delta:+.0f}%)")
    print(f"  {label:<38} ours={ours:<10.6g} published: {', '.join(parts)}")


def main() -> None:
    c = load_connectome("malecns-1.0")

    print("=" * 78)
    print("1. TOTAL NEURON COUNT")
    print("=" * 78)
    compare("proofread neurons", c.n_neurons, {"MaleCNS 'more than 166,000'": 166_000})

    print("\n" + "=" * 78)
    print("2. VISUAL PROJECTION NEURONS PRESYNAPTIC TO THE GIANT FIBER, PER HEMISPHERE")
    print("=" * 78)
    print("   Published counts are per optic lobe, from two other datasets:")
    print("     FAFB20  : 55 LC4 / 108 LPLC2 onto GF dendrites (2442 / 1366 synapses)")
    print("     hemibrain: 71 LC4 /  85 LPLC2 onto GF dendrites (2290 / 1443 synapses)")
    print()
    for cell_type in ("LC4", "LPLC2", "LC6", "LC22"):
        for side in ("L", "R"):
            pre_ids = ids_for(c, cell_type, side=side)
            gf_ids = ids_for(c, "DNp01", side=side)
            weight, edges = block(c, pre_ids, gf_ids)
            print(
                f"  {cell_type:<6} {side}  n={len(pre_ids):<4} "
                f"-> DNp01_{side}: {edges:>4} edges, {weight:>8,.0f} synapses"
            )

    print("\n" + "=" * 78)
    print("3. ARE THE LC6 / LC22 ZEROS REAL, OR AN ARTEFACT OF THE 5-SYNAPSE THRESHOLD?")
    print("=" * 78)
    unthresholded = load_connectome("malecns-1.0", min_synapses=1)
    for cell_type in ("LC4", "LC6", "LC22", "LPLC2"):
        weight, edges = block(
            unthresholded, ids_for(unthresholded, cell_type), ids_for(unthresholded, "DNp01")
        )
        weight5, edges5 = block(c, ids_for(c, cell_type), ids_for(c, "DNp01"))
        print(
            f"  {cell_type:<6} -> DNp01   min_syn=1: {edges:>4} edges / {weight:>8,.0f}"
            f"      min_syn=5: {edges5:>4} edges / {weight5:>8,.0f}"
        )

    print("\n" + "=" * 78)
    print("4. THE DESCENDING-TO-MOTOR HOP (the reason for the dataset switch)")
    print("=" * 78)
    for side in ("L", "R"):
        weight, edges = block(c, ids_for(c, "DNp01", side=side), ids_for(c, "TTMn", side=side))
        print(f"  DNp01_{side} -> TTMn_{side}: {edges} edges, {weight:,.0f} synapses")
    ipsi, _ = block(c, ids_for(c, "DNp01", side="L"), ids_for(c, "TTMn", side="R"))
    print(f"  DNp01_L -> TTMn_R (contralateral): {ipsi:,.0f} synapses")

    print("\n  GF drives the wing depressors indirectly, via the peripherally synapsing")
    print("  interneuron. Direct DNp01 -> DLMn is expected to be zero:")
    for target in ("DLMn*", "PSI*"):
        weight, edges = block(c, ids_for(c, "DNp01"), ids_for(c, target))
        print(f"    DNp01 -> {target:<7}: {edges} edges, {weight:,.0f} synapses")
    weight, edges = block(c, ids_for(c, "PSI*"), ids_for(c, "DLMn*"))
    print(f"    PSI*  -> DLMn*  : {edges} edges, {weight:,.0f} synapses")

    print("\n" + "=" * 78)
    print("5. SIGN CONVENTION - a pathway that must come out negative")
    print("=" * 78)
    ann = c.annotations
    for cell_type in ("LC4", "LPLC2", "DNp01", "TTMn"):
        idx = c.indices_of(ids_for(c, cell_type))
        outgoing = c.weights[:, idx]
        share = (outgoing.data > 0).mean() if outgoing.nnz else float("nan")
        print(f"  {cell_type:<6} outgoing edges: {share:.0%} excitatory")
    gaba = ann.index[ann["cell_type"].str.startswith("CT1")]
    if len(gaba):
        idx = c.indices_of(np.asarray(gaba))
        out = c.weights[:, idx]
        print(f"  CT1 (glutamatergic) outgoing: {(out.data > 0).mean():.0%} excitatory")


if __name__ == "__main__":
    main()

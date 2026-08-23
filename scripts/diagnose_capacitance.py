"""What does capacitance scaling actually do to the giant fiber?

The ablation showed the two extremes failing in opposite directions: under Shiu et al.'s
uniform assumption ~3 coincident LC4 spikes fire the giant fiber, and under full
synapse-count normalisation it goes silent entirely. Both cannot be right, and the useful
output is not a winner but a bracket - so this reports the numbers that define it.
"""

from __future__ import annotations

import numpy as np

from data.cell_types import ids_for
from data.loader import load_connectome
from sim.lif import LIFParams, _capacitance_scale

#: Analytic peak of (g/3)(exp(-t/20) - exp(-t/5)): a single synaptic event moves the
#: membrane by about this fraction of the conductance it delivers.
PEAK_FACTOR = 0.1575


def main() -> None:
    connectome = load_connectome("malecns-1.0")
    params = LIFParams()
    scale = _capacitance_scale(
        LIFParams(capacitance_mode="synapse_count"), connectome.weights
    )
    gap = params.v_threshold - params.v_rest

    incoming = np.abs(connectome.weights).sum(axis=1).A1
    print(f"input synapses: median {np.median(incoming[incoming > 0]):,.0f}, "
          f"max {incoming.max():,.0f}")

    print(f"\n  {'cell':<10} {'input syn':>10} {'scale':>8} "
          f"{'coincident LC4 spikes to fire':>32}")
    lc4 = connectome.indices_of(ids_for(connectome, "LC4"))
    for name in ("DNp01", "TTMn", "DNp11", "GFC2"):
        for index in connectome.indices_of(ids_for(connectome, name))[:1]:
            block = connectome.weights[index, lc4]
            per_spike_mv = (
                float(block.data.mean()) * params.w_synapse if block.nnz else 0.0
            )
            for mode, divisor in (("uniform", 1.0), ("synapse_count", float(scale[index]))):
                effective = per_spike_mv / divisor
                needed = gap / (effective * PEAK_FACTOR) if effective > 0 else float("inf")
                label = f"{name} ({mode})"
                print(f"  {label:<26} {incoming[index]:>10,.0f} {divisor:>8.1f} "
                      f"{needed:>16.1f}")

    print("\n  Uniform makes the giant fiber fire on a handful of coincident inputs, which")
    print("  is implausible for one of the largest neurons in the animal. Full synapse-count")
    print("  normalisation pushes it past what the LC populations can deliver, so it never")
    print("  fires. The real cell is somewhere between, and nothing in the connectome says")
    print("  where: synapse count is a proxy for membrane area, and the relation between")
    print("  area and excitability is not something an EM volume measures.")


if __name__ == "__main__":
    main()

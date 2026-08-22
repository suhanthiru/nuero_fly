"""Why does the giant fiber fire continuously instead of once?

The first sweep had DNp01 emitting 223 spikes and firing 561 ms before contact. Real GFs
fire about once per looming stimulus, near contact. Before changing anything, this
establishes which side the problem is on: the drive profile the encoder produces, or the
gain of the LC -> GF connection in the connectome under Shiu et al.'s weight scaling.
"""

from __future__ import annotations

import numpy as np

from data.cell_types import ids_for
from data.loader import load_connectome
from sim.encoders.analytic import AnalyticLoomingEncoder, LoomingTrajectory
from sim.lif import LIFParams

RADIUS_MM = 10.0
COLLISION_MS = 600.0


def main() -> None:
    connectome = load_connectome("malecns-1.0")
    params = LIFParams()
    encoder = AnalyticLoomingEncoder.from_connectome(connectome)

    gf = connectome.indices_of(ids_for(connectome, "DNp01"))
    lc4 = connectome.indices_of(ids_for(connectome, "LC4"))
    lplc2 = connectome.indices_of(ids_for(connectome, "LPLC2"))

    print("=" * 78)
    print("1. HOW HARD IS THE GIANT FIBER DRIVEN PER PRESYNAPTIC SPIKE?")
    print("=" * 78)
    row = connectome.weights[gf]
    total_exc = float(row.data[row.data > 0].sum())
    total_inh = float(row.data[row.data < 0].sum())
    print(f"  DNp01 total excitatory input : {total_exc:>10,.0f} synapses")
    print(f"  DNp01 total inhibitory input : {total_inh:>10,.0f} synapses")
    print(f"  net                          : {total_exc + total_inh:>10,.0f}")

    lc4_block = connectome.weights[gf][:, lc4]
    per_lc4 = lc4_block.data.mean() if lc4_block.nnz else 0.0
    print(f"\n  mean LC4 -> GF weight        : {per_lc4:.1f} synapses")
    mv = per_lc4 * params.w_synapse
    print(f"  one LC4 spike delivers       : {mv:.2f} mV of g")

    # A single synaptic event's peak effect on v is ~0.1575 * g for these time constants
    # (analytic max of (g/3)(exp(-t/20) - exp(-t/5))).
    peak_factor = 0.1575
    gap = params.v_threshold - params.v_rest
    print(f"  peak membrane deflection     : {mv * peak_factor:.2f} mV")
    print(f"  gap from rest to threshold   : {gap:.2f} mV")
    print(f"  => coincident LC4 spikes needed to fire the GF: "
          f"{gap / (mv * peak_factor):.1f}")

    print("\n" + "=" * 78)
    print("2. WHAT DRIVE PROFILE DOES THE ENCODER PRODUCE?")
    print("=" * 78)
    for ratio in (10.0, 40.0, 80.0):
        speed = RADIUS_MM / ratio
        trajectory = LoomingTrajectory(
            half_size_over_speed_ms=ratio,
            radius_mm=RADIUS_MM,
            start_distance_mm=RADIUS_MM + speed * COLLISION_MS,
        )
        print(f"\n  l/|v| = {ratio:.0f} ms   (speed {speed:.3f} mm/ms)")
        print(f"    {'t (ms)':>8} {'theta':>8} {'theta_dot':>10} {'LC4 Hz':>8} "
              f"{'LPLC2 Hz':>9} {'pop spikes/ms':>14}")
        for t in (0, 50, 100, 200, 300, 400, 500, 550, 590, 600):
            scene = trajectory.state_at(float(t))
            th, thd, a, b = encoder.channel_rates(scene)
            population_rate = (len(lc4) * a + len(lplc2) * b) / 1000.0
            print(f"    {t:>8} {th:>8.1f} {thd:>10.3f} {a:>8.1f} {b:>9.1f} "
                  f"{population_rate:>14.2f}")

    print("\n" + "=" * 78)
    print("3. THE PROBLEM, STATED")
    print("=" * 78)
    print("  The GF needs only a few coincident LC4 spikes. The encoder drives 126 LC4 and")
    print("  185 LPLC2 cells, so even a few Hz per cell is many spikes per millisecond")
    print("  across the population - far above what the GF needs. Under this weight scaling")
    print("  the GF is not an integrator with a meaningful threshold; it fires as soon as")
    print("  the population produces any drive at all.")


if __name__ == "__main__":
    main()

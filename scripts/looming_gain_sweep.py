"""How much of the Phase 2 result is the connectome, and how much is our gain constant?

At nominal gain the giant fiber fires ~150 times per trial, always about 35 ms after
stimulus onset, and escape latency is flat across an eightfold change in looming speed.
That is a negative result, but on its own it does not say whether the connectome cannot
produce the mode split or whether we simply drove it too hard.

So this sweeps the encoder's rate scale over three orders of magnitude and reports, at each
scale, when the GF first fires and whether that timing tracks l/|v|. The rate scale is a
free parameter we invented (Ache et al. fit membrane responses, not firing rates), and the
build spec is explicit that any claim sensitive to scaling needs a sweep behind it.

Loads the connectome once and reuses it, which is most of the runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.cell_types import ids_for
from data.loader import load_connectome
from sim.decoder import decode
from sim.encoders.analytic import AnalyticLoomingEncoder, LoomingTrajectory, LoomingTuning
from sim.lif import LIF, LIFParams
from sim.neuron import StimulusSpec

DURATION_MS = 800.0
COLLISION_MS = 600.0
RADIUS_MM = 10.0


def schedule_for(encoder, trajectory, targets, params, n_steps):
    lc4 = np.array([b in encoder.lc4_ids for b in targets])
    lplc2 = np.array([b in encoder.lplc2_ids for b in targets])
    schedule = np.zeros((n_steps, len(targets)), dtype=np.float32)
    for step in range(n_steps):
        _, _, a, b = encoder.channel_rates(trajectory.state_at(step * params.dt))
        schedule[step, lc4] = a
        schedule[step, lplc2] = b
    return schedule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gains", type=float, nargs="+",
                        default=[0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
    parser.add_argument("--ratios", type=float, nargs="+",
                        default=[10.0, 20.0, 40.0, 80.0])
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("runs/looming_gain_sweep.json"))
    args = parser.parse_args()

    connectome = load_connectome("malecns-1.0")
    params = LIFParams()
    model = LIF(params)
    n_steps = int(round(DURATION_MS / params.dt))

    gf_idx = connectome.indices_of(ids_for(connectome, "DNp01"))
    ttm_idx = connectome.indices_of(ids_for(connectome, "TTMn"))
    record = np.unique(np.concatenate([gf_idx, ttm_idx])).astype(np.int32)

    rows = []
    for gain in args.gains:
        tuning = LoomingTuning(
            lc4_gain_hz_per_deg_per_ms=25.0 * gain,
            lc4_max_hz=150.0 * gain,
            lplc2_max_hz=150.0 * gain,
            max_rate_hz=max(150.0 * gain, 1e-6),
        )
        encoder = AnalyticLoomingEncoder.from_connectome(connectome, tuning=tuning)
        targets = encoder.target_ids()
        target_idx = connectome.indices_of(np.asarray(targets))

        for ratio in args.ratios:
            speed = RADIUS_MM / ratio
            trajectory = LoomingTrajectory(
                half_size_over_speed_ms=ratio, radius_mm=RADIUS_MM,
                start_distance_mm=RADIUS_MM + speed * COLLISION_MS,
            )
            schedule = schedule_for(encoder, trajectory, targets, params, n_steps)
            for trial in range(args.trials):
                outcome = model.simulate(
                    connectome.weights,
                    StimulusSpec(poisson_targets=target_idx, rate_schedule=schedule),
                    duration_ms=DURATION_MS,
                    seed=7000 + 100 * trial + int(ratio),
                    record=record,
                )
                event = decode(
                    outcome.spike_times, gf_indices=gf_idx, ttm_indices=ttm_idx,
                    collision_ms=COLLISION_MS,
                )
                rows.append({
                    "gain": gain, "ratio_ms": ratio, "trial": trial,
                    "mode": event.mode.value,
                    "gf_spikes": event.gf_spike_count,
                    "ttm_spikes": event.ttm_spike_count,
                    "latency_to_collision_ms": event.latency_to_collision_ms,
                })
            last = rows[-1]
            print(f"  gain {gain:<6} l/|v| {ratio:>4.0f} ms: GF={last['gf_spikes']:>4} "
                  f"TTMn={last['ttm_spikes']:>4} latency="
                  f"{last['latency_to_collision_ms'] if last['latency_to_collision_ms'] is not None else float('nan'):>8.1f}")

    print("\n" + "=" * 90)
    print("GF FIRST-SPIKE TIME RELATIVE TO COLLISION (ms), by encoder gain and l/|v|")
    print("=" * 90)
    header = "  gain   " + "".join(f"{r:>13.0f}" for r in args.ratios) + "     spread"
    print(header)
    summary = []
    for gain in args.gains:
        cells, values = [], []
        for ratio in args.ratios:
            latencies = [
                r["latency_to_collision_ms"] for r in rows
                if r["gain"] == gain and r["ratio_ms"] == ratio
                and r["latency_to_collision_ms"] is not None
            ]
            if latencies:
                mean = float(np.mean(latencies))
                values.append(mean)
                cells.append(f"{mean:>13.1f}")
            else:
                cells.append(f"{'never':>13}")
        spread = f"{max(values) - min(values):>10.1f}" if len(values) == len(args.ratios) else "         -"
        print(f"  {gain:<7}" + "".join(cells) + spread)
        summary.append({"gain": gain, "latencies": values})

    print("\n  A real mode split needs escape to come LATER (relative to collision) for slow")
    print("  looming than fast. A flat row means the timing is set by something other than")
    print("  the stimulus; 'never' means the drive was too weak to fire the GF at all.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"ratios": args.ratios, "gains": args.gains, "rows": rows, "summary": summary},
        indent=2,
    ))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

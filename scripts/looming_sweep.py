"""Phase 2 exit criterion: escape latency versus looming speed.

Sweeps l/|v| from 10 to 80 ms and asks whether the escape mode split falls out of the
connectome. Real flies bias towards short, giant-fiber-mediated takeoffs when looming is
fast, and towards longer, wing-raising takeoffs when it is slow (von Reyn et al. 2014,
2017). If that split does not appear here, that is reported as the result. It is not tuned
away: a negative finding about what the connectome alone predicts is a real finding, and
the value of this project rests on the behaviour coming from the graph rather than from our
parameters.

Every trial has the same duration and the same collision time, with the start distance
adjusted per l/|v|. That keeps the latency axis directly comparable across the sweep rather
than confounded with trial length.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.cell_types import ids_for
from data.loader import load_connectome
from sim.decoder import EscapeMode, decode
from sim.encoders.analytic import AnalyticLoomingEncoder, LoomingTrajectory, LoomingTuning
from sim.lif import LIF, LIFParams
from sim.neuron import StimulusSpec

# Fixed trial geometry, so every l/|v| is compared on the same clock.
DURATION_MS = 800.0
COLLISION_MS = 600.0
RADIUS_MM = 10.0

READOUT = {
    "DNp01": "DNp01",
    "TTMn": "TTMn",
    "DNp02": "DNp02",
    "DNp04": "DNp04",
    "DNp11": "DNp11",
    "GFC2": "GFC2",
    "PSI": "PSI",
    "DLMn": "DLMn*",
}


def build_schedule(encoder, trajectory, targets, params, n_steps):
    """(n_steps, n_targets) drive in Hz, plus the theta and theta-dot traces."""
    lc4 = np.array([body_id in encoder.lc4_ids for body_id in targets])
    lplc2 = np.array([body_id in encoder.lplc2_ids for body_id in targets])

    schedule = np.zeros((n_steps, len(targets)), dtype=np.float32)
    theta = np.zeros(n_steps, dtype=np.float32)
    theta_dot = np.zeros(n_steps, dtype=np.float32)

    for step in range(n_steps):
        scene = trajectory.state_at(step * params.dt)
        th, thd, lc4_hz, lplc2_hz = encoder.channel_rates(scene)
        theta[step] = th
        theta_dot[step] = thd
        # Both populations are driven uniformly at azimuth 0, so the per-neuron loop that
        # encode() would do is replaced by two broadcasts.
        schedule[step, lc4] = lc4_hz
        schedule[step, lplc2] = lplc2_hz
    return schedule, theta, theta_dot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="malecns-1.0")
    parser.add_argument(
        "--ratios", type=float, nargs="+",
        default=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        help="l/|v| values in ms",
    )
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument(
        "--gain-scale", type=float, default=1.0,
        help="multiplies every encoder rate. The absolute rate scale is a free parameter "
             "we chose, so the result's dependence on it must be measured, not assumed.",
    )
    parser.add_argument("--out", type=Path, default=Path("runs/looming_sweep.json"))
    args = parser.parse_args()

    print(f"loading {args.dataset} ...")
    connectome = load_connectome(args.dataset)
    params = LIFParams()
    model = LIF(params)
    n_steps = int(round(DURATION_MS / params.dt))

    tuning = LoomingTuning(
        lc4_gain_hz_per_deg_per_ms=25.0 * args.gain_scale,
        lc4_max_hz=150.0 * args.gain_scale,
        lplc2_max_hz=150.0 * args.gain_scale,
        max_rate_hz=max(150.0 * args.gain_scale, 1.0),
    )
    encoder = AnalyticLoomingEncoder.from_connectome(connectome, tuning=tuning)
    targets = encoder.target_ids()
    target_idx = connectome.indices_of(np.asarray(targets))
    print(f"  encoder drives {len(targets)} neurons "
          f"({len(encoder.lc4_ids)} LC4, {len(encoder.lplc2_ids)} LPLC2)")

    readout_idx = {
        name: connectome.indices_of(ids_for(connectome, pattern))
        for name, pattern in READOUT.items()
    }
    record = np.unique(np.concatenate(list(readout_idx.values()))).astype(np.int32)
    print(f"  recording {len(record)} readout neurons")

    results = []
    for ratio in args.ratios:
        speed = RADIUS_MM / ratio
        trajectory = LoomingTrajectory(
            half_size_over_speed_ms=ratio,
            radius_mm=RADIUS_MM,
            start_distance_mm=RADIUS_MM + speed * COLLISION_MS,
        )
        schedule, theta, theta_dot = build_schedule(
            encoder, trajectory, targets, params, n_steps
        )

        for trial in range(args.trials):
            outcome = model.simulate(
                connectome.weights,
                StimulusSpec(poisson_targets=target_idx, rate_schedule=schedule),
                duration_ms=DURATION_MS,
                seed=1000 * trial + int(ratio),
                record=record,
            )
            event = decode(
                outcome.spike_times,
                gf_indices=readout_idx["DNp01"],
                ttm_indices=readout_idx["TTMn"],
                collision_ms=COLLISION_MS,
                dn_indices={k: v for k, v in readout_idx.items()},
            )
            results.append(
                {
                    "ratio_ms": ratio,
                    "trial": trial,
                    "mode": event.mode.value,
                    "gf_spike_ms": event.gf_spike_ms,
                    "ttm_spike_ms": event.ttm_spike_ms,
                    "latency_to_collision_ms": event.latency_to_collision_ms,
                    "gf_spikes": event.gf_spike_count,
                    "ttm_spikes": event.ttm_spike_count,
                    "dn_spikes": event.dn_spike_counts,
                    "peak_theta_deg": float(theta.max()),
                    "peak_theta_dot_deg_per_ms": float(theta_dot.max()),
                    "total_spikes": int(outcome.spike_counts.sum()),
                    "active_neurons": int((outcome.spike_counts > 0).sum()),
                }
            )
            print(
                f"  l/|v|={ratio:>4.0f} ms  trial {trial}: mode={event.mode.value:<5} "
                f"GF={event.gf_spike_count:>3}  TTMn={event.ttm_spike_count:>3}  "
                f"latency={event.latency_to_collision_ms if event.latency_to_collision_ms is not None else float('nan'):>8.1f} ms  "
                f"({outcome.spike_counts.sum():,} spikes total)"
            )

    print("\n" + "=" * 78)
    print("PHASE 2 EXIT CRITERION - escape latency vs looming speed")
    print("=" * 78)
    print(f"{'l/|v| (ms)':>11} {'mode':>7} {'GF spikes':>10} {'TTMn spikes':>12} "
          f"{'latency to collision':>22}")
    for ratio in args.ratios:
        rows = [r for r in results if r["ratio_ms"] == ratio]
        modes = {r["mode"] for r in rows}
        gf = np.mean([r["gf_spikes"] for r in rows])
        ttm = np.mean([r["ttm_spikes"] for r in rows])
        latencies = [
            r["latency_to_collision_ms"] for r in rows
            if r["latency_to_collision_ms"] is not None
        ]
        latency = f"{np.mean(latencies):>8.1f} ms" if latencies else "     never"
        print(f"{ratio:>11.0f} {'/'.join(sorted(modes)):>7} {gf:>10.1f} {ttm:>12.1f} "
              f"{latency:>22}")

    short = sum(1 for r in results if r["mode"] == EscapeMode.SHORT.value)
    long_ = sum(1 for r in results if r["mode"] == EscapeMode.LONG.value)
    none = sum(1 for r in results if r["mode"] == EscapeMode.NONE.value)
    print(f"\n  short-mode trials: {short}\n  long-mode trials : {long_}\n"
          f"  no escape        : {none}  of {len(results)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "dataset": args.dataset,
        "duration_ms": DURATION_MS,
        "collision_ms": COLLISION_MS,
        "radius_mm": RADIUS_MM,
        "tuning": tuning.__dict__,
        "gain_scale": args.gain_scale,
        "lif_params": params.__dict__,
        "results": results,
    }, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

"""Neuron-model ablation: the same connectome, the same task, four dynamics.

Phase 2 ended with the giant fiber saturating - ~150 spikes per trial, escape timing flat
across an eightfold change in looming speed - and attributed that to Shiu et al.'s uniform
parameterisation, under which ~3 coincident LC4 spikes fire one of the largest neurons in
the animal. Phase 3 ended with the escape heading swamped by Poisson noise in a ~6-spike
decision signal. Both diagnoses point at the neuron model rather than the wiring, and both
are testable by swapping it.

The arms:

``lif-uniform``
    Shiu et al. exactly, and the Phase 1 baseline. Every neuron identical.

``lif-capacitance``
    Same dynamics, but incoming weight is divided by each neuron's own input synapse count,
    normalised to the population median - capacitance scaling with membrane area, using
    synapse count as the area proxy. Directly tests the Phase 2 diagnosis.

``conductance``
    Synapses as conductances with reversal potentials, so excitatory drive saturates as the
    membrane approaches the excitatory reversal instead of accumulating without bound. The
    physiologically standard formulation.

``rate``
    No spikes at all. Tests whether anything in this task needs spiking, and - because its
    rate-to-spike conversion is deterministic - whether Phase 3's heading failure really was
    Poisson noise.

Each arm is run at the nominal encoder gain (1.0), which is where the current-based model
saturates, and at the reduced gain (0.03) the demo defaults to. A model that behaves
sensibly at nominal gain needs no such fudge.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.loader import load_connectome
from sim.conductance import ConductanceLIF, ConductanceParams
from sim.lif import LIF, LIFParams
from sim.rate import RateModel, RateParams
from sim.trial import TrialSpec, run_looming_trial

RECORD_TYPES = ["DNp01", "TTMn", "GFC2", "LC4", "LPLC2"]


def build_models() -> dict[str, object]:
    return {
        "lif-uniform": LIF(LIFParams()),
        "lif-capacitance": LIF(LIFParams(capacitance_mode="synapse_count")),
        "conductance": ConductanceLIF(ConductanceParams()),
        "conductance-cap": ConductanceLIF(
            ConductanceParams(capacitance_mode="synapse_count")
        ),
        "rate": RateModel(RateParams()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratios", type=float, nargs="+",
                        default=[10.0, 20.0, 40.0, 80.0])
    parser.add_argument("--gains", type=float, nargs="+", default=[1.0, 0.03])
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--out", type=Path, default=Path("runs/model_ablation.json"))
    args = parser.parse_args()

    connectome = load_connectome("malecns-1.0")
    models = build_models()
    if args.models:
        models = {k: v for k, v in models.items() if k in args.models}

    rows = []
    for name, model in models.items():
        for gain in args.gains:
            for ratio in args.ratios:
                spec = TrialSpec(ratio_ms=ratio, gain_scale=gain, seed=11)
                try:
                    _, event = run_looming_trial(
                        connectome, spec=spec, record_types=RECORD_TYPES, model=model
                    )
                except Exception as error:  # a model may simply not run this task
                    print(f"  {name:<16} gain {gain:<5} l/|v| {ratio:>4.0f}: FAILED {error}")
                    continue
                rows.append({
                    "model": name,
                    "gain": gain,
                    "ratio_ms": ratio,
                    "gf_spikes": event.gf_spike_count,
                    "ttm_spikes": event.ttm_spike_count,
                    "gf_spike_ms": event.gf_spike_ms,
                    "latency_to_collision_ms": event.latency_to_collision_ms,
                    "mode": event.mode.value,
                })
                last = rows[-1]
                latency = last["latency_to_collision_ms"]
                print(
                    f"  {name:<16} gain {gain:<5} l/|v| {ratio:>4.0f} ms: "
                    f"GF={last['gf_spikes']:>4}  TTMn={last['ttm_spikes']:>4}  "
                    f"latency={'never' if latency is None else f'{latency:>8.1f} ms'}"
                )

    print("\n" + "=" * 92)
    print("NEURON-MODEL ABLATION - identical connectome, identical looming task")
    print("=" * 92)
    for gain in args.gains:
        print(f"\n  encoder gain {gain}")
        print(f"  {'model':<18} {'GF spikes (mean)':>17} {'latency spread':>16} "
              f"{'tracks l/|v|?':>15}")
        for name in models:
            subset = [r for r in rows if r["model"] == name and r["gain"] == gain]
            if not subset:
                continue
            gf = np.mean([r["gf_spikes"] for r in subset])
            latencies = [r["latency_to_collision_ms"] for r in subset
                         if r["latency_to_collision_ms"] is not None]
            if len(latencies) < 2:
                spread, verdict = float("nan"), "silent"
            else:
                spread = max(latencies) - min(latencies)
                # A real angular-threshold mechanism makes escape earlier before contact the
                # slower the approach, so latency should fall monotonically with l/|v|.
                by_ratio = [
                    np.mean([r["latency_to_collision_ms"] for r in subset
                             if r["ratio_ms"] == ratio
                             and r["latency_to_collision_ms"] is not None] or [np.nan])
                    for ratio in args.ratios
                ]
                clean = [v for v in by_ratio if not np.isnan(v)]
                monotone = len(clean) == len(by_ratio) and all(
                    b <= a + 1e-9 for a, b in zip(clean, clean[1:])
                )
                verdict = "yes" if (monotone and spread > 50) else "no"
            spread_text = "     -" if np.isnan(spread) else f"{spread:>10.1f} ms"
            print(f"  {name:<18} {gf:>17.1f} {spread_text:>16} {verdict:>15}")

    print("\n  'tracks l/|v|' means escape gets earlier relative to contact as looming slows,")
    print("  monotonically, with more than 50 ms of range - which is what an angular-size")
    print("  threshold predicts and what the real behaviour shows.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"ratios": args.ratios, "gains": args.gains, "rows": rows}, indent=2
    ))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

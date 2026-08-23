"""Phase 3 exit criterion: escape success and direction versus approach angle.

Real flies escape preferentially *away* from the threat. This asks whether ours does.

The sweep runs each approach angle twice, under two conditions:

``weighting off``
    Both optic lobes receive identical drive. This is the null. With no asymmetry in the
    input there is nothing in the model that could produce a heading, so the fly should jump
    straight back every time regardless of where the threat came from.

``weighting on``
    The hand-added hemisphere weighting in the encoder is enabled, so an off-axis stimulus
    drives the two eyes differently and the motor output can become asymmetric.

Running both is the point. Any directional tuning that appears only in the second condition
is a consequence of our scaffolding, not of the connectome, and reporting the tuning without
the null would be reporting our own assumption back to ourselves. Note also that Drosophila
giant fiber responses are reported to be largely azimuth-invariant (Jones et al., J Exp Biol
2023), so strong tuning here would itself be a warning sign.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.loader import load_connectome
from sim.lif import LIF, LIFParams
from sim.trial import TrialSpec, run_looming_trial
from world.arena import Arena
from world.predator import ApproachTrajectory

RECORD_TYPES = ["DNp01", "TTMn", "GFC2", "DNp02", "DNp04", "DNp11"]


def wrap180(degrees: float) -> float:
    return (degrees + 180.0) % 360.0 - 180.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--azimuths", type=float, nargs="+",
        default=[0, 30, 60, 90, 120, 150, 180, -30, -60, -90, -120, -150],
    )
    parser.add_argument("--ratio", type=float, default=20.0, help="l/|v| in ms")
    parser.add_argument("--gain", type=float, default=0.03)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("runs/escape_sweep.json"))
    args = parser.parse_args()

    connectome = load_connectome("malecns-1.0")
    model = LIF(LIFParams())

    rows = []
    for weighting in (False, True):
        label = "on " if weighting else "off"
        print(f"\n--- azimuth weighting {label} ---")
        for azimuth in args.azimuths:
            for trial in range(args.trials):
                spec = TrialSpec(
                    ratio_ms=args.ratio,
                    azimuth_deg=azimuth,
                    gain_scale=args.gain,
                    seed=4000 + 37 * trial + int(azimuth),
                    azimuth_weighting=weighting,
                )
                _, event = run_looming_trial(
                    connectome, spec=spec, record_types=RECORD_TYPES, model=model
                )

                # Physics: the same approach, adjudicated.
                trajectory = ApproachTrajectory(
                    ratio_ms=spec.ratio_ms,
                    radius_mm=spec.radius_mm,
                    azimuth_deg=azimuth,
                    collision_ms=spec.collision_ms,
                )
                takeoff = event.gf_spike_ms if event.gf_spike_ms is not None else event.ttm_spike_ms
                # The heading the decoder produces is in the fly's frame, where 180 is
                # straight back. The fly faces +X, so it is already a world heading.
                outcome = Arena(trajectory).run(
                    takeoff_ms=takeoff,
                    heading_deg=event.heading_deg,
                    duration_ms=spec.duration_ms,
                )

                rows.append({
                    "weighting": weighting,
                    "azimuth_deg": azimuth,
                    "trial": trial,
                    "took_off": outcome.took_off,
                    "takeoff_ms": takeoff,
                    "latency_to_collision_ms": event.latency_to_collision_ms,
                    "gf_spikes": event.gf_spike_count,
                    "ttm_left": event.ttm_left_count,
                    "ttm_right": event.ttm_right_count,
                    "left": event.left_count,
                    "right": event.right_count,
                    "heading_source": event.heading_source,
                    "heading_deg": event.heading_deg,
                    "error_from_away_deg": outcome.error_from_away_deg,
                    "escaped": outcome.escaped,
                    "closest_approach_mm": outcome.closest_approach_mm,
                })
                last = rows[-1]
                print(
                    f"  az={azimuth:>5.0f}  t{trial}  {last['heading_source']:>5} L/R="
                    f"{last['left']:>2}/{last['right']:<2} (TTMn {last['ttm_left']}/"
                    f"{last['ttm_right']})  heading={last['heading_deg']:>6.1f}  "
                    f"escaped={str(last['escaped']):<5}  "
                    f"closest={last['closest_approach_mm']:>7.2f} mm"
                )

    print("\n" + "=" * 84)
    print("PHASE 3 EXIT CRITERION - escape success and direction vs approach angle")
    print("=" * 84)
    for weighting in (False, True):
        subset = [r for r in rows if r["weighting"] == weighting]
        print(f"\n  azimuth weighting {'ON' if weighting else 'OFF (null)'}")
        print(f"  {'azimuth':>8} {'took off':>9} {'heading':>9} {'err from away':>14} "
              f"{'escaped':>9} {'closest mm':>11}")
        for azimuth in args.azimuths:
            group = [r for r in subset if r["azimuth_deg"] == azimuth]
            if not group:
                continue
            took = np.mean([r["took_off"] for r in group])
            heading = np.mean([r["heading_deg"] for r in group])
            errors = [r["error_from_away_deg"] for r in group
                      if r["error_from_away_deg"] is not None]
            escaped = np.mean([r["escaped"] for r in group])
            closest = np.mean([r["closest_approach_mm"] for r in group])
            error = f"{np.mean(errors):>10.1f} deg" if errors else "         -"
            print(f"  {azimuth:>8.0f} {took:>9.0%} {heading:>9.1f} {error:>14} "
                  f"{escaped:>9.0%} {closest:>11.2f}")

        headings = [r["heading_deg"] for r in subset]
        spread = max(headings) - min(headings) if headings else 0.0
        print(f"    heading spread across all azimuths: {spread:.1f} deg")
        errors = [r["error_from_away_deg"] for r in subset
                  if r["error_from_away_deg"] is not None]
        if errors:
            print(f"    mean error from 'directly away'  : {np.mean(errors):.1f} deg "
                  f"(90 deg = no better than chance for a fixed heading)")
        print(f"    escape success rate               : "
              f"{np.mean([r['escaped'] for r in subset]):.0%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"ratio_ms": args.ratio, "gain": args.gain, "azimuths": args.azimuths, "rows": rows},
        indent=2,
    ))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

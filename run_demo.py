"""Run the live demo: simulate one looming approach and stream it to the browser.

    python run_demo.py                      # default trial, serve on :8000
    python run_demo.py --ratio 15 --gain 0.03

Then start the frontend (`npm --prefix viz/frontend run dev`) and open it. The page
connects to this server's websocket and drives the 3D brain from the recorded activity.

The trial is simulated once at startup - a few seconds - and then played back under an
adjustable time dilation. See viz/server.py for why that is the right shape rather than a
compromise.
"""

from __future__ import annotations

import argparse

import uvicorn

import numpy as np

from data.loader import load_connectome
from sim.trial import TrialSpec, run_looming_trial
from world.arena import FLY_HALF_LENGTH_M, FLY_HALF_WIDTH_M, Arena
from world.predator import MM_PER_M, ApproachTrajectory
from viz.palette import CELL_TYPE_COLOR
from viz.server import DemoState, build_app, display_scale_for

# The cells the viewer draws, and therefore the ones worth recording. Taken from the palette
# so the streamed activity and the rendered geometry can never drift apart.
RECORD_TYPES = list(CELL_TYPE_COLOR)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="malecns-1.0")
    parser.add_argument("--ratio", type=float, default=40.0, help="l/|v| in ms")
    parser.add_argument("--gain", type=float, default=0.03, help="encoder gain scale")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--azimuth", type=float, default=35.0,
                        help="approach bearing in degrees; 0 is head-on")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(f"loading {args.dataset} ...")
    connectome = load_connectome(args.dataset)

    def simulate(ratio_ms: float, gain_scale: float, seed: int):
        spec = TrialSpec(
            ratio_ms=ratio_ms, gain_scale=gain_scale, seed=seed,
            azimuth_deg=args.azimuth, azimuth_weighting=True,
        )
        print(f"simulating l/|v|={ratio_ms:g} ms, gain={gain_scale:g}, seed={seed} ...")
        recording, event = run_looming_trial(
            connectome, spec=spec, record_types=RECORD_TYPES
        )
        latency = event.latency_to_collision_ms
        print(
            f"  mode={event.mode.value}  GF spikes={event.gf_spike_count}  "
            f"TTMn spikes={event.ttm_spike_count}  "
            f"latency={'n/a' if latency is None else f'{latency:.1f} ms'}"
        )
        # Physics for the same approach. Run here rather than inside sim/, which must not
        # import world/ - the layering rule from the build spec.
        trajectory = ApproachTrajectory(
            ratio_ms=spec.ratio_ms, radius_mm=spec.radius_mm,
            azimuth_deg=spec.azimuth_deg, collision_ms=spec.collision_ms,
        )
        takeoff = (
            event.gf_spike_ms if event.gf_spike_ms is not None else event.ttm_spike_ms
        )
        outcome = Arena(trajectory).run(
            takeoff_ms=takeoff, heading_deg=event.heading_deg,
            duration_ms=spec.duration_ms,
        )
        print(
            f"  escaped={outcome.escaped}  closest={outcome.closest_approach_mm:.2f} mm  "
            f"heading={event.heading_deg:.0f} deg  azimuth={spec.azimuth_deg:.0f} deg"
        )

        recording.meta["world"] = {
            "fly": (outcome.path * MM_PER_M).tolist(),
            "predator": (outcome.predator_path * MM_PER_M).tolist(),
            "predator_radius_mm": spec.radius_mm,
            "fly_size_mm": [
                FLY_HALF_LENGTH_M * MM_PER_M,
                FLY_HALF_WIDTH_M * MM_PER_M,
                FLY_HALF_WIDTH_M * MM_PER_M,
            ],
            "takeoff_step": (
                None if takeoff is None else int(round(takeoff / recording.dt_ms))
            ),
            "escaped": bool(outcome.escaped),
            "closest_approach_mm": float(outcome.closest_approach_mm),
            "azimuth_deg": float(spec.azimuth_deg),
            "heading_deg": float(event.heading_deg),
        }
        return recording, event

    recording, _ = simulate(args.ratio, args.gain, args.seed)

    state = DemoState(
        recording=recording,
        aggregates=recording.aggregate_by_type(),
        palette=dict(CELL_TYPE_COLOR),
        display_scale=display_scale_for(recording.activity),
        world=recording.meta.get("world"),
        rerun=lambda ratio, gain, seed: simulate(ratio, gain, seed),
    )

    print(f"\n  recording: {recording.n_steps:,} steps over "
          f"{len(recording.body_ids)} neurons")
    print(f"  streaming on ws://{args.host}:{args.port}/stream")
    print(f"  start the frontend with:  npm --prefix viz/frontend run dev\n")

    uvicorn.run(build_app(state), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

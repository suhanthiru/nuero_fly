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

from data.loader import load_connectome
from sim.trial import TrialSpec, run_looming_trial
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
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print(f"loading {args.dataset} ...")
    connectome = load_connectome(args.dataset)

    def simulate(ratio_ms: float, gain_scale: float, seed: int):
        spec = TrialSpec(ratio_ms=ratio_ms, gain_scale=gain_scale, seed=seed)
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
        return recording, event

    recording, _ = simulate(args.ratio, args.gain, args.seed)

    state = DemoState(
        recording=recording,
        aggregates=recording.aggregate_by_type(),
        palette=dict(CELL_TYPE_COLOR),
        display_scale=display_scale_for(recording.activity),
        rerun=lambda ratio, gain, seed: simulate(ratio, gain, seed),
    )

    print(f"\n  recording: {recording.n_steps:,} steps over "
          f"{len(recording.body_ids)} neurons")
    print(f"  streaming on ws://{args.host}:{args.port}/stream")
    print(f"  start the frontend with:  npm --prefix viz/frontend run dev\n")

    uvicorn.run(build_app(state), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

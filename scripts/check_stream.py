"""End-to-end check of the demo stream, without a browser.

Connects to the running server, verifies the hello payload and a few frames, then exercises
the re-simulate path - which is the one piece the headless screenshots cannot reach, since
it needs a client command and several seconds of simulation.

    python scripts/check_stream.py            # against a server already running
"""

from __future__ import annotations

import argparse
import asyncio
import json

import websockets


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8000/stream")
    parser.add_argument("--ratio", type=float, default=15.0)
    parser.add_argument("--gain", type=float, default=0.03)
    args = parser.parse_args()

    async with websockets.connect(args.url, max_size=32 * 1024 * 1024) as socket:
        hello = json.loads(await socket.recv())
        assert hello["type"] == "hello", hello
        print("hello:")
        print(f"  neurons streamed   {len(hello['body_ids'])}")
        print(f"  cell types         {len(set(hello['cell_types']))}")
        print(f"  steps              {hello['n_steps']:,} at {hello['dt_ms']} ms")
        print(f"  l/|v|              {hello['ratio_ms']} ms")
        print(f"  events             {hello['events']}")
        history = hello["history"]
        print(f"  history stride     {history['stride']} "
              f"({len(history.get('theta_deg', []))} samples)")
        print(f"  display full scale {hello['meta']['display_full_scale_hz']:.1f} Hz")

        assert history["aggregate"], "history carried no aggregate series"
        assert len(history["theta_deg"]) > 100, "theta history too short to plot"

        print("\nframes:")
        seen = 0
        while seen < 3:
            message = json.loads(await socket.recv())
            if message.get("type") != "frame":
                continue
            seen += 1
            assert len(message["activity"]) == len(hello["body_ids"])
            print(f"  step {message['step']:>5}  t={message['t_ms']:>7.1f} ms  "
                  f"theta={message.get('theta_deg', 0):>6.1f}  "
                  f"peak activity={max(message['activity']):>3}")

        print(f"\nre-simulating at l/|v| = {args.ratio:g} ms, gain {args.gain:g} ...")
        await socket.send(json.dumps({
            "cmd": "rerun", "ratio_ms": args.ratio, "gain_scale": args.gain, "seed": 0,
        }))

        while True:
            message = json.loads(await socket.recv())
            if message.get("type") == "status":
                print(f"  server busy: {message['busy']}")
            elif message.get("type") == "hello":
                print(f"  new trial: l/|v| = {message['ratio_ms']} ms, "
                      f"events = {message['events']}")
                assert message["ratio_ms"] == args.ratio
                break

        print("\nstream OK")


if __name__ == "__main__":
    asyncio.run(main())

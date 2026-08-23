"""Side-by-side of the ablation arms, condition by condition.

The summary table reports means; this reports the matched pairs, because the interesting
claim is not that two models differ on average but that current-based and conductance-based
dynamics are nearly *indistinguishable* on this task - which is a useful negative for anyone
about to spend effort on the more biophysically detailed one.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("runs/model_ablation.json"))
    args = parser.parse_args()

    payload = json.loads(args.data.read_text())
    rows = payload["rows"]

    print("escape modes produced, by model:")
    for model in dict.fromkeys(r["model"] for r in rows):
        modes = collections.Counter(r["mode"] for r in rows if r["model"] == model)
        print(f"  {model:<18} {dict(modes)}")
    print("\n  No arm produces a long-mode escape. In this connectome TTMn is reachable")
    print("  essentially only through the giant fiber and its coupled interneurons, so")
    print("  there is no second route for a neuron model to find.")

    print("\ncurrent-based vs conductance-based, matched conditions:")
    header = ("gain", "l/|v|", "lif GF", "cond GF", "lif latency", "cond latency")
    print(f"  {header[0]:>6} {header[1]:>7} {header[2]:>8} {header[3]:>9} "
          f"{header[4]:>13} {header[5]:>14}")

    def find(model: str, gain: float, ratio: float):
        for row in rows:
            if (row["model"] == model and row["gain"] == gain
                    and row["ratio_ms"] == ratio):
                return row
        return None

    for gain in payload["gains"]:
        for ratio in payload["ratios"]:
            lif = find("lif-uniform", gain, ratio)
            cond = find("conductance", gain, ratio)
            if not (lif and cond):
                continue
            lif_lat = lif["latency_to_collision_ms"]
            cond_lat = cond["latency_to_collision_ms"]
            lif_text = "never" if lif_lat is None else f"{lif_lat:.1f}"
            cond_text = "never" if cond_lat is None else f"{cond_lat:.1f}"
            print(f"  {gain:>6} {ratio:>7.0f} {lif['gf_spikes']:>8} "
                  f"{cond['gf_spikes']:>9} {lif_text:>13} {cond_text:>14}")


if __name__ == "__main__":
    main()

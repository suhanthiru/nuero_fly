"""Is the escape heading driven by the stimulus, or by Poisson noise?

Phase 3 found headings that scatter across nearly the whole circle in both conditions, with
a mean error from "directly away" close to the 90 degree chance level. That is a negative
result, but "it did not work" is not a finding - this establishes *why*.

The test: a real directional signal would make the left/right giant fiber asymmetry track
sin(azimuth), since a threat on the left should drive the left eye harder. Noise would not.
And comparing the spread of that asymmetry *across* azimuths against its spread *within* a
single azimuth separates signal from trial-to-trial randomness directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("runs/escape_sweep.json"))
    args = parser.parse_args()

    rows = json.loads(args.data.read_text())["rows"]

    for weighting in (False, True):
        subset = [r for r in rows if r["weighting"] == weighting]
        azimuth = np.array([r["azimuth_deg"] for r in subset], dtype=float)
        left = np.array([r["left"] for r in subset], dtype=float)
        right = np.array([r["right"] for r in subset], dtype=float)
        total = left + right
        live = total > 0
        asymmetry = np.where(live, (left - right) / np.maximum(total, 1.0), 0.0)

        # A threat on the fly's left (positive azimuth) should drive the left eye harder.
        expected = np.sin(np.radians(azimuth))
        correlation = float(np.corrcoef(asymmetry[live], expected[live])[0, 1])

        # Spread at a fixed azimuth is pure trial-to-trial noise; spread across azimuths is
        # signal plus noise. If they are the same size, there is no signal.
        within = [
            float(np.std(asymmetry[(azimuth == angle) & live]))
            for angle in sorted(set(azimuth))
            if ((azimuth == angle) & live).sum() > 1
        ]
        noise = float(np.mean(within)) if within else float("nan")
        spread = float(np.std(asymmetry[live]))

        label = "ON " if weighting else "OFF"
        print(f"  azimuth weighting {label}")
        print(f"    decision signal        : {total[live].mean():.1f} GF spikes per trial "
              f"(L {left[live].mean():.1f} / R {right[live].mean():.1f})")
        print(f"    corr(asymmetry, sin az): {correlation:+.3f}   "
              f"(a real directional signal would be strongly positive)")
        print(f"    spread across azimuths : {spread:.3f}")
        print(f"    spread within azimuth  : {noise:.3f}   (pure noise)")
        print(f"    signal-to-noise        : {spread / max(noise, 1e-9):.2f}")
        print()

    print("  A signal-to-noise ratio near 1 means the across-azimuth variation is no larger")
    print("  than the trial-to-trial variation at a single azimuth - the heading is being")
    print("  set by which giant fiber happened to receive more Poisson events, not by where")
    print("  the threat came from.")


if __name__ == "__main__":
    main()

"""Phase 2 figure: escape timing versus looming speed, across encoder gains.

Two panels.

Left: giant fiber first-spike time relative to collision, against l/|v|, one line per
encoder gain. An angular-size threshold predicts a straight line through the origin with
negative slope - escape happens earlier before contact the slower the approach - so that
reference is drawn for comparison. It is the shape to look for, not a fit.

Right: how flat each gain's curve is. Flat means the timing is set by something other than
the stimulus, which is what saturation looks like.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BACKGROUND = "#08090b"
INK = "#e6e8ea"
DIM = "#7b828c"
# Encoder-gain series, cool to warm as drive increases.
COLOURS = ["#4a9eff", "#22d3ee", "#67e8f9", "#fbbf24", "#ff8a3d", "#f43f5e"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("runs/looming_grid.json"))
    parser.add_argument("--out", type=Path, default=Path("runs/looming_latency.png"))
    args = parser.parse_args()

    payload = json.loads(args.data.read_text())
    rows = payload["rows"]
    ratios = sorted({r["ratio_ms"] for r in rows})
    gains = sorted({r["gain"] for r in rows})

    plt.rcParams.update({
        "figure.facecolor": BACKGROUND, "axes.facecolor": BACKGROUND,
        "savefig.facecolor": BACKGROUND, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": DIM, "ytick.color": DIM,
        "axes.edgecolor": "#2a2f36", "grid.color": "#1b1f25",
        "font.family": "monospace", "font.size": 9,
    })
    figure, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.6), width_ratios=[1.5, 1])

    spreads = []
    for colour, gain in zip(COLOURS, gains):
        xs, ys = [], []
        for ratio in ratios:
            values = [
                r["latency_to_collision_ms"] for r in rows
                if r["gain"] == gain and r["ratio_ms"] == ratio
                and r["latency_to_collision_ms"] is not None
            ]
            if values:
                xs.append(ratio)
                ys.append(float(np.mean(values)))
        if not xs:
            spreads.append((gain, np.nan))
            continue
        left.plot(xs, ys, "o-", color=colour, label=f"gain {gain:g}", lw=1.6, ms=4)
        spreads.append((gain, max(ys) - min(ys)))

    # Angular-threshold reference: tau* = r / tan(theta*/2), so escape time before
    # collision is proportional to l/|v|. Scaled to sit in view; the slope is the point.
    reference = -np.array(ratios) / np.tan(np.radians(65.0 / 2.0))
    left.plot(ratios, reference, "--", color=DIM, lw=1.2,
              label="angular threshold\n(theta* = 65 deg)")

    left.axhline(0, color="#3a4048", lw=1)
    left.set_xlabel("l / |v|  (ms)")
    left.set_ylabel("GF first spike relative to collision  (ms)")
    left.set_title("escape timing vs looming speed", color=INK, loc="left", pad=10)
    left.grid(True, lw=0.5, alpha=0.4)
    left.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="lower left")

    valid = [(g, s) for g, s in spreads if not np.isnan(s)]
    right.barh(
        [f"{g:g}" for g, _ in valid], [s for _, s in valid],
        color=[c for c, _ in zip(COLOURS, valid)],
    )
    right.set_xlabel("spread across l/|v|  (ms)")
    right.set_ylabel("encoder gain")
    right.set_title("how much timing tracks the stimulus", color=INK, loc="left", pad=10)
    right.grid(True, axis="x", lw=0.5, alpha=0.4)
    right.invert_yaxis()

    figure.suptitle(
        "Phase 2 - the latency scaling appears only at low encoder gain; "
        "the short/long mode split never appears",
        color=DIM, fontsize=9, x=0.01, ha="left",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")

    print("\nspread across l/|v| by gain:")
    for gain, spread in spreads:
        print(f"  gain {gain:<6} {spread:>8.1f} ms" if not np.isnan(spread)
              else f"  gain {gain:<6}    (GF never fired)")


if __name__ == "__main__":
    main()

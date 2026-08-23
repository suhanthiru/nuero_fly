"""Phase 3 figure: escape direction and success versus approach angle.

Left panel is the directional-tuning plot the exit criterion asks for, drawn in polar form
so "away from the threat" is a readable geometric statement rather than a number. The dashed
line is perfect avoidance - escape heading exactly opposite the approach.

Right panel is escape success rate by approach angle.

Both conditions are drawn together, because the null matters as much as the result: if the
two curves coincide, whatever tuning appears is not coming from the stimulus direction.

They do essentially coincide. Both scatter around the circle rather than following the
ideal, and the success curve is explained by geometry alone - a fly jumping in a roughly
fixed direction escapes frontal threats and is caught by rear ones.
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
NULL_COLOUR = "#64748b"
ON_COLOUR = "#4a9eff"
AWAY_COLOUR = "#f43f5e"


def wrap180(values):
    return (np.asarray(values) + 180.0) % 360.0 - 180.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("runs/escape_sweep.json"))
    parser.add_argument("--out", type=Path, default=Path("runs/escape_tuning.png"))
    args = parser.parse_args()

    payload = json.loads(args.data.read_text())
    rows = payload["rows"]
    azimuths = sorted({r["azimuth_deg"] for r in rows})

    plt.rcParams.update({
        "figure.facecolor": BACKGROUND, "axes.facecolor": BACKGROUND,
        "savefig.facecolor": BACKGROUND, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": DIM, "ytick.color": DIM,
        "axes.edgecolor": "#2a2f36", "grid.color": "#1b1f25",
        "font.family": "monospace", "font.size": 9,
    })
    figure = plt.figure(figsize=(12.0, 5.0))
    polar = figure.add_subplot(1, 2, 1, projection="polar")
    right = figure.add_subplot(1, 2, 2)

    for weighting, colour, label in (
        (False, NULL_COLOUR, "weighting off (null)"),
        (True, ON_COLOUR, "weighting on"),
    ):
        subset = [r for r in rows if r["weighting"] == weighting]
        headings, successes = [], []
        for azimuth in azimuths:
            group = [r for r in subset if r["azimuth_deg"] == azimuth]
            headings.append(np.mean([r["heading_deg"] for r in group]) if group else np.nan)
            successes.append(np.mean([r["escaped"] for r in group]) if group else np.nan)

        polar.plot(
            np.radians(azimuths + [azimuths[0]]),
            np.radians(wrap180(headings + [headings[0]])) % (2 * np.pi),
            "o-", color=colour, lw=1.6, ms=4, label=label,
        )
        right.plot(azimuths, np.array(successes) * 100, "o-", color=colour, lw=1.6,
                   ms=4, label=label)

    # Perfect avoidance: heading exactly opposite the approach bearing.
    ideal = np.array(azimuths + [azimuths[0]])
    polar.plot(
        np.radians(ideal), np.radians(wrap180(ideal + 180.0)) % (2 * np.pi),
        "--", color=AWAY_COLOUR, lw=1.2, label="directly away (ideal)",
    )

    polar.set_theta_zero_location("N")
    polar.set_title("escape heading vs approach azimuth", color=INK, pad=18, loc="left")
    polar.set_xlabel("approach azimuth (angular position)")
    polar.grid(True, lw=0.5, alpha=0.4)
    polar.legend(frameon=False, fontsize=7.5, labelcolor=INK,
                 loc="upper right", bbox_to_anchor=(1.28, 1.13))

    right.set_xlabel("approach azimuth (deg)")
    right.set_ylabel("escape success (%)")
    right.set_title("escape success vs approach angle", color=INK, loc="left", pad=10)
    right.set_ylim(-5, 105)
    right.grid(True, lw=0.5, alpha=0.4)
    right.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc="lower right")

    figure.suptitle(
        "Phase 3 - escape timing works, direction does not: at ~6 giant fiber spikes per "
        "trial the heading is set by Poisson noise, not by where the threat came from",
        color=DIM, fontsize=9, x=0.01, ha="left",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")

    for weighting in (False, True):
        subset = [r for r in rows if r["weighting"] == weighting]
        errors = [r["error_from_away_deg"] for r in subset
                  if r["error_from_away_deg"] is not None]
        headings = [r["heading_deg"] for r in subset]
        spread = float(np.ptp(wrap180(headings))) if headings else 0.0
        print(f"\n  weighting {'ON ' if weighting else 'OFF'}:")
        print(f"    heading spread across azimuth : {spread:>6.1f} deg")
        print(f"    mean error from directly away : {np.mean(errors):>6.1f} deg")
        print(f"    escape success                : "
              f"{np.mean([r['escaped'] for r in subset]):.0%}")


if __name__ == "__main__":
    main()

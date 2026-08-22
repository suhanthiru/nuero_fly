"""Run the Shiu et al. Brian2 model as the Phase 1 oracle.

This deliberately calls *their* ``model.py`` rather than a transcription of it, so that any
disagreement with our JAX implementation is a real disagreement and not a copying mistake
on our side. Their file is fetched into ``data/raw/shiu`` (gitignored) on first use.

Output is a JSON of per-neuron spike counts, directly comparable with the output of
``scripts/shiu_reproduction.py``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REF = Path(__file__).resolve().parent.parent / "data" / "raw" / "shiu"
MODEL_URL = "https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/model.py"

SUGAR_GRNS = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
MN9 = 720575940660219265

RELEASES = {
    "630": ("2023_03_23_completeness_630_final.csv", "2023_03_23_connectivity_630_final.parquet"),
    "783": ("Completeness_783.csv", "Connectivity_783.parquet"),
}


def load_reference_module():
    """Import their model.py, downloading it if absent."""
    path = REF / "model.py"
    if not path.exists():
        import urllib.request

        REF.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, path)
    spec = importlib.util.spec_from_file_location("shiu_model", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["shiu_model"] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", choices=sorted(RELEASES), default="630")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--duration", type=float, default=1000.0, help="ms per trial")
    parser.add_argument("--rate", type=float, default=150.0, help="Poisson rate, Hz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/shiu_brian2.json"))
    args = parser.parse_args()

    from brian2 import Hz, ms, prefs, seed as brian_seed

    prefs.codegen.target = "numpy"  # cython would be faster but needs a build step

    model = load_reference_module()
    completeness_file, connectivity_file = RELEASES[args.release]
    path_comp = str(REF / completeness_file)
    path_con = str(REF / connectivity_file)

    completeness = pd.read_csv(path_comp, index_col=0)
    flywire_ids = np.asarray(completeness.index, dtype=np.uint64)
    index_of = {int(f): i for i, f in enumerate(flywire_ids)}

    exc = [index_of[f] for f in SUGAR_GRNS]
    mn9 = index_of[MN9]

    params = dict(model.default_params)
    params["t_run"] = args.duration * ms
    params["r_poi"] = args.rate * Hz

    print(f"brian2 oracle: FlyWire {args.release}, {len(flywire_ids):,} neurons")
    print(f"  {args.trials} trial(s) x {args.duration:g} ms at {args.rate:g} Hz")

    totals = np.zeros(len(flywire_ids), dtype=np.float64)
    for trial in range(args.trials):
        brian_seed(args.seed + trial)
        spike_trains = model.run_trial(
            exc=exc, exc2=[], slnc=[], path_comp=path_comp, path_con=path_con, params=params
        )
        counts = np.zeros(len(flywire_ids), dtype=np.float64)
        for neuron, times in spike_trains.items():
            counts[int(neuron)] = len(times)
        totals += counts
        print(
            f"  trial {trial}: {int(counts.sum()):>8,} spikes, "
            f"{int(counts[mn9]):>4} from MN9, {int((counts > 0).sum()):>6,} active"
        )

    rates = totals / args.trials / (args.duration / 1000.0)
    order = np.argsort(rates)[::-1][:25]

    print(f"\n  MN9: {rates[mn9]:.1f} Hz")
    print(f"  neurons active: {int((rates > 0).sum()):,}")
    print(f"  spikes per trial: {totals.sum() / args.trials:,.0f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "source": f"shiu-{args.release}-reference-graph",
                "model": "brian2-reference",
                "trials": args.trials,
                "duration_ms": args.duration,
                "rate_hz": args.rate,
                "mn9_rate_hz": float(rates[mn9]),
                "n_active": int((rates > 0).sum()),
                "spikes_per_trial": float(totals.sum() / args.trials),
                "rates": {
                    str(int(flywire_ids[i])): float(rates[i])
                    for i in np.flatnonzero(rates > 0)
                },
                "top": [
                    {"flywire_id": int(flywire_ids[i]), "rate_hz": float(rates[i])}
                    for i in order
                ],
            },
            indent=2,
        )
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

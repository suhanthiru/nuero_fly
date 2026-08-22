"""Deterministic cascade on the real graph: one forced spike, no Poisson anywhere.

Everything cheap has been ruled out. A single synapse matches Brian2 exactly, including
spike times; the synaptic delay accounts for ~1%; the excess does not correlate with
inhibitory input and is a flat ~1.29x across the whole population. That pattern - uniform,
and invisible in a two-neuron test - is what recurrent amplification of a small systematic
difference looks like.

So: drive one neuron with one spike at a known time, on the full graph, with no stochastic
input at all, and compare the evoked cascade spike-for-spike. Any divergence is now
attributable, and can be bisected by time.

    <ref-venv>/bin/python scripts/cascade_compare.py --mode brian2
    <venv>/bin/python     scripts/cascade_compare.py --mode jax
    <venv>/bin/python     scripts/cascade_compare.py --mode compare
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REF = Path(__file__).resolve().parent.parent / "data" / "raw" / "shiu"
OUT = Path("runs")
COMPLETENESS = "2023_03_23_completeness_630_final.csv"
CONNECTIVITY = "2023_03_23_connectivity_630_final.parquet"

# All 21 sugar GRNs, driven periodically instead of by Poisson so that both simulators
# receive an identical, reproducible input train. A single spike evokes nothing - the graph
# is far too sparse for one spike to propagate - so sustained drive is needed to produce a
# cascade worth comparing.
SUGAR_GRNS = [
    720575940624963786, 720575940630233916, 720575940637568838, 720575940638202345,
    720575940617000768, 720575940630797113, 720575940632889389, 720575940621754367,
    720575940621502051, 720575940640649691, 720575940639332736, 720575940616885538,
    720575940639198653, 720575940620900446, 720575940617937543, 720575940632425919,
    720575940633143833, 720575940612670570, 720575940628853239, 720575940629176663,
    720575940611875570,
]
DRIVE_HZ = 150.0
DURATION = 500.0    # ms
START_AT = 10.0     # ms

# Brian2 delivers the kick in the synapses slot, so its neuron crosses threshold one step
# later than a directly forced spike does. Offsetting keeps the driven spike trains aligned.
DT = 0.1


def drive_times() -> np.ndarray:
    period = 1000.0 / DRIVE_HZ
    return np.arange(START_AT, DURATION, period)


def neuron_index() -> tuple[np.ndarray, dict[int, int]]:
    completeness = pd.read_csv(REF / COMPLETENESS, index_col=0)
    flywire_ids = np.asarray(completeness.index, dtype=np.uint64)
    return flywire_ids, {int(f): i for i, f in enumerate(flywire_ids)}


def run_jax() -> None:
    import pyarrow.parquet as pq
    import scipy.sparse as sp

    from sim.lif import LIF, LIFParams
    from sim.neuron import StimulusSpec

    flywire_ids, index_of = neuron_index()
    n = len(flywire_ids)
    table = pq.read_table(
        REF / CONNECTIVITY,
        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
    )
    weights = sp.coo_matrix(
        (
            table.column("Excitatory x Connectivity").to_numpy().astype(np.float32),
            (
                table.column("Postsynaptic_Index").to_numpy(),
                table.column("Presynaptic_Index").to_numpy(),
            ),
        ),
        shape=(n, n),
        dtype=np.float32,
    ).tocsr()
    weights.sum_duplicates()

    result = LIF(LIFParams()).simulate(
        weights,
        StimulusSpec(poisson_targets=np.array([], dtype=np.int64)),
        duration_ms=DURATION,
        seed=0,
        forced_spikes={
            index_of[f]: drive_times() + DT for f in SUGAR_GRNS
        },
    )
    OUT.mkdir(exist_ok=True)
    np.savez(
        OUT / "cascade_jax.npz",
        counts=result.spike_counts,
        driven=np.array([index_of[f] for f in SUGAR_GRNS]),
    )
    print(f"jax:    {int(result.spike_counts.sum()):,} spikes, "
          f"{int((result.spike_counts > 0).sum()):,} neurons active")


def run_brian2() -> None:
    from brian2 import (
        Network,
        NeuronGroup,
        SpikeGeneratorGroup,
        SpikeMonitor,
        Synapses,
        defaultclock,
        mV,
        ms,
        prefs,
    )
    import pandas as pd_

    prefs.codegen.target = "numpy"
    defaultclock.dt = 0.1 * ms

    flywire_ids, index_of = neuron_index()
    n = len(flywire_ids)
    connectivity = pd_.read_parquet(
        REF / CONNECTIVITY,
        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
    )

    params = {"v_0": -52 * mV, "v_rst": -52 * mV, "v_th": -45 * mV,
              "t_mbr": 20 * ms, "tau": 5 * ms}
    eqs = """
    dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau               : volt (unless refractory)
    rfc                            : second
    """
    neurons = NeuronGroup(
        n, model=eqs, method="linear", threshold="v > v_th",
        reset="v = v_rst; g = 0 * mV", refractory="rfc", namespace=params,
    )
    neurons.v = params["v_0"]
    neurons.g = 0 * mV
    neurons.rfc = 2.2 * ms

    synapses = Synapses(neurons, neurons, "w : volt", on_pre="g += w", delay=1.8 * ms)
    synapses.connect(
        i=connectivity["Presynaptic_Index"].values,
        j=connectivity["Postsynaptic_Index"].values,
    )
    synapses.w = connectivity["Excitatory x Connectivity"].values * 0.275 * mV

    # Deterministic drive: a fixed spike train kicking each GRN over threshold.
    times = drive_times()
    generator = SpikeGeneratorGroup(1, np.zeros(len(times), dtype=int), times * ms)
    kick = Synapses(generator, neurons, on_pre="v += 100 * mV")
    for flywire in SUGAR_GRNS:
        kick.connect(i=0, j=index_of[flywire])
    neurons.rfc[[index_of[f] for f in SUGAR_GRNS]] = 0 * ms

    monitor = SpikeMonitor(neurons)
    net = Network(neurons, synapses, generator, kick, monitor)
    net.run(DURATION * ms)

    counts = np.zeros(n)
    for neuron, times in monitor.spike_trains().items():
        counts[int(neuron)] = len(times)
    OUT.mkdir(exist_ok=True)
    np.savez(
        OUT / "cascade_brian2.npz",
        counts=counts,
        driven=np.array([index_of[f] for f in SUGAR_GRNS]),
    )
    print(f"brian2: {int(counts.sum()):,} spikes, {int((counts > 0).sum()):,} neurons active")


def compare() -> None:
    a = np.load(OUT / "cascade_jax.npz")
    b = np.load(OUT / "cascade_brian2.npz")
    ours, theirs, driven = a["counts"], b["counts"], a["driven"]

    print(f"driven GRNs   : ours {int(ours[driven].sum()):,}  "
          f"theirs {int(theirs[driven].sum()):,}  (should match exactly)")
    rest = np.ones_like(ours, dtype=bool)
    rest[driven] = False
    print(f"downstream    : ours {int(ours[rest].sum()):,}  theirs {int(theirs[rest].sum()):,}")

    print(f"ours   : {int(ours.sum()):,} spikes over {int((ours > 0).sum()):,} neurons")
    print(f"theirs : {int(theirs.sum()):,} spikes over {int((theirs > 0).sum()):,} neurons")
    if theirs.sum():
        print(f"ratio  : {ours.sum() / theirs.sum():.3f}")

    both = (ours > 0) | (theirs > 0)
    difference = ours - theirs
    order = np.argsort(-np.abs(difference))
    print("\nlargest per-neuron differences:")
    print("   index      ours   theirs     diff")
    for i in order[:15]:
        if difference[i] == 0:
            break
        print(f"  {i:>7}  {ours[i]:>8.0f} {theirs[i]:>8.0f} {difference[i]:>8.0f}")

    print(f"\nneurons active in both      : {int(((ours > 0) & (theirs > 0)).sum()):,}")
    print(f"neurons active only in ours : {int(((ours > 0) & (theirs == 0)).sum()):,}")
    print(f"neurons active only in theirs: {int(((ours == 0) & (theirs > 0)).sum()):,}")
    print(f"exact per-neuron agreement  : {int((ours == theirs)[both].sum()):,} of {int(both.sum()):,}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("jax", "brian2", "compare"), required=True)
    args = parser.parse_args()
    {"jax": run_jax, "brian2": run_brian2, "compare": compare}[args.mode]()


if __name__ == "__main__":
    main()

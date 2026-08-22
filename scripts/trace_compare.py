"""Compare our LIF against Brian2 on a single synapse, with no randomness anywhere.

The population comparison showed downstream neurons firing ~30% too fast while directly
stimulated ones matched. That points at synaptic integration rather than at the drive, but
population statistics cannot say *which part*. This drives one presynaptic cell at exact
known times and compares the postsynaptic membrane trace sample by sample, which can.

Run twice, once per environment, then compare::

    <ref-venv>/bin/python scripts/trace_compare.py --mode brian2
    <venv>/bin/python     scripts/trace_compare.py --mode jax
    <venv>/bin/python     scripts/trace_compare.py --mode compare
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

OUT = Path("runs")
SPIKE_TIMES = np.array([10.0, 30.0, 50.0, 70.0, 90.0])  # ms
DURATION = 120.0  # ms
DT = 0.1
SYNAPSE_COUNT = 10.0  # x 0.275 mV = 2.75 mV, deliberately subthreshold


def run_jax(count: float) -> None:
    import scipy.sparse as sp

    from sim.lif import LIF, LIFParams
    from sim.neuron import StimulusSpec

    weights = sp.csr_matrix(np.array([[0.0, 0.0], [count, 0.0]], dtype=np.float32))
    result = LIF(LIFParams()).simulate(
        weights,
        StimulusSpec(poisson_targets=np.array([], dtype=np.int64)),
        duration_ms=DURATION,
        seed=0,
        record=np.array([0, 1]),
        forced_spikes={0: SPIKE_TIMES},
    )
    OUT.mkdir(exist_ok=True)
    np.savez(
        OUT / "trace_jax.npz",
        v_post=result.voltages[1],
        v_pre=result.voltages[0],
        spikes_post=result.spike_times[1],
    )
    print(f"jax: post spiked {len(result.spike_times[1])} times")


def run_brian2(count: float) -> None:
    from brian2 import (
        Network,
        NeuronGroup,
        SpikeGeneratorGroup,
        StateMonitor,
        SpikeMonitor,
        Synapses,
        defaultclock,
        mV,
        ms,
        prefs,
    )

    prefs.codegen.target = "numpy"
    defaultclock.dt = DT * ms

    params = {
        "v_0": -52 * mV,
        "v_rst": -52 * mV,
        "v_th": -45 * mV,
        "t_mbr": 20 * ms,
        "tau": 5 * ms,
    }
    eqs = """
    dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau               : volt (unless refractory)
    rfc                            : second
    """

    neurons = NeuronGroup(
        1,
        model=eqs,
        method="linear",
        threshold="v > v_th",
        reset="v = v_rst; g = 0 * mV",
        refractory="rfc",
        namespace=params,
    )
    neurons.v = params["v_0"]
    neurons.g = 0 * mV
    neurons.rfc = 2.2 * ms

    generator = SpikeGeneratorGroup(1, np.zeros(len(SPIKE_TIMES), dtype=int), SPIKE_TIMES * ms)
    synapse = Synapses(generator, neurons, "w : volt", on_pre="g += w", delay=1.8 * ms)
    synapse.connect(i=0, j=0)
    synapse.w = count * 0.275 * mV

    state = StateMonitor(neurons, ["v", "g"], record=True)
    spikes = SpikeMonitor(neurons)
    net = Network(neurons, generator, synapse, state, spikes)
    net.run(DURATION * ms)

    OUT.mkdir(exist_ok=True)
    np.savez(
        OUT / "trace_brian2.npz",
        v_post=np.asarray(state.v[0] / mV),
        g_post=np.asarray(state.g[0] / mV),
        spikes_post=np.asarray(spikes.t / ms),
    )
    print(f"brian2: post spiked {len(spikes.t)} times")


def compare() -> None:
    ours = np.load(OUT / "trace_jax.npz")
    theirs = np.load(OUT / "trace_brian2.npz")
    a, b = ours["v_post"], theirs["v_post"]
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    print(f"samples: ours {len(ours['v_post'])}, theirs {len(theirs['v_post'])}")
    print(f"resting: ours {a[0]:.6f} mV, theirs {b[0]:.6f} mV")
    print(f"peak   : ours {a.max():.6f} mV, theirs {b.max():.6f} mV")
    print(f"peak deflection from rest: ours {a.max() + 52:.6f}, theirs {b.max() + 52:.6f}")
    print(f"max |difference|: {np.abs(a - b).max():.6f} mV")

    first = np.flatnonzero(np.abs(a - b) > 1e-4)
    if len(first):
        i = first[0]
        print(f"\nfirst divergence at sample {i} (t = {i * DT:.1f} ms):")
        lo, hi = max(0, i - 3), min(n, i + 12)
        print("   t(ms)      ours     theirs       diff")
        for j in range(lo, hi):
            print(f"  {j * DT:7.1f}  {a[j]:9.5f}  {b[j]:9.5f}  {a[j] - b[j]:9.5f}")
    else:
        print("\ntraces agree to 1e-4 mV")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("jax", "brian2", "compare"), required=True)
    parser.add_argument("--count", type=float, default=SYNAPSE_COUNT)
    args = parser.parse_args()

    if args.mode == "jax":
        run_jax(args.count)
    elif args.mode == "brian2":
        run_brian2(args.count)
    else:
        compare()


if __name__ == "__main__":
    main()

"""Does Brian2 actually freeze `g` during the refractory period?

Shiu et al.'s equations mark both states ``(unless refractory)``, which reads as "neither is
integrated while refractory". Our cascade comparison says otherwise: freezing `g` overshoots
the reference by 11%, while letting it keep decaying lands within 2%. That is strong
evidence but it is still inference, so this measures the behaviour directly.

Setup: one neuron with a deliberately long refractory period. Drive it over threshold, then
deliver a synaptic event while it is refractory, and watch `g`. If `g` holds flat, the
annotation clamps it. If it decays, it does not.
"""

from __future__ import annotations

import numpy as np
from brian2 import (
    Network,
    NeuronGroup,
    SpikeGeneratorGroup,
    StateMonitor,
    Synapses,
    defaultclock,
    mV,
    ms,
    prefs,
)

prefs.codegen.target = "numpy"
defaultclock.dt = 0.1 * ms

params = {"v_0": -52 * mV, "v_rst": -52 * mV, "v_th": -45 * mV,
          "t_mbr": 20 * ms, "tau": 5 * ms}
eqs = """
dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
dg/dt = -g / tau               : volt (unless refractory)
rfc                            : second
"""

neurons = NeuronGroup(
    1, model=eqs, method="linear", threshold="v > v_th",
    reset="v = v_rst; g = 0 * mV", refractory="rfc", namespace=params,
)
neurons.v = params["v_0"]
neurons.g = 0 * mV
import sys
RFC = float(sys.argv[1]) if len(sys.argv) > 1 else 40.0
neurons.rfc = RFC * ms   # long -> event lands during refractoriness; short -> control

# t = 5 ms: kick it over threshold so it spikes and enters refractoriness.
# t = 15 ms: deliver a synaptic event while it is still refractory.
generator = SpikeGeneratorGroup(2, [0, 1], [5.0, 15.0] * ms)
kick = Synapses(generator, neurons, on_pre="v += 100 * mV")
kick.connect(i=0, j=0)
event = Synapses(generator, neurons, on_pre="g += 5 * mV")
event.connect(i=1, j=0)

state = StateMonitor(neurons, ["v", "g"], record=True)
net = Network(neurons, generator, kick, event, state)
net.run(40 * ms)

times = np.asarray(state.t / ms)
g = np.asarray(state.g[0] / mV)
v = np.asarray(state.v[0] / mV)


def at(t_ms: float) -> int:
    return int(np.argmin(np.abs(times - t_ms)))


print("  t(ms)      v(mV)      g(mV)")
for t in (4.9, 5.1, 10.0, 15.0, 15.2, 16.0, 20.0, 25.0, 30.0, 35.0):
    i = at(t)
    print(f"  {times[i]:5.1f}  {v[i]:9.4f}  {g[i]:9.4f}")

g_after = g[at(15.2)]
g_later = g[at(25.0)]
print(f"\ng just after the event (t=15.2): {g_after:.4f} mV")
print(f"g ten ms later      (t=25.0): {g_later:.4f} mV")
print(f"rfc = {RFC} ms, spiked at ~5 ms, event at 15 ms")

expected_if_decaying = g_after * np.exp(-9.8 / 5.0)
print(f"\nif g decayed with tau=5ms it would be {expected_if_decaying:.4f} mV")
if abs(g_later - g_after) < 1e-3:
    print("VERDICT: g is FROZEN during the refractory period.")
elif abs(g_later - expected_if_decaying) < 0.05 * max(abs(expected_if_decaying), 1e-6):
    print("VERDICT: g KEEPS DECAYING during the refractory period.")
else:
    print("VERDICT: neither - g does something else entirely.")

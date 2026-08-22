# Connectome-driven looming-escape simulation

An embodied simulation of the *Drosophila* giant-fiber escape circuit, driven by a real
synapse-resolution connectome rather than by hand-written dynamics, with a live 3D
brain-activity view.

The scientific target is the **escape mode split**. Real flies show two takeoff modes: a
short-latency (~5-10 ms) giant-fiber-mediated escape at fast looming, and a longer-latency,
directionally-tuned escape at slow looming. Which mode fires depends on looming kinematics.
Whether that split falls out of the connectome alone is the question this project exists to
answer.

The value of the project depends entirely on the behaviour coming from the graph rather than
from our code. Every hand-tuned parameter erodes that, so each one carries a comment saying
why it exists.

## Status

Phase 0 (connectome loader) and Phase 1 (LIF core and published-result reproduction).
Phases 2-4 are designed but not built.

## Data

All sources are anonymous public buckets. No CAVE credentials, no login, nothing gated.

| dataset | role | licence |
|---|---|---|
| **MaleCNS v1.0** (Janelia FlyEM) | primary substrate | CC-BY |
| **FlyWire FAFB 783** (Codex flat files) | Phase 1 validation only | see FlyWire citation guidelines |

`bash scripts/fetch_data.sh` pulls both into `data/raw/`. Neither is committed.

### Why MaleCNS and not FlyWire

The original plan was FlyWire. FlyWire is a *brain* volume: it ends at the neck connective,
DNp01's axon is truncated there, and the tergotrochanteral motor neuron (TTMn) that actually
extends the legs lives in the ventral nerve cord, which is a different dataset from a
different animal. Simulating a "giant fiber escape circuit" without its motor neuron, or
stitching a female brain to a male nerve cord, were both bad options.

MaleCNS v1.0 is one male fly's entire central nervous system - central brain, both optic
lobes, and the full ventral nerve cord, with an **intact neck connective**. The giant fiber
runs continuously into the VNC in the same volume, so GF -> TTMn is real measured
connectivity in a single animal.

FlyWire 783 is kept loadable for one purpose only: the Phase 1 reproduction of Shiu et al.
runs on the data Shiu et al. used, so a mismatch there is unambiguously our bug and not a
dataset difference.

---

## Known limitations

These are material and belong up front, not buried in comments. Several of them bear
directly on the headline result.

### Synapse count is a proxy for synaptic weight, not a measurement

Weights are synapse count multiplied by a sign. Nothing in the EM data measures synaptic
strength. Two synapses of equal count can differ substantially in physiological effect.
**Any claim sensitive to the exact weight scaling requires a sensitivity sweep before it is
made.**

### Gap junctions are absent, and the giant fiber depends on them

Electron microscopy does not resolve electrical synapses, so the connectome contains none.
The GF -> TTM junction is *mixed* - it has a large electrical component, and that component
carries the fastest part of the response.

This is the sharpest limitation in the project: **the fastest part of the circuit is the part
the connectome represents worst.** We add nothing to compensate. There is no hand-inserted
electrical coupling anywhere in this codebase, by deliberate policy. If the short-latency
escape mode fails to appear, that is a result about what the connectome alone predicts, and
it will be reported as such rather than tuned away.

### One animal, one individual, one sex

Not a population average. Every count and every weight comes from a single fly. MaleCNS is
male, while much of the published escape-circuit literature is FlyWire-based and therefore
female. Sexually dimorphic cell types exist, and comparisons across the two datasets carry
that caveat.

### No neuromodulation

Escape thresholds in real flies are state-dependent - hunger, walking versus flight, prior
looming exposure. A static graph cannot represent this, and the modulatory transmitters
(dopamine, octopamine, serotonin) are assigned sign 0, which removes those edges entirely.
Anything resembling state dependence in this simulation was added by us, by hand, and is
labelled as such.

### The optic lobes are loaded but silent

Version 1 uses an analytic looming encoder that computes angular size and expansion rate from
scene geometry and injects current directly at the LC and LPLC2 somata. That bypasses the
optic lobe entirely, so those ~50k neurons carry no activity and render as dim anatomical
context only. This is deliberate, documented, and the reason `SensoryEncoder` is a swappable
interface: an ommatidial encoder is a module swap, not a refactor.

---

## Conventions

**Weight matrix orientation.** `weights[post, pre]` - rows are postsynaptic, columns
presynaptic. This makes the per-timestep input current a single CSR matvec
(`I = weights @ spikes`) and makes "who drives this neuron" a single cheap row slice, which
is what the click-to-explain inspector needs.

**Neurotransmitter sign.** Acetylcholine `+1`; GABA, glutamate and histamine `-1`; dopamine,
octopamine and serotonin `0`. Glutamate is inhibitory because GluCl-alpha dominates at fly
central synapses. The full rationale and citations are in `data/neurotransmitters.py`, which
is kept deliberately small and is covered by a test against a known-inhibitory pathway.

## Citation

Work using this code must cite the underlying connectome datasets. See the MaleCNS release
notes and the FlyWire citation guidelines at <https://flywire.ai/guidelines>.

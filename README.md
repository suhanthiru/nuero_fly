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

Phase 0 (connectome loader), Phase 1 (LIF core, validated against the Shiu et al. reference
implementation), Phase 2 (analytic looming encoder and the l/|v| sweep), and the live demo:
the 3D viewer is now driven by simulated activity.

Phase 3 (MuJoCo arena, scripted predator, escape adjudication).

### Phase 3 result: escape timing works, escape direction does not

The exit criterion asked whether the fly escapes preferentially away from the threat. It
does not, and the sweep says why rather than just that.

* **Takeoff happens.** The circuit fires on essentially every approach, at every azimuth.
* **Direction is noise.** Mean error from "directly away" is 96 degrees with the hemisphere
  weighting off and 80 degrees with it on, against 90 degrees for chance. Headings scatter
  across ~345 degrees in both conditions.
* **The cause is the size of the decision signal.** The heading is read from the left/right
  giant fiber asymmetry, and there are only ~6 GF spikes per trial across both cells. The
  trial-to-trial spread of that asymmetry at a *fixed* azimuth is nearly as large as its
  spread *across* azimuths - a signal-to-noise ratio of about 1.5. Which giant fiber happens
  to receive more Poisson events decides the heading.
* **The hand-added weighting does work, and is still not enough.** With it on, the L/R
  asymmetry does correlate with sin(azimuth) at r = +0.55, so real directional information
  reaches the giant fibers. It is simply swamped at this spike count.
* **Escape success (47-58%) is geometry, not computation.** A fly jumping in a roughly fixed
  direction escapes frontal threats and is caught by rear ones, which is exactly the shape
  of the success-vs-azimuth curve.
* **A persistent left bias** shows up in the null: 4.3 left GF spikes against 2.8 right.
  That traces to the anatomical asymmetry Phase 0 found - 71 LC4 on the left against 55 on
  the right - which is more likely a proofreading difference between the two optic lobes
  than biology, and is a caution about any left/right claim from this dataset.

Reproduce with `scripts/escape_sweep.py`, `scripts/diagnose_heading.py` and
`scripts/plot_escape.py`.

### Running the demo

```
python run_demo.py --ratio 40 --gain 0.03      # simulate a trial and stream it
npm --prefix viz/frontend run dev              # then open the printed URL
```

The trial is simulated once at startup and played back at 20 Hz under an adjustable time
dilation. That is not a shortcut: a trial is 8,000 timesteps over 165k neurons and cannot be
produced at wall-clock speed, and a giant fiber escape lasts a few milliseconds, which at
real time would fall inside a single frame. `?t=350&view=neck` deep-links a moment.

Brightness is smoothed activity, not spikes - spike trains convolved with a 50 ms
exponential, which is what a calcium indicator does to the same signal. Full brightness
means a neuron firing at the reference rate, and the stream auto-ranges with the chosen
scale reported in the header, because activity spans more than two orders of magnitude
between the LC populations and the giant fiber.

### Phase 2 result: the escape mode split does not appear

The scientific target was the short/long escape mode split. It does not emerge, and the
sweep says something more specific than "it didn't work":

* **No long-mode escape at any encoder gain.** Every trial that produces a takeoff is
  giant-fiber mediated. In this model TTMn is driven essentially only through the GF and
  the GF-coupled interneurons, so there is no GF-independent route to the motor neuron and
  therefore no second mode to find.
* **At our first-choice gain the latency scaling is destroyed too.** GF first-spike time
  varies by 8 ms across an eightfold change in looming speed - it is set by our visual
  latency constant, not by the stimulus.
* **At 30-100x lower gain the published latency relationship does appear**: escape occurs
  progressively earlier before contact as looming slows, which is what an angular-size
  threshold predicts, and at the fastest looming the GF fires ~4 ms before contact.
* **But nothing in the data picks that gain.** It is a free parameter of ours, and the span
  between "saturated" and "silent" is about two orders of magnitude. The latency scaling is
  therefore conditional on our choice, not a prediction of the connectome.

The mechanism behind the saturation is measurable and is a limitation of the *neuron model*
rather than of the wiring: one LC4 spike delivers 13.9 mV to the giant fiber, which is
2.2 mV of membrane deflection against the 7 mV gap from rest to threshold. Roughly **3.2
coincident LC4 spikes fire the GF**. With 311 driven cells, any appreciable firing rate
saturates it. Shiu et al.'s parameterisation gives every neuron the same 20 ms membrane
time constant and the same 7 mV threshold, and the giant fiber is one of the largest
neurons in the animal. A single-compartment model with brain-wide uniform parameters cannot
represent a cell whose whole function is to integrate hundreds of inputs to a sharp
threshold near contact.

Reproduce with `scripts/looming_sweep.py` and `scripts/looming_gain_sweep.py`.

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

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

### Neuron-model ablation: the wiring is not what is limiting this

Phases 2 and 3 both ended by blaming the neuron model - saturation at nominal gain, and a
decision signal too small to carry a direction. `NeuronModel` was an interface from Phase 1
precisely so those could be tested, so they were: the same connectome, the same looming
task, four dynamics.

| arm | GF spikes (gain 1.0) | tracks l/\|v\| at gain 1.0 | at gain 0.03 |
|---|---|---|---|
| `lif-uniform` (Shiu et al.) | 153 | no (1.5 ms spread) | yes |
| `lif-capacitance` | 0 | silent | silent |
| `conductance` | 173 | no (1.6 ms spread) | yes |
| `conductance-cap` | 0 | silent | silent |
| `rate` | 50 | **yes** (219 ms spread) | silent |

Four things came out of it, and none of them is "model X wins".

**Conductance synapses do not fix the saturation.** This was the obvious physiological
hypothesis - reversal potentials bound the drive, so excitation should stop accumulating.
It does not help: 173 GF spikes against 153, and the escape *latency* agrees with the
current-based model to within 1.3 ms in every single condition tested. The extra biophysical
realism buys nothing on this task, which is a useful negative for anyone about to spend
effort on it.

**Capacitance scaling brackets the answer rather than settling it.** DNp01 has 15,484 input
synapses against a population median of 202. Under the uniform assumption that means ~3.4
coincident LC4 spikes fire the giant fiber; under full synapse-count normalisation it needs
~264, which is more than the LC populations can deliver, so it never fires. The real cell is
somewhere inside that 78-fold bracket and **nothing in the connectome says where** - synapse
count is a proxy for membrane area, and the relation between area and excitability is not
something an EM volume measures.

**Each model has its own operating window, and they do not overlap.** The rate model is the
only arm that behaves at the nominal encoder gain, and it is the only one that goes silent
at the reduced gain the demo defaults to. So the Phase 2 conclusion generalises: the latency
scaling is conditional not on the encoder gain alone but on the *joint* choice of gain and
neuron model, and the two trade off against each other. There is no setting at which all
four agree, and no data here to choose between them.

**No arm produces the mode split.** Every escape in every model is giant-fiber mediated,
which confirms the Phase 3 structural finding: in this connectome TTMn is reachable
essentially only through the GF and its coupled interneurons, so there is no second route
for a neuron model to find. That result is about the wiring, and it is the one conclusion
here that survives changing the dynamics.

Reproduce with `scripts/model_ablation.py`, `scripts/compare_models.py` and
`scripts/diagnose_capacitance.py`.

### Running the demo

```
python run_demo.py --ratio 20 --gain 0.03 --azimuth 35   # simulate a trial and stream it
npm --prefix viz/frontend run dev                        # then open the printed URL
```

Two cameras on one simulation, switched with **w** (world) and **b** (brain) - step N in the
arena is step N of the spike train. The brain view shows the escape pathway lighting up; the
world view shows the fly, the floor, and the predator closing, with the takeoff and the
escape arc as they actually came out of the physics. **f** toggles the camera following the
fly. `?mode=world&t=620` deep-links a moment in either view.

The fly is the [flybody](https://github.com/google-deepmind/mujoco_menagerie/tree/main/flybody)
model (Vaxenburg et al., *Nature* 2025; Apache-2.0), baked into a single posed mesh by
`scripts/bake_fly_mesh.py` after `scripts/fetch_flybody.sh`. It is drawn at true scale -
2.5 mm body, 5 mm leg span - against a 20 mm stimulus, so it really is a speck.

**The body is a shell.** Its legs and wings are frozen in the model's resting pose and are
never actuated: the escape is still a rigid-body impulse, exactly as in Phase 3. Rendering
an articulated fly whose joints never move risks implying leg mechanics that are not
simulated, so the HUD says so. Driving those joints for real is the NeuroMechFly-class work
the build spec deferred, and the blocker is not the body but the controller - the decoder
produces one heading and one takeoff time, while a 102-DOF fly needs commands for every leg
joint.

The predator is a flat **disc** rather than a solid, because that is the stimulus the
behavioural literature actually presents - a dark disc expanding on a screen - and it is the
shape the encoder's angular-size geometry assumes. Its straight-line, constant-velocity
approach is likewise deliberate: `l/|v|` is only defined for constant velocity, so a
pursuing predator would invalidate the Phase 2 comparison to published latency data.

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

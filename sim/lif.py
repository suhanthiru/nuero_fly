"""Leaky integrate-and-fire, in JAX, over a sparse connectome.

This is a deliberate reimplementation of the model in Shiu, Sterne et al., whose Brian2
reference lives at https://github.com/philshiu/Drosophila_brain_model. The equations,
constants and per-timestep ordering below are theirs; reproducing their published
sugar -> proboscis result is the gate that licenses every later number this project
produces, and that reproduction is only meaningful if the model is genuinely the same one.

Their formulation, verbatim from ``model.py``::

    dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau              : volt (unless refractory)
    threshold: v > v_th
    reset:     v = v_rst; g = 0 * mV
    synapse:   on_pre 'g += w', delay t_dly

Note the shapes of this that are easy to get wrong:

* ``g`` has units of voltage and drives ``v`` towards ``v_0 + g``. It is not a current.
* ``g`` is **zeroed on spike**, not merely decayed.
* Both states are frozen during the refractory period - ``unless refractory`` applies to
  each differential equation, so a refractory neuron neither leaks nor integrates.
* **Synaptic input arriving at a refractory neuron is discarded, not banked.** Brian2
  rejects writes to a variable carrying ``(unless refractory)`` for the whole refractory
  window, so those events are lost rather than applied on recovery. This is not what the
  equations suggest on a plain reading, and implementing the plain reading inflates every
  downstream firing rate by ~29%, in proportion to how long each neuron spends refractory.
  No subthreshold test catches it, because a neuron that never spikes is never refractory.
  Measured directly in ``scripts/probe_refractory.py``; pinned by a test.
* Poisson stimulation is applied directly to ``v``, not through ``g``, at a weight of
  ``w_syn * f_poi`` = 68.75 mV. That is ten times the 7 mV gap between rest and threshold,
  so a stimulated neuron fires on essentially every Poisson event. Stimulated neurons also
  have their refractory period set to zero.

Units are millivolts and milliseconds throughout.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from .neuron import NeuronModel, SimulationResult, StimulusSpec


@dataclass(frozen=True)
class LIFParams:
    """Constants from Shiu et al. Sources are cited per line in their ``model.py``.

    Every value here is theirs. None of it is tuned by us, and none of it should be: the
    entire point of the Phase 1 gate is that the model is not ours to adjust.
    """

    # Kakaria and de Bivort 2017, https://doi.org/10.3389/fnbeh.2017.00008
    v_rest: float = -52.0        # mV
    v_reset: float = -52.0       # mV
    v_threshold: float = -45.0   # mV
    tau_membrane: float = 20.0   # ms

    # Jürgensen et al, https://doi.org/10.1088/2634-4386/ac3ba6
    tau_synapse: float = 5.0     # ms

    # Lazar et al, https://doi.org/10.7554/eLife.62362
    t_refractory: float = 2.2    # ms

    # Paul et al 2015, doi: 10.3389/fncel.2015.00029
    t_delay: float = 1.8         # ms

    # Free parameter in the original model.
    w_synapse: float = 0.275     # mV per synapse

    poisson_rate_hz: float = 150.0
    poisson_scale: float = 250.0  # stimulation weight = w_synapse * poisson_scale

    dt: float = 0.1              # ms

    @property
    def delay_steps(self) -> int:
        return int(round(self.t_delay / self.dt))

    @property
    def refractory_steps(self) -> int:
        return int(round(self.t_refractory / self.dt))

    @property
    def poisson_weight(self) -> float:
        return self.w_synapse * self.poisson_scale


def _decay_coefficients(params: LIFParams) -> tuple[float, float, float]:
    """Exact-integration coefficients for one timestep.

    Brian2 integrates this system with ``method='linear'``, i.e. exactly rather than by
    Euler steps, so we do the same - at dt = 0.1 ms against tau = 5 ms, forward Euler would
    drift enough to matter over a 10,000-step trial.

    With ``a = 1/tau_membrane`` and ``b = 1/tau_synapse``, and g decaying as
    ``g(t) = g0 exp(-b t)``, the membrane equation has the closed-form solution::

        v(t) = v_rest + (v0 - v_rest - C) exp(-a t) + C exp(-b t),   C = a g0 / (a - b)

    Returns ``(exp(-a dt), exp(-b dt), a / (a - b))``.
    """
    a = 1.0 / params.tau_membrane
    b = 1.0 / params.tau_synapse
    if abs(a - b) < 1e-12:
        raise ValueError(
            "tau_membrane and tau_synapse must differ; the closed-form solution is "
            "degenerate when they are equal"
        )
    return float(np.exp(-a * params.dt)), float(np.exp(-b * params.dt)), float(a / (a - b))


def _csr_to_coo_arrays(weights: sp.csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten a ``[post, pre]`` CSR matrix into (post, pre, weight) triples.

    ``post`` comes out sorted ascending because that is CSR row order, which lets the
    per-step accumulation use a sorted segment_sum instead of scatter-add with atomics.
    """
    matrix = weights.tocsr()
    counts = np.diff(matrix.indptr)
    post = np.repeat(np.arange(matrix.shape[0], dtype=np.int32), counts)
    return post, matrix.indices.astype(np.int32), matrix.data.astype(np.float32)


class LIF(NeuronModel):
    """Sparse leaky integrate-and-fire, vectorised over neurons and scanned over time."""

    name = "lif"

    def __init__(self, params: LIFParams | None = None) -> None:
        self.params = params or LIFParams()

    # -- the per-timestep update ---------------------------------------------------
    # n_neurons, n_record, n_steps and delay_steps are static: they set array shapes and
    # the scan length, so they have to be known at trace time.
    @staticmethod
    @partial(jax.jit, static_argnums=(6, 7, 8, 16))
    def _run(
        post_idx,
        pre_idx,
        weight,
        stim_idx,
        silence_mask,
        record_idx,
        n_neurons: int,
        n_record: int,
        n_steps,
        decay_v,
        decay_g,
        coupling,
        v_rest,
        v_reset,
        v_threshold,
        refractory_steps,
        delay_steps,
        poisson_weight,
        forced_idx,
        forced,
        schedule,
        key,
    ):
        def step(carry, inputs):
            forced_now, rates_now = inputs
            v, g, refractory, buffer, slot, counts, rng = carry

            active = refractory <= 0

            # 1. Integrate, but only where not refractory. Brian2's `(unless refractory)`
            #    freezes both states, so a refractory neuron neither leaks nor integrates.
            c = coupling * g
            v_next = v_rest + (v - v_rest - c) * decay_v + c * decay_g
            g_next = g * decay_g
            v = jnp.where(active, v_next, v)
            g = jnp.where(active, g_next, g)

            # 2. Threshold. Refractory neurons cannot spike; silenced ones never can.
            forced = jnp.zeros(n_neurons, dtype=bool).at[forced_idx].set(forced_now)
            spike = (active & (v > v_threshold) | forced) & ~silence_mask

            # 3. Synaptic delivery of spikes emitted `delay_steps` ago. Reading the ring
            #    buffer slot before overwriting it is what makes the delay exact.
            #
            #    Input arriving at a refractory neuron is DISCARDED, not banked. This is not
            #    obvious from the equations and it is not what a careful reimplementation
            #    naturally does, but it is what Brian2 does: a variable carrying
            #    `(unless refractory)` rejects synaptic writes for the whole refractory
            #    window, verified directly in scripts/probe_refractory.py. Applying that
            #    input instead - the obvious reading - inflates every downstream firing rate
            #    by ~29%, in proportion to how much time each neuron spends refractory. It is
            #    invisible in any subthreshold test, because a neuron that never spikes is
            #    never refractory.
            delayed = buffer[slot]
            contribution = weight * delayed[pre_idx].astype(weight.dtype)
            arriving = jax.ops.segment_sum(
                contribution, post_idx, num_segments=n_neurons, indices_are_sorted=True
            )
            g = g + jnp.where(active, arriving, 0.0)

            # 4. Poisson drive, applied straight to v as in the reference, and before the
            #    reset - so a stimulated neuron that just spiked has the boost erased,
            #    exactly as Brian2's synapses-before-resets ordering does.
            #
            #    The rate is per-neuron and per-timestep, which is what lets a sensory
            #    encoder drive each cell at its own time-varying rate. A constant stimulus
            #    is just a schedule that does not vary.
            rng, subkey = jax.random.split(rng)
            events = jax.random.bernoulli(subkey, rates_now)
            v = v.at[stim_idx].add(poisson_weight * events.astype(v.dtype))

            # 5. Reset.
            v = jnp.where(spike, v_reset, v)
            g = jnp.where(spike, 0.0, g)
            refractory = jnp.where(spike, refractory_steps, jnp.maximum(refractory - 1, 0))

            buffer = buffer.at[slot].set(spike)
            slot = (slot + 1) % delay_steps

            carry = (v, g, refractory, buffer, slot, counts + spike, rng)
            return carry, (spike[record_idx], v[record_idx])

        init = (
            jnp.full(n_neurons, v_rest, dtype=jnp.float32),
            jnp.zeros(n_neurons, dtype=jnp.float32),
            jnp.zeros(n_neurons, dtype=jnp.int32),
            jnp.zeros((delay_steps, n_neurons), dtype=bool),
            jnp.int32(0),
            jnp.zeros(n_neurons, dtype=jnp.int32),
            key,
        )
        final, (raster, voltage) = jax.lax.scan(
            step, init, (forced, schedule), length=n_steps
        )
        return final[5], raster, voltage

    # -- public interface ----------------------------------------------------------
    def simulate(
        self,
        weights: sp.csr_matrix,
        stimulus: StimulusSpec,
        *,
        duration_ms: float,
        seed: int,
        record: np.ndarray | None = None,
        forced_spikes: dict[int, np.ndarray] | None = None,
    ) -> SimulationResult:
        params = self.params
        n_neurons = weights.shape[0]
        n_steps = int(round(duration_ms / params.dt))

        post_idx, pre_idx, synapse_count = _csr_to_coo_arrays(weights)
        weight = synapse_count * params.w_synapse

        stim_idx = np.asarray(stimulus.poisson_targets, dtype=np.int32).reshape(-1)
        stim_mask = np.zeros(n_neurons, dtype=bool)
        stim_mask[stim_idx.astype(np.int64)] = True

        silence_mask = np.zeros(n_neurons, dtype=bool)
        if stimulus.silenced is not None and len(stimulus.silenced):
            silence_mask[np.asarray(stimulus.silenced, dtype=np.int64)] = True

        record_idx = (
            np.asarray(record, dtype=np.int32)
            if record is not None
            else np.zeros(0, dtype=np.int32)
        )

        # The reference sets the refractory period to zero for Poisson targets, so a
        # stimulated neuron can follow its drive without being gated by its own last spike.
        refractory_steps = np.where(stim_mask, 0, params.refractory_steps).astype(np.int32)

        # Exact externally-imposed spike times, stored against their own index list rather
        # than dense over every neuron: a dense (n_steps, n_neurons) array is 1.3 GB of
        # mostly-zeros on a whole-brain graph.
        forced_idx = np.asarray(sorted(forced_spikes or {}), dtype=np.int32)
        forced = np.zeros((n_steps, forced_idx.size), dtype=bool)
        for column, neuron in enumerate(forced_idx):
            steps = np.round(np.asarray((forced_spikes or {})[int(neuron)]) / params.dt)
            steps = steps.astype(int)
            forced[steps[(steps >= 0) & (steps < n_steps)], column] = True

        decay_v, decay_g, coupling = _decay_coefficients(params)

        # Per-neuron, per-step Bernoulli probability. A constant stimulus is a flat
        # schedule; a sensory encoder supplies a time-varying one.
        if stimulus.rate_schedule is not None:
            schedule_hz = np.asarray(stimulus.rate_schedule, dtype=np.float32)
            if schedule_hz.shape != (n_steps, stim_idx.size):
                raise ValueError(
                    f"rate_schedule has shape {schedule_hz.shape}, expected "
                    f"{(n_steps, stim_idx.size)} to match duration and poisson_targets"
                )
        else:
            rate_hz = (
                stimulus.rate_hz if stimulus.rate_hz is not None else params.poisson_rate_hz
            )
            schedule_hz = np.full((n_steps, stim_idx.size), rate_hz, dtype=np.float32)
        schedule = schedule_hz * (params.dt / 1000.0)

        counts, raster, voltage = self._run(
            jnp.asarray(post_idx),
            jnp.asarray(pre_idx),
            jnp.asarray(weight),
            jnp.asarray(stim_idx),
            jnp.asarray(silence_mask),
            jnp.asarray(record_idx),
            n_neurons,
            int(record_idx.size),
            n_steps,
            jnp.float32(decay_v),
            jnp.float32(decay_g),
            jnp.float32(coupling),
            jnp.float32(params.v_rest),
            jnp.float32(params.v_reset),
            jnp.float32(params.v_threshold),
            jnp.asarray(refractory_steps, dtype=jnp.int32),
            params.delay_steps,
            jnp.float32(params.poisson_weight),
            jnp.asarray(forced_idx),
            jnp.asarray(forced),
            jnp.asarray(schedule),
            jax.random.PRNGKey(seed),
        )

        counts = np.asarray(counts)
        spike_times: dict[int, np.ndarray] = {}
        voltages: dict[int, np.ndarray] = {}
        if record_idx.size:
            raster = np.asarray(raster)      # (steps, n_record)
            voltage = np.asarray(voltage)    # (steps, n_record)
            for column, neuron in enumerate(record_idx):
                steps = np.flatnonzero(raster[:, column])
                spike_times[int(neuron)] = steps * params.dt
                voltages[int(neuron)] = voltage[:, column]

        return SimulationResult(
            spike_counts=counts,
            duration_ms=duration_ms,
            dt_ms=params.dt,
            spike_times=spike_times,
            voltages=voltages,
            raster=np.asarray(raster) if record_idx.size else np.zeros((0, 0), dtype=bool),
            record_indices=record_idx,
            meta={
                "model": self.name,
                "params": params.__dict__,
                "seed": seed,
                "n_neurons": n_neurons,
                "n_synapses": int(weight.size),
                "n_stimulated": int(stim_mask.sum()),
                "n_silenced": int(silence_mask.sum()),
                "rate_hz": None if stimulus.rate_schedule is not None else float(schedule_hz[0, 0]) if schedule_hz.size else None,
                "rate_schedule": stimulus.rate_schedule is not None,
            },
        )

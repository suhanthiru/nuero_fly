"""Conductance-based integrate-and-fire.

The same wiring, the same threshold, but synapses act as conductances with reversal
potentials rather than as fixed voltage injections::

    dv/dt = [ (v_rest - v) + g_e (E_e - v) + g_i (E_i - v) ] / tau_m
    dg/dt = -g / tau_syn

The difference that matters is the driving force. In the current-based model an excitatory
spike adds the same voltage no matter how depolarised the cell already is, so drive
accumulates without limit - which is exactly the saturation Phase 2 measured, where the
giant fiber fired ~150 times per trial because nothing bounded its input. Here the term
``g_e (E_e - v)`` shrinks as ``v`` climbs toward the excitatory reversal potential, so a cell
being hammered by hundreds of synapses approaches ``E_e`` and stops being driven harder.
Inhibition likewise shunts rather than subtracting without bound.

This is the physiologically standard formulation and it is what a real neuron does. It is
included here as an ablation rather than as a correction: the Phase 1 reproduction was
validated against the current-based model, so any result that changes under this one is a
result about the neuron model, not about the connectome.

Integration is exponential Euler - conductances are held constant across a step, which makes
the membrane equation linear over that step and solvable exactly. That is the usual scheme
for conductance-based models and is second-order accurate in the conductance dynamics at
dt = 0.1 ms against tau_syn = 5 ms.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from .lif import LIFParams, _capacitance_scale, _csr_to_coo_arrays
from .neuron import NeuronModel, SimulationResult, StimulusSpec


@dataclass(frozen=True)
class ConductanceParams(LIFParams):
    """LIF constants plus reversal potentials.

    Reversals are the standard ones for fly central synapses: acetylcholine gates a cation
    channel reversing near 0 mV, while GABA and GluCl gate chloride reversing near or just
    below rest. The inhibitory reversal being close to rest is why inhibition here mostly
    *shunts* - it divides the excitatory drive rather than subtracting from it.
    """

    excitatory_reversal_mv: float = 0.0
    inhibitory_reversal_mv: float = -80.0

    #: Conductance added per synapse, in units of the leak conductance. Set so that a
    #: single synapse has an effect comparable to the current-based model's 0.275 mV at
    #: rest, keeping the two ablation arms on the same footing at low drive. Ours, not
    #: theirs - the current-based model has no conductance to translate.
    g_per_synapse: float = 0.0055


class ConductanceLIF(NeuronModel):
    """Integrate-and-fire with conductance synapses and reversal potentials."""

    name = "conductance"

    def __init__(self, params: ConductanceParams | None = None) -> None:
        self.params = params or ConductanceParams()

    @staticmethod
    @partial(jax.jit, static_argnums=(6, 7, 8, 14))
    def _run(
        post_idx, pre_idx, weight, stim_idx, silence_mask, record_idx,
        n_neurons: int, n_record: int, n_steps,
        decay_g, v_rest, v_reset, v_threshold, refractory_steps, delay_steps,
        e_exc, e_inh, dt, tau_m, poisson_weight, forced_idx, forced, schedule, key,
    ):
        def step(carry, inputs):
            forced_now, rates_now = inputs
            v, ge, gi, refractory, buffer, slot, counts, rng = carry

            active = refractory <= 0

            # Exponential Euler: hold conductances constant across the step, which makes the
            # membrane equation linear and solvable exactly over that interval.
            total = 1.0 + ge + gi
            v_inf = (v_rest + ge * e_exc + gi * e_inh) / total
            decay_v = jnp.exp(-dt * total / tau_m)
            v_next = v_inf + (v - v_inf) * decay_v

            v = jnp.where(active, v_next, v)
            ge = jnp.where(active, ge * decay_g, ge)
            gi = jnp.where(active, gi * decay_g, gi)

            forced = jnp.zeros(n_neurons, dtype=bool).at[forced_idx].set(forced_now)
            spike = (active & (v > v_threshold) | forced) & ~silence_mask

            # Synaptic delivery, discarded at refractory neurons exactly as in the
            # current-based model - see the note in sim/lif.py, this is Brian2's behaviour
            # for a state variable carrying `(unless refractory)` and it is worth ~29%.
            delayed = buffer[slot]
            contribution = weight * delayed[pre_idx].astype(weight.dtype)
            arriving = jax.ops.segment_sum(
                contribution, post_idx, num_segments=n_neurons, indices_are_sorted=True
            )
            ge = ge + jnp.where(active, jnp.maximum(arriving, 0.0), 0.0)
            gi = gi + jnp.where(active, jnp.maximum(-arriving, 0.0), 0.0)

            rng, subkey = jax.random.split(rng)
            events = jax.random.bernoulli(subkey, rates_now)
            v = v.at[stim_idx].add(poisson_weight * events.astype(v.dtype))

            v = jnp.where(spike, v_reset, v)
            ge = jnp.where(spike, 0.0, ge)
            gi = jnp.where(spike, 0.0, gi)
            refractory = jnp.where(spike, refractory_steps, jnp.maximum(refractory - 1, 0))

            buffer = buffer.at[slot].set(spike)
            slot = (slot + 1) % delay_steps

            carry = (v, ge, gi, refractory, buffer, slot, counts + spike, rng)
            return carry, (spike[record_idx], v[record_idx])

        init = (
            jnp.full(n_neurons, v_rest, dtype=jnp.float32),
            jnp.zeros(n_neurons, dtype=jnp.float32),
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
        return final[6], raster, voltage

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
        # Signed conductance: positive entries become excitatory conductance, negative
        # inhibitory. Sign is split inside the step so one array carries both.
        weight = synapse_count * params.g_per_synapse
        weight = weight / _capacitance_scale(params, weights)[post_idx]

        stim_idx = np.asarray(stimulus.poisson_targets, dtype=np.int32).reshape(-1)
        stim_mask = np.zeros(n_neurons, dtype=bool)
        stim_mask[stim_idx.astype(np.int64)] = True

        silence_mask = np.zeros(n_neurons, dtype=bool)
        if stimulus.silenced is not None and len(stimulus.silenced):
            silence_mask[np.asarray(stimulus.silenced, dtype=np.int64)] = True

        record_idx = (
            np.asarray(record, dtype=np.int32) if record is not None
            else np.zeros(0, dtype=np.int32)
        )
        refractory_steps = np.where(stim_mask, 0, params.refractory_steps).astype(np.int32)

        forced_idx = np.asarray(sorted(forced_spikes or {}), dtype=np.int32)
        forced = np.zeros((n_steps, forced_idx.size), dtype=bool)
        for column, neuron in enumerate(forced_idx):
            steps = np.round(
                np.asarray((forced_spikes or {})[int(neuron)]) / params.dt
            ).astype(int)
            forced[steps[(steps >= 0) & (steps < n_steps)], column] = True

        if stimulus.rate_schedule is not None:
            schedule_hz = np.asarray(stimulus.rate_schedule, dtype=np.float32)
        else:
            rate = (
                stimulus.rate_hz if stimulus.rate_hz is not None else params.poisson_rate_hz
            )
            schedule_hz = np.full((n_steps, stim_idx.size), rate, dtype=np.float32)
        schedule = schedule_hz * (params.dt / 1000.0)

        counts, raster, voltage = self._run(
            jnp.asarray(post_idx), jnp.asarray(pre_idx), jnp.asarray(weight),
            jnp.asarray(stim_idx), jnp.asarray(silence_mask), jnp.asarray(record_idx),
            n_neurons, int(record_idx.size), n_steps,
            jnp.float32(np.exp(-params.dt / params.tau_synapse)),
            jnp.float32(params.v_rest), jnp.float32(params.v_reset),
            jnp.float32(params.v_threshold),
            jnp.asarray(refractory_steps, dtype=jnp.int32), params.delay_steps,
            jnp.float32(params.excitatory_reversal_mv),
            jnp.float32(params.inhibitory_reversal_mv),
            jnp.float32(params.dt), jnp.float32(params.tau_membrane),
            jnp.float32(params.poisson_weight),
            jnp.asarray(forced_idx), jnp.asarray(forced), jnp.asarray(schedule),
            jax.random.PRNGKey(seed),
        )

        counts = np.asarray(counts)
        spike_times: dict[int, np.ndarray] = {}
        voltages: dict[int, np.ndarray] = {}
        if record_idx.size:
            raster = np.asarray(raster)
            voltage = np.asarray(voltage)
            for column, neuron in enumerate(record_idx):
                spike_times[int(neuron)] = np.flatnonzero(raster[:, column]) * params.dt
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
                "capacitance_mode": params.capacitance_mode,
            },
        )

"""Firing-rate model: no spikes, no threshold, no refractory period.

    tau dr/dt = -r + f(W r + input)

with ``f`` a rectified linear transfer saturating at a maximum rate. This is the coarsest
model in the ablation and it is here to answer one question: **does spiking matter for this
task at all?** If the rate model reproduces the same escape timing and the same dependence
on looming speed, then everything the spiking models compute is available from mean rates
alone, and the spike-level machinery is not earning its cost. If it does not, the difference
localises what spiking contributes.

It is also roughly two orders of magnitude cheaper, which matters for the lesion sweep.

**Spike times from rates.** The decoder and the escape adjudication both need spike times,
so rates are converted deterministically: each neuron accumulates ``r * dt`` and emits a
spike each time that accumulator passes 1, then subtracts 1. This is the standard rate-to-
spike conversion, it preserves the rate exactly, and it introduces no randomness - which is
worth noting, because the Phase 3 finding was that trial-to-trial *Poisson* variability in a
~6-spike decision signal is what destroyed the escape heading. This model has no such
variability by construction, so it is the natural test of whether that diagnosis was right.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

from .lif import _capacitance_scale, _csr_to_coo_arrays
from .neuron import NeuronModel, SimulationResult, StimulusSpec


@dataclass(frozen=True)
class RateParams:
    """Transfer function and time constant.

    The gain and threshold are chosen so the input-output relation matches the spiking
    model's in the low-drive regime: the LIF needs roughly 7 mV of summed deflection to fire,
    and a synapse delivers 0.275 mV, so the threshold sits at the equivalent summed synapse
    count. Ours, not from the literature.
    """

    tau_ms: float = 20.0
    dt: float = 0.1
    #: Summed signed synaptic input required before a neuron responds at all.
    threshold: float = 25.0
    #: Output rate per unit of input above threshold.
    gain_hz: float = 3.0
    max_rate_hz: float = 400.0
    #: Delay on synaptic transmission, matching the spiking models.
    t_delay: float = 1.8
    capacitance_mode: str = "uniform"

    @property
    def delay_steps(self) -> int:
        return int(round(self.t_delay / self.dt))


class RateModel(NeuronModel):
    """Threshold-linear rate dynamics over the same connectome."""

    name = "rate"

    def __init__(self, params: RateParams | None = None) -> None:
        self.params = params or RateParams()

    @staticmethod
    @partial(jax.jit, static_argnums=(5, 6, 7, 11))
    def _run(
        post_idx, pre_idx, weight, stim_idx, record_idx,
        n_neurons: int, n_record: int, n_steps,
        decay, threshold, gain, delay_steps, max_rate, dt_s, silence_mask, schedule,
    ):
        def step(carry, drive_now):
            rate, buffer, slot, accumulator, counts = carry

            delayed = buffer[slot]
            contribution = weight * delayed[pre_idx]
            recurrent = jax.ops.segment_sum(
                contribution, post_idx, num_segments=n_neurons, indices_are_sorted=True
            )
            external = jnp.zeros(n_neurons).at[stim_idx].add(drive_now)

            target = jnp.clip(gain * (recurrent + external - threshold), 0.0, max_rate)
            target = jnp.where(silence_mask, 0.0, target)
            rate = rate + (target - rate) * decay

            # Deterministic rate-to-spike conversion: accumulate and emit on each unit.
            accumulator = accumulator + rate * dt_s
            spike = accumulator >= 1.0
            accumulator = accumulator - spike.astype(accumulator.dtype)

            buffer = buffer.at[slot].set(rate)
            slot = (slot + 1) % delay_steps

            carry = (rate, buffer, slot, accumulator, counts + spike)
            return carry, (spike[record_idx], rate[record_idx])

        init = (
            jnp.zeros(n_neurons, dtype=jnp.float32),
            jnp.zeros((delay_steps, n_neurons), dtype=jnp.float32),
            jnp.int32(0),
            jnp.zeros(n_neurons, dtype=jnp.float32),
            jnp.zeros(n_neurons, dtype=jnp.int32),
        )
        final, (raster, rates) = jax.lax.scan(step, init, schedule, length=n_steps)
        return final[4], raster, rates

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
        # Rates are in Hz and weights in synapses, so scale the recurrent term to keep the
        # threshold in the same synapse-count units the transfer function expects.
        weight = synapse_count / max(params.max_rate_hz, 1e-9)
        weight = weight / _capacitance_scale(params, weights)[post_idx]

        stim_idx = np.asarray(stimulus.poisson_targets, dtype=np.int32).reshape(-1)
        silence_mask = np.zeros(n_neurons, dtype=bool)
        if stimulus.silenced is not None and len(stimulus.silenced):
            silence_mask[np.asarray(stimulus.silenced, dtype=np.int64)] = True

        record_idx = (
            np.asarray(record, dtype=np.int32) if record is not None
            else np.zeros(0, dtype=np.int32)
        )

        # External drive enters as an input current in the same units as the threshold, so a
        # target rate of R Hz corresponds to the drive that would produce it.
        if stimulus.rate_schedule is not None:
            schedule_hz = np.asarray(stimulus.rate_schedule, dtype=np.float32)
        else:
            rate = stimulus.rate_hz if stimulus.rate_hz is not None else 150.0
            schedule_hz = np.full((n_steps, stim_idx.size), rate, dtype=np.float32)
        drive = schedule_hz / max(params.gain_hz, 1e-9) + params.threshold

        counts, raster, rates = self._run(
            jnp.asarray(post_idx), jnp.asarray(pre_idx), jnp.asarray(weight),
            jnp.asarray(stim_idx), jnp.asarray(record_idx),
            n_neurons, int(record_idx.size), n_steps,
            jnp.float32(params.dt / params.tau_ms),
            jnp.float32(params.threshold), jnp.float32(params.gain_hz),
            params.delay_steps, jnp.float32(params.max_rate_hz),
            jnp.float32(params.dt / 1000.0),
            jnp.asarray(silence_mask), jnp.asarray(drive),
        )

        counts = np.asarray(counts)
        spike_times: dict[int, np.ndarray] = {}
        traces: dict[int, np.ndarray] = {}
        if record_idx.size:
            raster = np.asarray(raster)
            rates = np.asarray(rates)
            for column, neuron in enumerate(record_idx):
                spike_times[int(neuron)] = np.flatnonzero(raster[:, column]) * params.dt
                traces[int(neuron)] = rates[:, column]

        return SimulationResult(
            spike_counts=counts,
            duration_ms=duration_ms,
            dt_ms=params.dt,
            spike_times=spike_times,
            # Rates, not membrane voltages - there is no membrane in this model.
            voltages=traces,
            raster=np.asarray(raster) if record_idx.size else np.zeros((0, 0), dtype=bool),
            record_indices=record_idx,
            meta={
                "model": self.name,
                "params": params.__dict__,
                "seed": seed,
                "n_neurons": n_neurons,
                "note": "voltages field carries firing rates in Hz, not millivolts",
            },
        )

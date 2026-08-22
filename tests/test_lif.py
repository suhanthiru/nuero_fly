"""LIF numerics.

The integrator is checked against a closed-form/numerical solution rather than against
itself, and the sign convention is checked at the level of dynamics rather than weights -
an inverted transmitter table would still produce a network that spikes plausibly, so it
has to be caught by asking whether inhibition actually inhibits.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from scipy.integrate import solve_ivp

from sim.lif import LIF, LIFParams, _decay_coefficients
from sim.neuron import StimulusSpec


def graph(n: int, edges: dict[tuple[int, int], float]) -> sp.csr_matrix:
    """Build a tiny ``[post, pre]`` matrix. Keys are ``(pre, post)`` synapse counts."""
    matrix = sp.lil_matrix((n, n), dtype=np.float32)
    for (pre, post), count in edges.items():
        matrix[post, pre] = count
    return matrix.tocsr()


def run(weights, stim, *, duration=200.0, seed=0, record=None, silenced=None, rate=None):
    return LIF().simulate(
        weights,
        StimulusSpec(
            poisson_targets=np.asarray(stim),
            silenced=None if silenced is None else np.asarray(silenced),
            rate_hz=rate,
        ),
        duration_ms=duration,
        seed=seed,
        record=None if record is None else np.asarray(record),
    )


class TestIntegrator:
    def test_one_step_matches_numerical_ode(self):
        """The exact-integration coefficients against a high-accuracy ODE solve."""
        params = LIFParams()
        decay_v, decay_g, coupling = _decay_coefficients(params)
        v0, g0 = -50.0, 3.0

        def rhs(_t, y):
            v, g = y
            return [
                (params.v_rest - v + g) / params.tau_membrane,
                -g / params.tau_synapse,
            ]

        solution = solve_ivp(rhs, (0.0, params.dt), [v0, g0], rtol=1e-12, atol=1e-14)
        v_numeric, g_numeric = solution.y[:, -1]

        c = coupling * g0
        v_exact = params.v_rest + (v0 - params.v_rest - c) * decay_v + c * decay_g
        assert v_exact == pytest.approx(v_numeric, abs=1e-9)
        assert g0 * decay_g == pytest.approx(g_numeric, abs=1e-9)

    def test_error_does_not_compound_over_many_steps(self):
        """200 successive updates against one 20 ms solve.

        Forward Euler would visibly drift here; exact integration must not.
        """
        params = LIFParams()
        decay_v, decay_g, coupling = _decay_coefficients(params)
        v, g = -50.0, 5.0
        for _ in range(200):
            c = coupling * g
            v = params.v_rest + (v - params.v_rest - c) * decay_v + c * decay_g
            g = g * decay_g

        def rhs(_t, y):
            return [
                (params.v_rest - y[0] + y[1]) / params.tau_membrane,
                -y[1] / params.tau_synapse,
            ]

        solution = solve_ivp(rhs, (0.0, 20.0), [-50.0, 5.0], rtol=1e-12, atol=1e-14)
        assert v == pytest.approx(solution.y[0, -1], abs=1e-7)
        assert g == pytest.approx(solution.y[1, -1], abs=1e-7)

    def test_equal_time_constants_are_rejected(self):
        with pytest.raises(ValueError, match="degenerate"):
            _decay_coefficients(LIFParams(tau_membrane=5.0, tau_synapse=5.0))


class TestQuiescence:
    def test_unstimulated_network_is_silent(self):
        result = run(graph(3, {(0, 1): 100.0}), stim=[])
        assert result.spike_counts.sum() == 0

    def test_isolated_neuron_sits_exactly_at_rest(self):
        result = run(graph(2, {}), stim=[], record=[0])
        trace = result.voltages[0]
        assert np.allclose(trace, LIFParams().v_rest)


class TestSynapticSign:
    def test_excitation_drives_the_postsynaptic_cell(self):
        result = run(graph(2, {(0, 1): 100.0}), stim=[0])
        assert result.spike_counts[0] > 0
        assert result.spike_counts[1] > 0

    def test_inhibition_suppresses_it(self):
        """The dynamics-level sign check.

        Neuron 2 is driven by neuron 0 and inhibited by neuron 1. Adding the inhibitory
        drive must reduce neuron 2's output. If the transmitter sign convention were
        inverted somewhere upstream, this is the test that fails.
        """
        excited = graph(3, {(0, 2): 60.0})
        inhibited = graph(3, {(0, 2): 60.0, (1, 2): -60.0})

        without = run(excited, stim=[0, 1], duration=500.0).spike_counts[2]
        with_inhibition = run(inhibited, stim=[0, 1], duration=500.0).spike_counts[2]

        assert without > 0, "excitatory control produced no spikes to suppress"
        assert with_inhibition < without

    def test_inhibition_alone_produces_nothing(self):
        result = run(graph(2, {(0, 1): -100.0}), stim=[0], duration=500.0)
        assert result.spike_counts[1] == 0


class TestRefractoryAndDelay:
    def test_refractory_period_caps_the_firing_rate(self):
        """A non-stimulated neuron cannot exceed 1000 / t_refractory spikes per second."""
        params = LIFParams()
        result = run(graph(2, {(0, 1): 400.0}), stim=[0], duration=1000.0, rate=2000.0)
        ceiling = 1000.0 / params.t_refractory
        assert result.rates_hz()[1] <= ceiling + 1e-6

    def test_stimulated_neurons_have_no_refractory_period(self):
        """The reference sets rfc = 0 for Poisson targets, so they can outrun the cap."""
        params = LIFParams()
        result = run(graph(1, {}), stim=[0], duration=1000.0, rate=2000.0)
        assert result.rates_hz()[0] > 1000.0 / params.t_refractory

    def test_postsynaptic_response_waits_for_the_delay(self):
        params = LIFParams()
        result = run(
            graph(2, {(0, 1): 100.0}), stim=[0], duration=300.0, record=[0, 1], seed=3
        )
        assert len(result.spike_times[0]) > 0
        assert len(result.spike_times[1]) > 0
        assert result.spike_times[1][0] >= result.spike_times[0][0] + params.t_delay


class TestLesionHook:
    def test_silenced_neuron_never_spikes_and_drives_nothing(self):
        weights = graph(3, {(0, 1): 100.0, (1, 2): 100.0})
        intact = run(weights, stim=[0], duration=500.0)
        assert intact.spike_counts[1] > 0 and intact.spike_counts[2] > 0

        lesioned = run(weights, stim=[0], duration=500.0, silenced=[1])
        assert lesioned.spike_counts[1] == 0
        assert lesioned.spike_counts[2] == 0


class TestDeterminism:
    def test_same_seed_gives_identical_spike_trains(self):
        weights = graph(4, {(0, 1): 80.0, (1, 2): 80.0, (2, 3): 80.0})
        a = run(weights, stim=[0], duration=400.0, seed=7, record=[1, 2, 3])
        b = run(weights, stim=[0], duration=400.0, seed=7, record=[1, 2, 3])
        assert np.array_equal(a.spike_counts, b.spike_counts)
        for neuron in (1, 2, 3):
            assert np.array_equal(a.spike_times[neuron], b.spike_times[neuron])

    def test_different_seeds_diverge(self):
        weights = graph(2, {(0, 1): 80.0})
        a = run(weights, stim=[0], duration=400.0, seed=1)
        b = run(weights, stim=[0], duration=400.0, seed=2)
        assert not np.array_equal(a.spike_counts, b.spike_counts)


class TestStimulusRate:
    def test_stimulated_rate_tracks_the_poisson_rate(self):
        """A stimulated neuron fires on essentially every Poisson event.

        The stimulation weight is 68.75 mV against a 7 mV gap to threshold, so each event
        is ten times over-threshold and the neuron's rate should follow the drive closely.
        """
        for rate in (50.0, 150.0, 300.0):
            result = run(graph(1, {}), stim=[0], duration=2000.0, rate=rate, seed=11)
            assert result.rates_hz()[0] == pytest.approx(rate, rel=0.12)


class TestRefractoryDiscardsInput:
    """Synaptic input arriving at a refractory neuron is discarded, not banked.

    This is Brian2's behaviour for a state variable declared ``(unless refractory)``, and
    it is not what the equations suggest on a plain reading. Implementing the obvious
    reading instead - accumulating that input and applying it once the neuron recovers -
    inflated every downstream firing rate in the Shiu reproduction by ~29%, scaling with
    how much time each neuron spent refractory. Nothing subthreshold catches it, because a
    neuron that never spikes is never refractory.

    Verified against Brian2 directly in scripts/probe_refractory.py.
    """

    def test_input_during_refractory_is_dropped(self):
        params = LIFParams()
        # A single synaptic event peaks at only ~0.16x its weight, so 300 synapses
        # (82.5 mV -> ~13 mV deflection) is needed to clear the 7 mV gap to threshold
        # from one event alone.
        weights = graph(2, {(0, 1): 300.0})

        # Neuron 1 spikes at 10 ms, so it is refractory until 12.2 ms. Neuron 0 spikes at
        # 9 ms, and its input lands at 9 + 1.8 = 10.8 ms - inside that window.
        during = LIF().simulate(
            weights,
            StimulusSpec(poisson_targets=np.array([], dtype=np.int64)),
            duration_ms=60.0,
            seed=0,
            record=np.array([1]),
            forced_spikes={1: np.array([10.0]), 0: np.array([9.0])},
        )
        # The only spike neuron 1 should have is the forced one.
        assert during.spike_counts[1] == 1

        # Control: the same input arriving after the refractory window does drive a spike.
        after = LIF().simulate(
            weights,
            StimulusSpec(poisson_targets=np.array([], dtype=np.int64)),
            duration_ms=60.0,
            seed=0,
            record=np.array([1]),
            forced_spikes={1: np.array([10.0]), 0: np.array([20.0])},
        )
        assert after.spike_counts[1] == 2, "control failed: input outside refractory should fire"

        # And the membrane must show nothing at all from the discarded event, rather than a
        # delayed bump once the neuron recovers.
        trace = during.voltages[1]
        recovered = trace[int(13.0 / params.dt) : int(25.0 / params.dt)]
        assert np.allclose(recovered, params.v_rest, atol=1e-3)

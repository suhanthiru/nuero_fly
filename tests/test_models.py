"""The alternative neuron models in the ablation.

Each is tested for the property that makes it a *different* model, not just for running:
conductance synapses must saturate, the rate model must be deterministic, and capacitance
scaling must actually make large cells harder to drive. Without those, an ablation comparing
them would be comparing three copies of the same thing.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from sim.conductance import ConductanceLIF, ConductanceParams
from sim.lif import LIF, LIFParams, _capacitance_scale
from sim.neuron import StimulusSpec
from sim.rate import RateModel, RateParams


def graph(n: int, edges: dict[tuple[int, int], float]) -> sp.csr_matrix:
    matrix = sp.lil_matrix((n, n), dtype=np.float32)
    for (pre, post), count in edges.items():
        matrix[post, pre] = count
    return matrix.tocsr()


def run(model, weights, stim, *, duration=400.0, seed=0, record=None, rate=None):
    return model.simulate(
        weights,
        StimulusSpec(poisson_targets=np.asarray(stim), rate_hz=rate),
        duration_ms=duration,
        seed=seed,
        record=None if record is None else np.asarray(record),
    )


class TestCapacitanceScaling:
    def test_uniform_mode_scales_nothing(self):
        weights = graph(3, {(0, 1): 100.0, (1, 2): 5000.0})
        scale = _capacitance_scale(LIFParams(), weights)
        assert np.allclose(scale, 1.0)

    def test_large_cells_are_harder_to_drive(self):
        """A neuron with far more input synapses must need proportionally more of them."""
        weights = graph(4, {(0, 1): 10.0, (0, 2): 1000.0, (3, 2): 1000.0})
        scale = _capacitance_scale(
            LIFParams(capacitance_mode="synapse_count"), weights
        )
        assert scale[2] > scale[1]

    def test_a_heavily_innervated_neuron_stops_firing_on_a_handful_of_inputs(self):
        """The point of the whole variant, on a toy network.

        Under the uniform assumption a strong synapse fires the target. Once the target's own
        input count is taken into account, the same synapse no longer suffices.
        """
        # Neurons 3-5 give the population a median input count to normalise against;
        # neuron 1 is the heavily-innervated one. Neuron 2 never fires - its only job is to
        # make neuron 1 large.
        weights = graph(
            6,
            {(0, 1): 300.0, (2, 1): 30000.0, (0, 3): 200.0, (0, 4): 200.0, (0, 5): 200.0},
        )
        excitable = run(LIF(LIFParams()), weights, stim=[0], duration=600.0)
        scaled = run(
            LIF(LIFParams(capacitance_mode="synapse_count")), weights, stim=[0],
            duration=600.0,
        )
        assert excitable.spike_counts[1] > 0
        assert scaled.spike_counts[1] < excitable.spike_counts[1]

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError, match="capacitance_mode"):
            _capacitance_scale(LIFParams(capacitance_mode="nonsense"), graph(2, {}))


class TestConductanceModel:
    def test_it_runs_and_propagates(self):
        weights = graph(2, {(0, 1): 400.0})
        result = run(ConductanceLIF(), weights, stim=[0], duration=500.0)
        assert result.spike_counts[0] > 0
        assert result.spike_counts[1] > 0

    def test_membrane_never_exceeds_the_excitatory_reversal(self):
        """The property that distinguishes this model.

        In the current-based model an excitatory spike adds the same voltage however
        depolarised the cell already is, so drive accumulates without bound. Here the driving
        force collapses as v approaches E_e, so no amount of input can push past it.
        """
        params = ConductanceParams(excitatory_reversal_mv=0.0)
        weights = graph(2, {(0, 1): 100000.0})  # absurdly strong, deliberately
        result = run(
            ConductanceLIF(params), weights, stim=[0], duration=400.0, record=[1], rate=400.0
        )
        trace = result.voltages[1]
        assert trace.max() <= params.excitatory_reversal_mv + 1e-3

    def test_inhibition_still_suppresses(self):
        excited = graph(3, {(0, 2): 400.0})
        inhibited = graph(3, {(0, 2): 400.0, (1, 2): -400.0})
        without = run(ConductanceLIF(), excited, stim=[0, 1], duration=600.0)
        with_inh = run(ConductanceLIF(), inhibited, stim=[0, 1], duration=600.0)
        assert without.spike_counts[2] > 0
        assert with_inh.spike_counts[2] < without.spike_counts[2]

    def test_is_deterministic_in_seed(self):
        weights = graph(2, {(0, 1): 400.0})
        a = run(ConductanceLIF(), weights, stim=[0], seed=3)
        b = run(ConductanceLIF(), weights, stim=[0], seed=3)
        assert np.array_equal(a.spike_counts, b.spike_counts)


class TestRateModel:
    def test_it_runs_and_propagates(self):
        weights = graph(2, {(0, 1): 4000.0})
        result = run(RateModel(), weights, stim=[0], duration=500.0, rate=300.0)
        assert result.spike_counts[0] > 0

    def test_has_no_stochasticity_at_all(self):
        """Different seeds must give identical output - there is no RNG in this model.

        That is what makes it the clean test of Phase 3's diagnosis, which was that
        trial-to-trial Poisson variability in a ~6-spike decision signal destroyed the
        escape heading.
        """
        weights = graph(3, {(0, 1): 4000.0, (1, 2): 4000.0})
        a = run(RateModel(), weights, stim=[0], seed=1, rate=300.0)
        b = run(RateModel(), weights, stim=[0], seed=999, rate=300.0)
        assert np.array_equal(a.spike_counts, b.spike_counts)

    def test_rate_to_spike_conversion_preserves_the_rate(self):
        """Spikes are emitted once per unit of accumulated rate, so count ~ rate x time."""
        params = RateParams()
        weights = graph(1, {})
        target_hz = 200.0
        result = run(
            RateModel(params), weights, stim=[0], duration=2000.0, rate=target_hz,
            record=[0],
        )
        settled = result.voltages[0][-1000:].mean()  # rates, not millivolts
        assert settled == pytest.approx(target_hz, rel=0.1)
        expected = target_hz * 2.0  # 2 seconds
        assert result.spike_counts[0] == pytest.approx(expected, rel=0.15)

    def test_silenced_neurons_stay_at_zero(self):
        weights = graph(3, {(0, 1): 4000.0, (1, 2): 4000.0})
        result = RateModel().simulate(
            weights,
            StimulusSpec(
                poisson_targets=np.array([0]), silenced=np.array([1]), rate_hz=300.0
            ),
            duration_ms=500.0,
            seed=0,
        )
        assert result.spike_counts[1] == 0
        assert result.spike_counts[2] == 0

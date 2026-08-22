"""Neuron model interface.

The neuron model sits behind an interface from the start so that swapping leaky
integrate-and-fire for a conductance-based model, a rate model, or a learned surrogate is a
module swap rather than a rewrite. Nothing outside :mod:`sim` should care which one is in
use, and nothing in :mod:`sim` may import from ``viz`` or ``world``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class StimulusSpec:
    """Which neurons are driven, and which are prevented from firing.

    ``poisson_targets`` receive independent Poisson drive. ``silenced`` are held below
    threshold for the whole run - the lesion hook, and the reason the lesion sweep is cheap:
    it is a mask over a connectome that never has to be rebuilt.
    """

    poisson_targets: np.ndarray
    silenced: np.ndarray | None = None
    rate_hz: float | None = None
    # Optional (n_steps, len(poisson_targets)) array of per-neuron rates in Hz. This is how
    # a sensory encoder drives the network: each target follows its own time course rather
    # than a single constant rate. Overrides rate_hz when present.
    rate_schedule: np.ndarray | None = None


@dataclass(frozen=True)
class SimulationResult:
    """Outcome of one run."""

    spike_counts: np.ndarray            # (N,) spikes per neuron over the whole run
    duration_ms: float
    dt_ms: float
    spike_times: dict[int, np.ndarray]  # index -> spike times (ms), for recorded neurons
    # Membrane trace per recorded neuron. Cheap when few neurons are recorded, and the only
    # way to check the integrator against a closed-form solution rather than against itself.
    voltages: dict[int, np.ndarray]
    meta: dict[str, Any]

    def rates_hz(self) -> np.ndarray:
        return self.spike_counts / (self.duration_ms / 1000.0)


class NeuronModel(ABC):
    """A dynamical model that turns a weighted graph plus a stimulus into spikes."""

    name: str

    @abstractmethod
    def simulate(
        self,
        weights: sp.csr_matrix,
        stimulus: StimulusSpec,
        *,
        duration_ms: float,
        seed: int,
        record: np.ndarray | None = None,
    ) -> SimulationResult:
        """Run one trial.

        ``weights`` is oriented ``[post, pre]`` and carries signed synapse counts, as built
        by :mod:`data.loader`. Scaling from synapse count to physical units is the model's
        business, not the loader's.

        Must be deterministic in ``seed``: identical inputs and seed give byte-identical
        spike trains. The lesion sweep compares runs against each other, so any hidden
        nondeterminism would show up as a fake effect.
        """

    def with_overrides(self, **kwargs) -> "NeuronModel":
        return replace(self, **kwargs)  # type: ignore[type-var]

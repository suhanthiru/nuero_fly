"""Sensory encoding interface.

The encoder is the boundary between the world and the connectome, and it is deliberately
a swappable interface rather than inlined logic. Version 1 computes angular size and
expansion rate analytically from scene geometry and drives the looming-sensitive visual
projection neurons directly, which bypasses the optic lobe entirely - those ~50k neurons
stay silent and render as anatomical context. Replacing that with an ommatidial model that
actually drives the retina must be a module swap, not a refactor.

Consequently: nothing outside :mod:`sim.encoders.analytic` may assume that LC or LPLC2
neurons are where drive is injected. The rest of the pipeline sees an opaque mapping from
neuron to drive.

Drive is expressed as a **firing rate in Hz**, not a current in nA as the original spec
suggested. That is a deliberate deviation. The neuron model was validated against Shiu et
al.'s reference implementation whose only stimulation mechanism is a Poisson process onto
the membrane, and inventing a second, unvalidated injection pathway to satisfy a units
convention would put untested numerics underneath every result that follows. Rate is also
the quantity the looming literature actually reports for these cells.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class SceneState:
    """The world as the encoder sees it, at one instant.

    Distances are millimetres, time is milliseconds, angles are degrees. The fly sits at
    the origin; azimuth is measured about the fly's vertical axis with 0 directly ahead and
    positive to the fly's left.
    """

    time_ms: float
    distance_mm: float          # centre-to-centre separation
    radius_mm: float            # physical half-size of the approaching object
    closing_speed_mm_per_ms: float
    azimuth_deg: float = 0.0
    elevation_deg: float = 0.0

    @property
    def has_collided(self) -> bool:
        return self.distance_mm <= self.radius_mm


class SensoryEncoder(ABC):
    """Turns scene geometry into per-neuron drive."""

    name: str

    @abstractmethod
    def encode(self, scene: SceneState) -> dict[int, float]:
        """Map body id -> drive in Hz for this instant.

        Neurons absent from the returned mapping receive no drive. Implementations must be
        pure: the same scene must give the same answer, so that runs stay reproducible.
        """

    @abstractmethod
    def target_ids(self) -> list[int]:
        """Every neuron this encoder can ever drive.

        Declared up front so the simulator can allocate a fixed drive schedule rather than
        discovering targets as it goes.
        """

"""Scripted predator approaches.

A looming disc on a straight-line course toward the fly, parameterised by azimuth and by
the ratio l/|v| the looming literature sweeps. There is no pursuit: the predator commits to
its trajectory and does not react to the escape. That is deliberate for v1 and matches how
the behavioural experiments are actually run - a projected disc on a screen does not chase.
A pursuit agent is a later phase.

**Coordinate frame.** The fly sits at the origin facing +X. Azimuth is measured about the
vertical axis from +X toward +Y, so 0 degrees is head-on, +90 is off the fly's left, -90 off
its right, and 180 is directly behind. Everything is metres and seconds here, because that
is what MuJoCo wants; the neural side works in millimetres and milliseconds, and
:meth:`ApproachTrajectory.scene_at` is the one place the two meet.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from sim.encoders.base import SceneState

MM_PER_M = 1000.0


@dataclass(frozen=True)
class ApproachTrajectory:
    """A predator closing on the fly at constant speed along a fixed bearing.

    ``ratio_ms`` is l/|v| in milliseconds; together with ``radius_mm`` it fixes the closing
    speed, which is what makes two approaches with the same ratio produce identical angular
    profiles regardless of absolute scale.
    """

    ratio_ms: float = 40.0
    radius_mm: float = 10.0
    azimuth_deg: float = 0.0
    collision_ms: float = 600.0
    #: Height of the predator's centre above the floor, in metres. Defaults to the fly's
    #: own centre height so a head-on approach actually intersects it - with the predator
    #: travelling above the fly, a stationary fly is never caught and every escape looks
    #: successful. The predator is a kinematic volume, not a colliding body, so it is
    #: allowed to intersect the floor: it stands in for a disc filling the visual field.
    height_m: float = 0.0008

    @property
    def speed_mm_per_ms(self) -> float:
        return self.radius_mm / self.ratio_ms

    @property
    def start_distance_mm(self) -> float:
        """Chosen so contact always happens at ``collision_ms``, whatever the ratio.

        Fixing the moment of contact rather than the starting distance keeps escape latency
        comparable across the sweep instead of confounded with trial length.
        """
        return self.radius_mm + self.speed_mm_per_ms * self.collision_ms

    def distance_mm(self, time_ms: float) -> float:
        """Centre-to-centre separation. Never closer than the predator's own radius."""
        raw = self.start_distance_mm - self.speed_mm_per_ms * time_ms
        return max(self.radius_mm, raw)

    def has_contacted(self, time_ms: float) -> bool:
        return self.start_distance_mm - self.speed_mm_per_ms * time_ms <= self.radius_mm

    def bearing(self) -> np.ndarray:
        """Unit vector from the fly toward the predator, in world coordinates."""
        angle = math.radians(self.azimuth_deg)
        return np.array([math.cos(angle), math.sin(angle), 0.0])

    def position_m(self, time_ms: float) -> np.ndarray:
        """Predator centre in metres at ``time_ms``."""
        offset = self.bearing() * (self.distance_mm(time_ms) / MM_PER_M)
        return np.array([offset[0], offset[1], self.height_m])

    def scene_at(self, time_ms: float) -> SceneState:
        """The encoder's view of this instant.

        The single conversion point between the world's metres and the encoder's
        millimetres. After contact the closing speed is reported as zero, because the
        stimulus stops - which matters, since both encoder channels are gated on expansion.
        """
        return SceneState(
            time_ms=time_ms,
            distance_mm=self.distance_mm(time_ms),
            radius_mm=self.radius_mm,
            closing_speed_mm_per_ms=(
                0.0 if self.has_contacted(time_ms) else self.speed_mm_per_ms
            ),
            azimuth_deg=self.azimuth_deg,
        )

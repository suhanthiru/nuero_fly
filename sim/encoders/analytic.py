"""Analytic looming encoder: scene geometry -> LC4 and LPLC2 drive.

The formulation is taken from Ache, Polsky, Alghailani et al., "Neural Basis for Looming
Size and Velocity Encoding in the Drosophila Giant Fiber Escape Pathway", Current Biology
29:1073-1081 (2019). They show that the giant fiber's looming response is reproduced by
summing two separately-carried features:

* **LC4 encodes angular velocity**, as a roughly *linear* function of theta-dot.
* **LPLC2 encodes angular size**, as a *Gaussian* function of theta.

Silencing LPLC2 removes the size component from GF recordings and leaves the velocity
component intact, which is what makes these two separable channels rather than a modelling
convenience. Their synapse counts onto the GF - 55 LC4 and 108 LPLC2 in FAFB - are also
close to what our own Phase 0 count found in MaleCNS, so the two channels are carried by
the same cells here.

Geometry. For an object of physical radius ``R`` at distance ``d``, closing at speed ``v``::

    theta      = 2 * arctan(R / d)
    theta_dot  = 2 * R * v / (d^2 + R^2)

Substituting ``d = v * tau`` for time-to-collision ``tau`` gives the standard form in terms
of the ratio ``r = R/|v|``, the parameter the looming literature sweeps::

    theta(tau)     = 2 * arctan(r / tau)
    theta_dot(tau) = 2 * r / (tau^2 + r^2)

so theta-dot approaches ``2/r`` as tau -> 0: small r means a fast, late, violent expansion,
large r a slow early one. The stimulus is a looming *disc*, which is what the behavioural
literature actually presents, so the arctangent form applies and the disc subtends 90 deg
when its surface reaches the eye - theta-dot is ``1/r`` at that moment, not ``2/r``.

WHAT IS OURS AND NOT THEIRS. The shape of the two tuning curves is published. Their
parameters here are not: Ache et al. fit membrane responses, not firing rates, and we need
rates. Every constant in :class:`LoomingTuning` is therefore our choice, marked as such,
and the escape result's sensitivity to them has to be swept before any claim leans on it.
This is the largest block of hand-set numbers in the project and it sits directly upstream
of the headline behaviour, which is exactly why it is isolated in one dataclass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .base import SceneState, SensoryEncoder


@dataclass(frozen=True)
class LoomingTuning:
    """Tuning constants. **Every value here is ours, not from the literature.**

    Ache et al. give the functional forms; converting them to firing rates needs a scale,
    and that scale is a free parameter. Held together in one place so a sensitivity sweep
    can vary them as a block and so nobody mistakes them for measured quantities.
    """

    # LC4: linear in angular velocity. The gain sets how much of the achievable theta-dot
    # range maps onto the neuron's usable rate range. Peak theta-dot at contact is 2/r
    # rad/ms, i.e. ~11 deg/ms at r = 10 ms and ~1.4 deg/ms at r = 80 ms, so this gain puts
    # fast looming near saturation and slow looming well below it - which is the regime the
    # mode split is supposed to live in.
    lc4_gain_hz_per_deg_per_ms: float = 25.0
    lc4_max_hz: float = 150.0

    # LPLC2: Gaussian in angular size. Centred where GF responses peak for typical looming
    # stimuli; the width is wide enough that the channel is active across most of an
    # approach rather than only at one instant.
    lplc2_peak_deg: float = 65.0
    lplc2_width_deg: float = 25.0
    lplc2_max_hz: float = 150.0

    # Neither channel is driven by a receding or stationary object.
    min_theta_dot_deg_per_ms: float = 0.0

    # Stands in for the optic lobe we are not simulating. Phototransduction plus lamina,
    # medulla and lobula processing delays the visual response by roughly this much in
    # Drosophila; because the analytic encoder injects at LC directly, that latency has to
    # be reinstated by hand or every escape comes out implausibly early.
    visual_latency_ms: float = 25.0

    # Ceiling on rate. The neuron model was validated with Poisson drive in this range and
    # the stimulation weight is ~10x the gap to threshold, so a driven neuron fires close to
    # 1:1 with its drive. Going far above this leaves the validated regime.
    max_rate_hz: float = 150.0


def angular_size_deg(radius_mm: float, distance_mm: float) -> float:
    """Subtended angle of a disc of half-width ``radius_mm`` at ``distance_mm``.

    A disc rather than a sphere, because that is the stimulus the looming literature
    presents and the one the arctangent form describes. It subtends 90 degrees when its
    surface reaches the eye and only approaches 180 as the distance goes to zero.
    """
    if distance_mm <= 0.0:
        return 180.0
    return 2.0 * math.degrees(math.atan(radius_mm / distance_mm))


def angular_velocity_deg_per_ms(
    radius_mm: float, distance_mm: float, closing_speed_mm_per_ms: float
) -> float:
    """d(theta)/dt for an object closing head-on.

    ``2 R v / (d^2 + R^2)``, the derivative of the expression above with ``dd/dt = -v``.
    A stationary or receding object gives zero or negative expansion.
    """
    if distance_mm <= 0.0 or closing_speed_mm_per_ms <= 0.0:
        return 0.0
    radians_per_ms = (
        2.0 * radius_mm * closing_speed_mm_per_ms
        / (distance_mm * distance_mm + radius_mm * radius_mm)
    )
    return math.degrees(radians_per_ms)


@dataclass(frozen=True)
class LoomingTrajectory:
    """A head-on approach, parameterised the way the looming literature parameterises it.

    ``half_size_over_speed_ms`` is the ratio r = R/|v| that the sweep varies. Two stimuli
    with the same r have identical angular profiles regardless of absolute size or speed,
    which is why it is the natural axis.
    """

    half_size_over_speed_ms: float
    radius_mm: float = 10.0
    start_distance_mm: float = 400.0
    azimuth_deg: float = 0.0

    @property
    def speed_mm_per_ms(self) -> float:
        return self.radius_mm / self.half_size_over_speed_ms

    @property
    def collision_time_ms(self) -> float:
        """When the surface reaches the fly, measured from the start of the trial."""
        return (self.start_distance_mm - self.radius_mm) / self.speed_mm_per_ms

    def state_at(self, time_ms: float) -> SceneState:
        """Scene at ``time_ms``. After contact the object stops, as it does on a screen."""
        raw = self.start_distance_mm - self.speed_mm_per_ms * time_ms
        collided = raw <= self.radius_mm
        return SceneState(
            time_ms=time_ms,
            distance_mm=max(self.radius_mm, raw),
            radius_mm=self.radius_mm,
            closing_speed_mm_per_ms=0.0 if collided else self.speed_mm_per_ms,
            azimuth_deg=self.azimuth_deg,
        )


def _hemisphere_weight(azimuth_deg: float, side: str) -> float:
    """How strongly a stimulus at this azimuth drives one optic lobe.

    HAND-ADDED, and a crude stand-in. Real retinotopy lives in the optic lobe, which the
    analytic encoder bypasses, so there is nothing in the model to produce it. A raised
    cosine centred on each eye is the least-committal thing that produces the ipsilateral
    bias needed for any directional result, and it should be treated as scaffolding rather
    than as a prediction. Note also that Drosophila GF responses are reported to be largely
    azimuth-invariant (Jones et al., J Exp Biol 2023), so a strong azimuth dependence here
    would itself be suspect.
    """
    centre = 60.0 if side == "L" else -60.0
    offset = math.radians(azimuth_deg - centre)
    return max(0.0, math.cos(offset / 2.0))


class AnalyticLoomingEncoder(SensoryEncoder):
    """Drives LC4 from angular velocity and LPLC2 from angular size.

    This is the only place in the codebase that knows drive is injected at LC neurons.
    """

    name = "analytic-looming"

    def __init__(
        self,
        lc4_ids: dict[int, str],
        lplc2_ids: dict[int, str],
        tuning: LoomingTuning | None = None,
        *,
        azimuth_weighting: bool = False,
    ) -> None:
        """``lc4_ids`` and ``lplc2_ids`` map body id -> hemisphere ("L", "R" or "")."""
        self.lc4_ids = dict(lc4_ids)
        self.lplc2_ids = dict(lplc2_ids)
        self.tuning = tuning or LoomingTuning()
        self.azimuth_weighting = azimuth_weighting

    @classmethod
    def from_connectome(cls, connectome, **kwargs) -> "AnalyticLoomingEncoder":
        from data.cell_types import ids_for  # edge of the package; kept out of module scope

        def sided(cell_type: str) -> dict[int, str]:
            ids = ids_for(connectome, cell_type)
            sides = connectome.annotations.loc[ids, "side"]
            return {int(i): str(s) for i, s in zip(ids, sides)}

        return cls(sided("LC4"), sided("LPLC2"), **kwargs)

    def target_ids(self) -> list[int]:
        return sorted(set(self.lc4_ids) | set(self.lplc2_ids))

    def channel_rates(self, scene: SceneState) -> tuple[float, float, float, float]:
        """The two channels plus the geometry they came from.

        Returns ``(theta_deg, theta_dot_deg_per_ms, lc4_hz, lplc2_hz)``. Exposed separately
        from :meth:`encode` so the trace panel and the tests can see theta and theta-dot
        without going through per-neuron drive.
        """
        tuning = self.tuning

        # Apply the visual latency by reading the scene as it was, not as it is.
        delayed_time = scene.time_ms - tuning.visual_latency_ms
        if delayed_time < 0:
            return 0.0, 0.0, 0.0, 0.0
        distance = scene.distance_mm + scene.closing_speed_mm_per_ms * tuning.visual_latency_ms

        theta = angular_size_deg(scene.radius_mm, distance)
        theta_dot = angular_velocity_deg_per_ms(
            scene.radius_mm, distance, scene.closing_speed_mm_per_ms
        )

        # LC4: linear in angular velocity (Ache et al. 2019).
        lc4 = 0.0
        if theta_dot > tuning.min_theta_dot_deg_per_ms:
            lc4 = min(tuning.lc4_gain_hz_per_deg_per_ms * theta_dot, tuning.lc4_max_hz)

        # LPLC2: Gaussian in angular size (Ache et al. 2019).
        deviation = (theta - tuning.lplc2_peak_deg) / tuning.lplc2_width_deg
        lplc2 = tuning.lplc2_max_hz * math.exp(-0.5 * deviation * deviation)
        if theta_dot <= tuning.min_theta_dot_deg_per_ms:
            lplc2 = 0.0  # a stationary object of the right size is not a looming object

        return theta, theta_dot, lc4, lplc2

    def encode(self, scene: SceneState) -> dict[int, float]:
        _, _, lc4_hz, lplc2_hz = self.channel_rates(scene)
        cap = self.tuning.max_rate_hz

        drive: dict[int, float] = {}
        for population, rate in ((self.lc4_ids, lc4_hz), (self.lplc2_ids, lplc2_hz)):
            for body_id, side in population.items():
                value = rate
                if self.azimuth_weighting:
                    value *= _hemisphere_weight(scene.azimuth_deg, side)
                if value > 0.0:
                    drive[body_id] = min(value, cap)
        return drive

"""MuJoCo arena: a fly, a floor, and an approaching predator.

The fly is a rigid body and the escape is a triggered impulse. There are no leg mechanics
and no wing aerodynamics - leg dynamics is a project of its own and is not on the critical
path to asking whether the connectome produces an escape at the right *time* and in the
right *direction*.

**How a trial is structured, and why.** The predator's course is scripted and the fly does
not move until it jumps, so the approach needs no physics at all: it is pure kinematics, and
the neural simulation can be run over it in one batch. Physics only matters from the moment
of takeoff. So a trial is:

    1. kinematics  - predator closes; the encoder and neuron model decide when to fire
    2. physics     - at takeoff an impulse is applied; the predator continues on its course
    3. adjudication- did the predator's sphere ever reach the fly?

Splitting it this way is not a shortcut. Running the LIF closed-loop with the physics clock
would buy nothing while the fly is stationary, and would cost the batched `lax.scan` that
makes the neural side fast.

Units here are MuJoCo's: metres, kilograms, seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .predator import MM_PER_M, ApproachTrajectory

# Drosophila melanogaster, roughly. Body ~2.5 mm long, ~1 mg.
FLY_HALF_LENGTH_M = 0.00125
FLY_HALF_WIDTH_M = 0.0008
FLY_MASS_KG = 1e-6

# Escape takeoff speed. Measured escape velocities for Drosophila are a few tenths of a
# metre per second; this is a hand-set stand-in for leg mechanics we do not model, and the
# escape success rate obviously depends on it, so it is swept rather than trusted.
TAKEOFF_SPEED_M_PER_S = 0.35
TAKEOFF_ELEVATION_DEG = 45.0

PHYSICS_TIMESTEP_S = 2e-4


def _model_xml(predator_radius_m: float) -> str:
    return f"""
<mujoco model="escape-arena">
  <option gravity="0 0 -9.81" timestep="{PHYSICS_TIMESTEP_S}" integrator="RK4"/>
  <default>
    <geom friction="0.8 0.005 0.0001"/>
  </default>
  <worldbody>
    <light pos="0 0 0.3"/>
    <geom name="floor" type="plane" size="0.5 0.5 0.01" rgba="0.15 0.16 0.18 1"/>
    <body name="fly" pos="0 0 {FLY_HALF_WIDTH_M}">
      <freejoint name="fly_root"/>
      <geom name="fly_body" type="ellipsoid"
            size="{FLY_HALF_LENGTH_M} {FLY_HALF_WIDTH_M} {FLY_HALF_WIDTH_M}"
            mass="{FLY_MASS_KG}" rgba="0.9 0.75 0.35 1"/>
    </body>
    <body name="predator" mocap="true" pos="1 0 0.01">
      <geom name="predator_body" type="sphere" size="{predator_radius_m}"
            rgba="0.75 0.25 0.3 0.6" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


@dataclass
class EscapeOutcome:
    """What happened physically."""

    escaped: bool
    took_off: bool
    takeoff_ms: float | None
    #: Closest the predator's surface ever came to the fly's surface, in millimetres.
    #: Negative means contact.
    closest_approach_mm: float
    #: Horizontal direction the fly actually travelled, in degrees, same convention as
    #: azimuth. None if it never left the ground.
    escape_heading_deg: float | None
    #: Angle between the escape heading and the direction *away* from the threat. Zero is
    #: a perfect escape directly away; 180 is straight into it.
    error_from_away_deg: float | None
    displacement_mm: float
    path: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))


def _wrap180(degrees: float) -> float:
    return (degrees + 180.0) % 360.0 - 180.0


class Arena:
    """One trial's physics."""

    def __init__(self, trajectory: ApproachTrajectory) -> None:
        self.trajectory = trajectory
        self.model = mujoco.MjModel.from_xml_string(
            _model_xml(trajectory.radius_mm / MM_PER_M)
        )
        self.data = mujoco.MjData(self.model)
        self.fly_body = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "fly"
        )
        self.predator_mocap = self.model.body_mocapid[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "predator")
        ]

    def run(
        self,
        *,
        takeoff_ms: float | None,
        heading_deg: float,
        duration_ms: float,
        speed_m_per_s: float = TAKEOFF_SPEED_M_PER_S,
        elevation_deg: float = TAKEOFF_ELEVATION_DEG,
    ) -> EscapeOutcome:
        """Step the trial and adjudicate it.

        ``takeoff_ms`` of None means the circuit never fired, in which case the fly simply
        sits there and is caught if the predator passes through it.
        """
        model, data = self.model, self.data
        mujoco.mj_resetData(model, data)

        steps = int(round(duration_ms / 1000.0 / PHYSICS_TIMESTEP_S))
        capture_gap_m = (
            self.trajectory.radius_mm / MM_PER_M + FLY_HALF_WIDTH_M
        )
        launched = False
        # Pose to hold the fly at until it jumps. Zeroing velocity alone is not enough:
        # mj_step still integrates gravity within the step, which accumulates to a few
        # tenths of a millimetre of sag over a trial and shows up as spurious displacement.
        resting_qpos = data.qpos.copy()
        closest_m = float("inf")
        path = np.empty((steps, 3))
        start_xy = None

        for step in range(steps):
            time_ms = step * PHYSICS_TIMESTEP_S * 1000.0
            data.mocap_pos[self.predator_mocap] = self.trajectory.position_m(time_ms)

            if takeoff_ms is not None and not launched and time_ms >= takeoff_ms:
                self._launch(heading_deg, speed_m_per_s, elevation_deg)
                launched = True
                start_xy = data.xpos[self.fly_body][:2].copy()
            elif not launched:
                data.qpos[:] = resting_qpos
                data.qvel[:] = 0.0

            mujoco.mj_step(model, data)

            fly = data.xpos[self.fly_body]
            path[step] = fly
            separation = float(
                np.linalg.norm(fly - data.mocap_pos[self.predator_mocap])
            )
            closest_m = min(closest_m, separation - capture_gap_m)

        heading, error, displacement = self._summarise(path, launched, start_xy)
        return EscapeOutcome(
            escaped=closest_m > 0.0,
            took_off=launched,
            takeoff_ms=takeoff_ms if launched else None,
            closest_approach_mm=closest_m * MM_PER_M,
            escape_heading_deg=heading,
            error_from_away_deg=error,
            displacement_mm=displacement,
            path=path,
        )

    def _launch(self, heading_deg: float, speed: float, elevation_deg: float) -> None:
        """Set the fly's velocity. The escape is an impulse, not a leg model."""
        heading = np.radians(heading_deg)
        elevation = np.radians(elevation_deg)
        horizontal = speed * np.cos(elevation)
        self.data.qvel[0] = horizontal * np.cos(heading)
        self.data.qvel[1] = horizontal * np.sin(heading)
        self.data.qvel[2] = speed * np.sin(elevation)

    def _summarise(
        self, path: np.ndarray, launched: bool, start_xy
    ) -> tuple[float | None, float | None, float]:
        if not launched or start_xy is None:
            return None, None, 0.0
        travelled = path[-1][:2] - start_xy
        displacement = float(np.linalg.norm(travelled)) * MM_PER_M
        if displacement < 1e-3:
            return None, None, displacement
        heading = float(np.degrees(np.arctan2(travelled[1], travelled[0])))
        # Directly away from the threat is the predator's bearing plus 180 degrees.
        away = _wrap180(self.trajectory.azimuth_deg + 180.0)
        return heading, abs(_wrap180(heading - away)), displacement

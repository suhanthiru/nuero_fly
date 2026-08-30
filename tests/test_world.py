"""Predator geometry and the arena's adjudication.

The physics is simple enough that the risk is not numerical - it is getting a convention
wrong, so that every escape looks successful or every heading points the wrong way. These
pin the conventions rather than the numbers.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sim.decoder import heading_from_asymmetry
from world.arena import PHYSICS_TIMESTEP_S, Arena
from world.predator import ApproachTrajectory


class TestApproachGeometry:
    def test_contact_happens_at_the_stated_time_whatever_the_ratio(self):
        """Start distance is derived so the moment of contact is fixed.

        Without this, slow approaches would take longer trials and escape latency would be
        confounded with trial length across the sweep.
        """
        for ratio in (10.0, 40.0, 80.0):
            trajectory = ApproachTrajectory(ratio_ms=ratio, collision_ms=600.0)
            assert trajectory.distance_mm(599.0) > trajectory.radius_mm
            assert trajectory.distance_mm(600.0) == pytest.approx(trajectory.radius_mm)
            assert trajectory.has_contacted(600.1)
            assert not trajectory.has_contacted(599.0)

    def test_closing_speed_follows_the_ratio(self):
        trajectory = ApproachTrajectory(ratio_ms=40.0, radius_mm=10.0)
        assert trajectory.speed_mm_per_ms == pytest.approx(0.25)

    @pytest.mark.parametrize(
        ("azimuth", "expected"),
        [(0.0, (1.0, 0.0)), (90.0, (0.0, 1.0)), (180.0, (-1.0, 0.0)), (-90.0, (0.0, -1.0))],
    )
    def test_bearing_convention(self, azimuth, expected):
        """0 is head-on (+X), +90 is off the fly's left (+Y)."""
        bearing = ApproachTrajectory(azimuth_deg=azimuth).bearing()
        assert bearing[0] == pytest.approx(expected[0], abs=1e-9)
        assert bearing[1] == pytest.approx(expected[1], abs=1e-9)

    def test_scene_reports_no_expansion_after_contact(self):
        """Both encoder channels are gated on expansion, so this matters."""
        trajectory = ApproachTrajectory(ratio_ms=40.0, collision_ms=600.0)
        assert trajectory.scene_at(300.0).closing_speed_mm_per_ms > 0
        assert trajectory.scene_at(700.0).closing_speed_mm_per_ms == 0.0

    def test_predator_travels_at_the_flys_height(self):
        """Otherwise a head-on approach passes over a stationary fly and never catches it."""
        trajectory = ApproachTrajectory()
        assert trajectory.height_m < 0.002


class TestArenaAdjudication:
    def trajectory(self, azimuth: float = 0.0) -> ApproachTrajectory:
        return ApproachTrajectory(ratio_ms=40.0, azimuth_deg=azimuth, collision_ms=600.0)

    def test_a_fly_that_never_jumps_is_caught(self):
        """The control that proves the adjudication can fail at all."""
        outcome = Arena(self.trajectory()).run(
            takeoff_ms=None, heading_deg=180.0, duration_ms=800.0
        )
        assert not outcome.took_off
        assert not outcome.escaped
        assert outcome.closest_approach_mm < 0

    @pytest.mark.parametrize("azimuth", [0.0, 90.0, 180.0, -90.0])
    def test_jumping_away_escapes_and_jumping_into_it_does_not(self, azimuth):
        arena = Arena(self.trajectory(azimuth))
        away = (azimuth + 180.0 + 180.0) % 360.0 - 180.0

        escaped = arena.run(takeoff_ms=400.0, heading_deg=away, duration_ms=800.0)
        assert escaped.escaped
        assert escaped.error_from_away_deg == pytest.approx(0.0, abs=1e-6)

        caught = arena.run(takeoff_ms=400.0, heading_deg=azimuth, duration_ms=800.0)
        assert not caught.escaped
        assert caught.error_from_away_deg == pytest.approx(180.0, abs=1e-6)

    def test_the_fly_does_not_drift_before_it_jumps(self):
        """Held still until takeoff, so displacement is the escape and not gravity.

        Checked on the path rather than on total displacement: with a late takeoff the fly
        legitimately moves, so only the pre-takeoff samples are informative.
        """
        outcome = Arena(self.trajectory()).run(
            takeoff_ms=400.0, heading_deg=180.0, duration_ms=800.0
        )
        before = outcome.path[: int(0.3 / PHYSICS_TIMESTEP_S)]  # 300 ms, before takeoff
        assert np.allclose(before, before[0], atol=1e-9)

    def test_both_paths_are_recorded_on_the_neural_clock(self):
        """The 3D view drives fly and predator from one index, so they must share a clock."""
        outcome = Arena(self.trajectory()).run(
            takeoff_ms=400.0, heading_deg=180.0, duration_ms=800.0
        )
        assert outcome.path.shape == outcome.predator_path.shape
        assert outcome.path.shape[0] == int(round(0.8 / PHYSICS_TIMESTEP_S))
        # The predator must actually close on the fly over the trial.
        start = np.linalg.norm(outcome.predator_path[0])
        end = np.linalg.norm(outcome.predator_path[-1])
        assert end < start

    def test_late_takeoff_fails_and_early_takeoff_succeeds(self):
        arena = Arena(self.trajectory())
        early = arena.run(takeoff_ms=300.0, heading_deg=180.0, duration_ms=800.0)
        late = arena.run(takeoff_ms=599.0, heading_deg=180.0, duration_ms=800.0)
        assert early.closest_approach_mm > late.closest_approach_mm


class TestHeadingReadout:
    def test_symmetric_output_jumps_straight_back(self):
        """The null. Equal drive must give no directional preference at all."""
        assert heading_from_asymmetry(5, 5) == pytest.approx(180.0)
        assert heading_from_asymmetry(0, 0) == pytest.approx(180.0)

    def test_asymmetry_steers_contralaterally(self):
        """More left-side drive sends the fly to its right, i.e. toward negative azimuth."""
        left_heavy = heading_from_asymmetry(10, 0)
        right_heavy = heading_from_asymmetry(0, 10)
        assert left_heavy == pytest.approx(-90.0)
        assert right_heavy == pytest.approx(90.0)

    def test_heading_is_monotonic_in_asymmetry(self):
        # Sweeping from all-right to all-left takes the heading from +90 through 180 to
        # -90, so it is monotonic once unwrapped past the wrap point.
        headings = [heading_from_asymmetry(n, 10 - n) for n in range(0, 11)]
        unwrapped = np.unwrap(np.radians(headings))
        assert np.all(np.diff(unwrapped) > 0)
        assert headings[0] == pytest.approx(90.0)
        assert headings[-1] == pytest.approx(-90.0)

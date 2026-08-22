"""Looming geometry and the two encoding channels.

Geometry is checked against hand-computed angles and against numerical differentiation,
not against another part of the same code. The tuning curves are checked for the shapes
Ache et al. 2019 report - linear in angular velocity for LC4, Gaussian in angular size for
LPLC2 - rather than for particular rate values, since the rate scale is ours and arbitrary.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from sim.encoders.analytic import (
    AnalyticLoomingEncoder,
    LoomingTrajectory,
    LoomingTuning,
    angular_size_deg,
    angular_velocity_deg_per_ms,
)
from sim.encoders.base import SceneState
from sim.encoders.ommatidial import OmmatidialEncoder


class TestAngularSize:
    @pytest.mark.parametrize(
        ("radius", "distance", "expected_deg"),
        [
            (10.0, 10.0, 90.0),                    # atan(1) = 45 deg, doubled: contact
            (10.0, 10.0 / math.tan(math.radians(30.0)), 60.0),
            (1.0, 100.0, 2 * math.degrees(math.atan(0.01))),
            (5.0, 5.0 / math.tan(math.radians(45.0)), 90.0),
        ],
    )
    def test_matches_hand_computed_geometry(self, radius, distance, expected_deg):
        assert angular_size_deg(radius, distance) == pytest.approx(expected_deg, abs=1e-9)

    def test_approaches_180_only_at_zero_distance(self):
        assert angular_size_deg(10.0, 10.0) == pytest.approx(90.0)
        assert angular_size_deg(10.0, 1.0) == pytest.approx(
            2 * math.degrees(math.atan(10.0))
        )
        assert angular_size_deg(10.0, 0.0) == 180.0

    def test_grows_monotonically_as_the_object_approaches(self):
        distances = np.linspace(500.0, 15.0, 60)
        sizes = [angular_size_deg(10.0, d) for d in distances]
        assert np.all(np.diff(sizes) > 0)


class TestAngularVelocity:
    def test_matches_hand_computed_value(self):
        # 2 R v / (d^2 + R^2) = 2*10*1 / (100^2 + 10^2) rad/ms
        expected = math.degrees(2 * 10.0 * 1.0 / (100.0**2 + 10.0**2))
        assert angular_velocity_deg_per_ms(10.0, 100.0, 1.0) == pytest.approx(expected)

    def test_matches_numerical_derivative_of_angular_size(self):
        """The analytic derivative against a finite difference of the size function."""
        radius, speed = 10.0, 0.5
        for distance in (300.0, 150.0, 80.0, 40.0, 20.0):
            dt = 1e-4
            before = angular_size_deg(radius, distance + speed * dt / 2)
            after = angular_size_deg(radius, distance - speed * dt / 2)
            numeric = (after - before) / dt
            analytic = angular_velocity_deg_per_ms(radius, distance, speed)
            assert analytic == pytest.approx(numeric, rel=1e-5)

    def test_expansion_rate_at_surface_contact_is_one_over_r(self):
        """At d = R the expression 2Rv/(d^2+R^2) collapses to v/R = 1/(R/|v|)."""
        for ratio in (10.0, 40.0, 80.0):
            radius, speed = 10.0, 10.0 / ratio
            at_contact = angular_velocity_deg_per_ms(radius, radius, speed)
            assert at_contact == pytest.approx(math.degrees(1.0 / ratio), rel=1e-9)

    def test_expansion_rate_grows_as_the_object_closes(self):
        radius, speed = 10.0, 0.5
        rates = [angular_velocity_deg_per_ms(radius, d, speed)
                 for d in (400.0, 200.0, 100.0, 50.0, 25.0, 12.0)]
        assert np.all(np.diff(rates) > 0)

    def test_is_zero_for_a_stationary_object(self):
        assert angular_velocity_deg_per_ms(10.0, 50.0, 0.0) == 0.0


class TestScaleInvariance:
    def test_equal_ratios_give_identical_angular_profiles(self):
        """Two stimuli with the same l/|v| look identical, whatever their absolute size.

        This is why the looming literature sweeps that ratio rather than size or speed.
        """
        a = LoomingTrajectory(half_size_over_speed_ms=20.0, radius_mm=10.0,
                              start_distance_mm=10.0 + 0.5 * 600.0)
        b = LoomingTrajectory(half_size_over_speed_ms=20.0, radius_mm=40.0,
                              start_distance_mm=40.0 + 2.0 * 600.0)
        for t in (0.0, 100.0, 300.0, 500.0, 590.0):
            sa, sb = a.state_at(t), b.state_at(t)
            theta_a = angular_size_deg(sa.radius_mm, sa.distance_mm)
            theta_b = angular_size_deg(sb.radius_mm, sb.distance_mm)
            assert theta_a == pytest.approx(theta_b, rel=1e-9)

    def test_collision_time_is_consistent(self):
        trajectory = LoomingTrajectory(
            half_size_over_speed_ms=20.0, radius_mm=10.0, start_distance_mm=310.0
        )
        assert trajectory.speed_mm_per_ms == pytest.approx(0.5)
        assert trajectory.collision_time_ms == pytest.approx(600.0)
        assert trajectory.state_at(600.0).distance_mm == pytest.approx(10.0)


def encoder(**kwargs) -> AnalyticLoomingEncoder:
    return AnalyticLoomingEncoder({1: "L", 2: "R"}, {3: "L", 4: "R"}, **kwargs)


def scene(distance: float, *, speed: float = 0.5, time_ms: float = 1000.0) -> SceneState:
    return SceneState(
        time_ms=time_ms, distance_mm=distance, radius_mm=10.0,
        closing_speed_mm_per_ms=speed,
    )


class TestChannels:
    def test_lc4_is_linear_in_angular_velocity(self):
        """Ache et al. 2019: LC4 carries a linear function of angular velocity."""
        enc = encoder(tuning=LoomingTuning(visual_latency_ms=0.0, lc4_max_hz=1e9))
        samples = []
        for distance in (400.0, 200.0, 120.0, 80.0, 50.0):
            _, theta_dot, lc4, _ = enc.channel_rates(scene(distance))
            samples.append((theta_dot, lc4))
        ratios = [rate / dot for dot, rate in samples if dot > 0]
        assert np.allclose(ratios, ratios[0], rtol=1e-9)

    def test_lc4_saturates_at_its_ceiling(self):
        enc = encoder(tuning=LoomingTuning(visual_latency_ms=0.0, lc4_max_hz=100.0))
        _, _, lc4, _ = enc.channel_rates(scene(10.5, speed=5.0))
        assert lc4 <= 100.0

    def test_lplc2_is_gaussian_and_peaks_at_its_preferred_size(self):
        """Ache et al. 2019: LPLC2 carries a Gaussian function of angular size."""
        tuning = LoomingTuning(visual_latency_ms=0.0, lplc2_peak_deg=65.0,
                               lplc2_width_deg=25.0)
        enc = encoder(tuning=tuning)

        best, best_rate = None, -1.0
        for distance in np.linspace(200.0, 11.0, 400):
            theta, _, _, lplc2 = enc.channel_rates(scene(float(distance)))
            if lplc2 > best_rate:
                best, best_rate = theta, lplc2
        assert best == pytest.approx(tuning.lplc2_peak_deg, abs=2.0)
        assert best_rate == pytest.approx(tuning.lplc2_max_hz, rel=0.02)

    def test_lplc2_falls_off_symmetrically_in_size(self):
        tuning = LoomingTuning(visual_latency_ms=0.0, lplc2_peak_deg=65.0,
                               lplc2_width_deg=25.0)
        enc = encoder(tuning=tuning)
        peak = tuning.lplc2_max_hz
        one_sigma = peak * math.exp(-0.5)

        # theta = peak +/- width, solved back to a distance.
        for theta in (65.0 - 25.0, 65.0 + 25.0):
            distance = 10.0 / math.tan(math.radians(theta / 2.0))
            _, _, _, lplc2 = enc.channel_rates(scene(distance))
            assert lplc2 == pytest.approx(one_sigma, rel=0.02)

    def test_a_stationary_object_drives_nothing(self):
        enc = encoder(tuning=LoomingTuning(visual_latency_ms=0.0))
        _, _, lc4, lplc2 = enc.channel_rates(scene(20.0, speed=0.0))
        assert lc4 == 0.0
        assert lplc2 == 0.0

    def test_visual_latency_delays_the_response(self):
        enc = encoder(tuning=LoomingTuning(visual_latency_ms=25.0))
        assert enc.channel_rates(scene(300.0, time_ms=10.0)) == (0.0, 0.0, 0.0, 0.0)
        assert enc.channel_rates(scene(300.0, time_ms=30.0))[3] > 0.0

    def test_encode_covers_declared_targets(self):
        enc = encoder(tuning=LoomingTuning(visual_latency_ms=0.0))
        drive = enc.encode(scene(30.0))
        assert set(drive) <= set(enc.target_ids())
        assert enc.target_ids() == [1, 2, 3, 4]
        assert all(rate > 0 for rate in drive.values())


class TestOmmatidialStub:
    def test_construction_raises_with_a_useful_message(self):
        with pytest.raises(NotImplementedError, match="AnalyticLoomingEncoder"):
            OmmatidialEncoder()

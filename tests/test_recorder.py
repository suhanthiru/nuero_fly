"""The activity kernel and the recording it produces.

The kernel is what the viewer actually displays, so its normalisation is a claim about what
"bright" means on screen. These pin that claim: a neuron firing at the reference rate reads
1.0, decay follows the stated time constant, and silence stays exactly zero.
"""

from __future__ import annotations

import numpy as np
import pytest

from sim.recorder import (
    ACTIVITY_TAU_MS,
    REFERENCE_RATE_HZ,
    Recording,
    exponential_activity,
)


class TestActivityKernel:
    def test_silence_stays_zero(self):
        raster = np.zeros((500, 3), dtype=bool)
        assert np.all(exponential_activity(raster, dt_ms=0.1) == 0.0)

    def test_single_spike_decays_with_the_stated_time_constant(self):
        raster = np.zeros((1000, 1), dtype=bool)
        raster[100, 0] = True
        activity = exponential_activity(raster, dt_ms=0.1)

        peak = activity[100, 0]
        assert peak > 0
        # One time constant later the signal must be down by 1/e.
        later = 100 + int(ACTIVITY_TAU_MS / 0.1)
        assert activity[later, 0] == pytest.approx(peak * np.exp(-1.0), rel=0.02)

    def test_is_causal(self):
        """Nothing may appear before the spike that caused it."""
        raster = np.zeros((400, 1), dtype=bool)
        raster[200, 0] = True
        activity = exponential_activity(raster, dt_ms=0.1)
        assert np.all(activity[:200, 0] == 0.0)
        assert activity[200, 0] > 0.0

    def test_reference_rate_reads_one(self):
        """A neuron firing steadily at the reference rate settles at 1.0.

        This is the normalisation the whole display rests on: full brightness means the
        reference rate, not "the brightest thing in this particular run".
        """
        dt = 0.1
        duration_ms = 2000.0
        steps = int(duration_ms / dt)
        period = int(round(1000.0 / REFERENCE_RATE_HZ / dt))
        raster = np.zeros((steps, 1), dtype=bool)
        raster[::period, 0] = True

        activity = exponential_activity(raster, dt_ms=dt)
        settled = activity[steps // 2 :, 0].mean()
        assert settled == pytest.approx(1.0, rel=0.05)

    def test_scales_linearly_with_rate(self):
        dt = 0.1
        steps = 20000
        values = []
        for rate in (50.0, 100.0, 150.0):
            period = int(round(1000.0 / rate / dt))
            raster = np.zeros((steps, 1), dtype=bool)
            raster[::period, 0] = True
            values.append(exponential_activity(raster, dt_ms=dt)[steps // 2 :, 0].mean())
        assert values[1] / values[0] == pytest.approx(2.0, rel=0.05)
        assert values[2] / values[0] == pytest.approx(3.0, rel=0.05)

    def test_empty_raster_is_handled(self):
        assert exponential_activity(np.zeros((0, 0), dtype=bool), dt_ms=0.1).size == 0


class TestRecording:
    def make(self) -> Recording:
        activity = np.zeros((100, 4), dtype=np.float32)
        activity[:, 0] = 0.5   # LC4
        activity[:, 1] = 1.0   # LC4
        activity[:, 2] = 0.25  # DNp01
        return Recording(
            dt_ms=0.1,
            duration_ms=10.0,
            activity=activity,
            body_ids=np.array([1, 2, 3, 4]),
            cell_types=["LC4", "LC4", "DNp01", "DNp01"],
            spike_counts=np.array([5, 9, 1, 0]),
        )

    def test_aggregate_is_the_mean_over_a_cell_type(self):
        aggregate = self.make().aggregate_by_type()
        assert set(aggregate) == {"LC4", "DNp01"}
        assert aggregate["LC4"][0] == pytest.approx(0.75)
        assert aggregate["DNp01"][0] == pytest.approx(0.125)

    def test_frame_indices_decimate_to_the_render_rate(self):
        recording = self.make()
        # 0.1 ms steps at 20 Hz means one frame every 500 steps.
        assert recording.frame_indices(20.0).tolist() == [0]
        assert recording.frame_indices(2000.0).tolist()[:3] == [0, 5, 10]

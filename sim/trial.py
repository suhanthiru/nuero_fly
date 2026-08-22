"""One looming trial, end to end: geometry -> drive -> spikes -> smoothed activity.

Composes the encoder, neuron model, decoder and recorder into the single call both the
live demo and the batch sweeps make. Kept in ``sim`` because every piece it touches is a
simulation concern; the set of cell types to record is passed in rather than imported, so
this does not have to know anything about how they are drawn.

``sim`` imports nothing from ``viz`` or ``world``, and that holds here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from data.cell_types import ids_for
from .decoder import TakeoffEvent, decode
from .encoders.analytic import AnalyticLoomingEncoder, LoomingTrajectory, LoomingTuning
from .lif import LIF, LIFParams
from .neuron import StimulusSpec
from .recorder import Recording, SceneTrace, record


@dataclass(frozen=True)
class TrialSpec:
    """Everything that defines a trial, so a run can be reproduced from this alone."""

    ratio_ms: float = 40.0          # l/|v|
    duration_ms: float = 800.0
    collision_ms: float = 600.0
    radius_mm: float = 10.0
    azimuth_deg: float = 0.0
    gain_scale: float = 0.03        # see note below
    seed: int = 0

    def trajectory(self) -> LoomingTrajectory:
        speed = self.radius_mm / self.ratio_ms
        return LoomingTrajectory(
            half_size_over_speed_ms=self.ratio_ms,
            radius_mm=self.radius_mm,
            start_distance_mm=self.radius_mm + speed * self.collision_ms,
            azimuth_deg=self.azimuth_deg,
        )


# Why the default gain is 0.03 and not 1.0.
#
# The Phase 2 sweep found the giant fiber saturates at the nominal gain: it fires ~150 times
# per trial starting ~35 ms in, and escape timing stops tracking the stimulus at all
# (8 ms of variation across an eightfold change in looming speed). At ~0.03 the timing does
# track l/|v| in the direction an angular-size threshold predicts.
#
# This default is therefore chosen so the demo shows a circuit doing something legible,
# NOT because 0.03 is correct - nothing in the data selects it. Any figure produced from a
# default-gain run inherits that choice and must say so. The honest result of Phase 2 is the
# whole gain sweep, not any single row of it.


def run_looming_trial(
    connectome,
    *,
    spec: TrialSpec | None = None,
    record_types: list[str],
    params: LIFParams | None = None,
    model: LIF | None = None,
) -> tuple[Recording, TakeoffEvent]:
    """Simulate one approach and return the recording plus the decoded outcome."""
    spec = spec or TrialSpec()
    params = params or LIFParams()
    model = model or LIF(params)
    n_steps = int(round(spec.duration_ms / params.dt))

    tuning = LoomingTuning(
        lc4_gain_hz_per_deg_per_ms=25.0 * spec.gain_scale,
        lc4_max_hz=150.0 * spec.gain_scale,
        lplc2_max_hz=150.0 * spec.gain_scale,
        max_rate_hz=max(150.0 * spec.gain_scale, 1e-6),
    )
    encoder = AnalyticLoomingEncoder.from_connectome(connectome, tuning=tuning)
    targets = encoder.target_ids()
    target_idx = connectome.indices_of(np.asarray(targets))

    trajectory = spec.trajectory()
    is_lc4 = np.array([b in encoder.lc4_ids for b in targets])
    is_lplc2 = np.array([b in encoder.lplc2_ids for b in targets])

    schedule = np.zeros((n_steps, len(targets)), dtype=np.float32)
    theta = np.zeros(n_steps, dtype=np.float32)
    theta_dot = np.zeros(n_steps, dtype=np.float32)
    distance = np.zeros(n_steps, dtype=np.float32)
    for step in range(n_steps):
        scene = trajectory.state_at(step * params.dt)
        th, thd, lc4_hz, lplc2_hz = encoder.channel_rates(scene)
        theta[step] = th
        theta_dot[step] = thd
        distance[step] = scene.distance_mm
        schedule[step, is_lc4] = lc4_hz
        schedule[step, is_lplc2] = lplc2_hz

    # Record every cell type the caller wants to display, in a stable order.
    body_ids: list[int] = []
    cell_types: list[str] = []
    for cell_type in record_types:
        for body_id in ids_for(connectome, cell_type):
            body_ids.append(int(body_id))
            cell_types.append(cell_type)
    record_idx = connectome.indices_of(np.asarray(body_ids))

    result = model.simulate(
        connectome.weights,
        StimulusSpec(poisson_targets=target_idx, rate_schedule=schedule),
        duration_ms=spec.duration_ms,
        seed=spec.seed,
        record=record_idx,
    )

    def indices_for(pattern: str) -> np.ndarray:
        return connectome.indices_of(ids_for(connectome, pattern))

    event = decode(
        result.spike_times,
        gf_indices=indices_for("DNp01"),
        ttm_indices=indices_for("TTMn"),
        collision_ms=spec.collision_ms,
    )

    recording = record(
        result,
        body_ids=np.asarray(body_ids),
        cell_types=cell_types,
        scene=SceneTrace(
            theta_deg=theta,
            theta_dot_deg_per_ms=theta_dot,
            distance_mm=distance,
            collision_ms=spec.collision_ms,
            ratio_ms=spec.ratio_ms,
        ),
        events={
            "mode": event.mode.value,
            "gf_spike_ms": event.gf_spike_ms,
            "ttm_spike_ms": event.ttm_spike_ms,
            "latency_to_collision_ms": event.latency_to_collision_ms,
            "gf_spike_count": event.gf_spike_count,
            "ttm_spike_count": event.ttm_spike_count,
        },
        extra_meta={"trial": spec.__dict__, "tuning": tuning.__dict__},
    )
    return recording, event

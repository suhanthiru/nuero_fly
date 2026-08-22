"""Run output: spike trains, smoothed activity, scene traces and provenance.

Two jobs.

**Turning spikes into something displayable.** Spikes are instantaneous and the renderer
runs at 20 Hz, so raw spikes would alias badly - most of them fall between frames. Instead
each spike train is convolved with a causal exponential kernel, which is what a calcium
indicator does to the same signal. The result is directly comparable to published fly
imaging, which is the point: the visualisation should show the same quantity an experiment
would have measured.

**Being the substrate the lesion sweep consumes.** Every run carries the metadata needed to
reproduce it - dataset version, seed, model parameters, git SHA - and serialises to a form
that a sweep over hundreds of cell types can write in parallel and read back.

Normalisation deserves a note, because it decides what "bright" means. Intensity is scaled
so that **1.0 is a neuron firing steadily at the reference rate** (the encoder's 150 Hz
ceiling), not so that each neuron uses its own full range. Per-neuron normalisation would
make a silent cell and a barely-active cell look identical, which is exactly the comparison
the viewer exists to support.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: Time constant of the smoothing kernel. Chosen to sit in the range of a slow calcium
#: indicator, so the rendered signal is comparable to published imaging.
ACTIVITY_TAU_MS = 50.0

#: Firing rate that maps to full brightness.
REFERENCE_RATE_HZ = 150.0


def exponential_activity(
    raster: np.ndarray,
    *,
    dt_ms: float,
    tau_ms: float = ACTIVITY_TAU_MS,
    reference_rate_hz: float = REFERENCE_RATE_HZ,
) -> np.ndarray:
    """Convolve a ``(n_steps, n_neurons)`` spike raster with a causal exponential.

    Implemented as the equivalent one-pole recursion ``a[t] = a[t-1] * decay + spike[t]``
    rather than an explicit convolution, which would cost O(n_steps * kernel) for no gain.

    Scaled so a neuron firing steadily at ``reference_rate_hz`` reads 1.0: in the continuous
    limit a Poisson train at rate r settles at ``r * tau``, so that is the divisor.
    """
    if raster.size == 0:
        return np.zeros_like(raster, dtype=np.float32)

    decay = float(np.exp(-dt_ms / tau_ms))
    steady_state = max(reference_rate_hz * tau_ms / 1000.0, 1e-9)

    activity = np.empty(raster.shape, dtype=np.float32)
    running = np.zeros(raster.shape[1], dtype=np.float32)
    spikes = raster.astype(np.float32)
    for step in range(raster.shape[0]):
        running = running * decay + spikes[step]
        activity[step] = running
    return activity / steady_state


@dataclass(frozen=True)
class SceneTrace:
    """What the world was doing, sampled on the simulation clock."""

    theta_deg: np.ndarray
    theta_dot_deg_per_ms: np.ndarray
    distance_mm: np.ndarray
    collision_ms: float
    ratio_ms: float


@dataclass
class Recording:
    """One trial, in a form both the viewer and a batch sweep can consume."""

    dt_ms: float
    duration_ms: float
    #: (n_steps, n_recorded) smoothed intensity, 1.0 == REFERENCE_RATE_HZ
    activity: np.ndarray
    #: body ids, in the column order of `activity`
    body_ids: np.ndarray
    cell_types: list[str]
    spike_counts: np.ndarray
    scene: SceneTrace | None = None
    events: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return int(self.activity.shape[0])

    def frame_indices(self, render_hz: float) -> np.ndarray:
        """Step indices to sample for a given render rate.

        The simulation runs at 0.1 ms and the display at 20 Hz, so this decimates rather
        than interpolating - the activity kernel has already done the smoothing that makes
        decimation safe.
        """
        stride = max(1, int(round(1000.0 / render_hz / self.dt_ms)))
        return np.arange(0, self.n_steps, stride)

    def aggregate_by_type(self) -> dict[str, np.ndarray]:
        """Mean intensity per cell type, for the trace panel."""
        out: dict[str, np.ndarray] = {}
        types = np.asarray(self.cell_types)
        for cell_type in sorted(set(self.cell_types)):
            if not cell_type:
                continue
            out[cell_type] = self.activity[:, types == cell_type].mean(axis=1)
        return out

    # -- persistence ---------------------------------------------------------------
    def save(self, path: Path) -> None:
        """Write to .npz plus a sidecar .json of metadata.

        Arrays go to npz because a sweep produces many of them; metadata stays readable so
        a failed run can be diagnosed without loading anything.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {
            "activity": self.activity,
            "body_ids": self.body_ids,
            "spike_counts": self.spike_counts,
        }
        if self.scene is not None:
            arrays.update(
                theta_deg=self.scene.theta_deg,
                theta_dot=self.scene.theta_dot_deg_per_ms,
                distance_mm=self.scene.distance_mm,
            )
        np.savez_compressed(path.with_suffix(".npz"), **arrays)
        path.with_suffix(".json").write_text(json.dumps({
            "dt_ms": self.dt_ms,
            "duration_ms": self.duration_ms,
            "cell_types": self.cell_types,
            "events": self.events,
            "meta": self.meta,
            "scene": None if self.scene is None else {
                "collision_ms": self.scene.collision_ms,
                "ratio_ms": self.scene.ratio_ms,
            },
        }, indent=2, default=str))


def record(
    result,
    *,
    body_ids: np.ndarray,
    cell_types: list[str],
    scene: SceneTrace | None = None,
    events: dict[str, Any] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> Recording:
    """Build a :class:`Recording` from a :class:`sim.neuron.SimulationResult`."""
    activity = exponential_activity(result.raster, dt_ms=result.dt_ms)
    meta = dict(result.meta)
    meta.update(extra_meta or {})
    meta["activity_tau_ms"] = ACTIVITY_TAU_MS
    meta["reference_rate_hz"] = REFERENCE_RATE_HZ
    return Recording(
        dt_ms=result.dt_ms,
        duration_ms=result.duration_ms,
        activity=activity,
        body_ids=np.asarray(body_ids),
        cell_types=list(cell_types),
        spike_counts=result.spike_counts[result.record_indices],
        scene=scene,
        events=events or {},
        meta=meta,
    )

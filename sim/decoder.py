"""Motor readout: descending and motor neuron spikes -> a takeoff event.

Two things make this readout defensible rather than arbitrary.

First, **a single giant fiber spike is sufficient to drive a takeoff** (von Reyn et al.,
Nat Neurosci 2014). So the GF threshold is one spike, not a tuned burst count - there is
nothing to tune.

Second, the escape has two modes, and the distinction is behavioural rather than neural:
Drosophila take off in a *short* mode (<7 ms from the start of wing motion to loss of
tarsal contact) or a *long* mode (>=7 ms, wings raised first). The GF is required for short
takeoffs and contributes to some long ones. We cannot measure wing motion here, so we
classify by **which pathway fired**, and say so plainly rather than pretending to measure
a duration:

* GF (DNp01) fired  -> short-mode escape
* TTMn fired without a preceding GF spike, or only the slower descending neurons fired
  -> long-mode escape
* nothing fired -> no escape

This is a stated mapping from circuit activity onto a behavioural category, not a
measurement of takeoff duration. Any claim about mode fractions inherits that assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class EscapeMode(str, Enum):
    NONE = "none"
    SHORT = "short"       # giant-fiber mediated
    LONG = "long"         # driven without a giant fiber spike


@dataclass(frozen=True)
class TakeoffEvent:
    """What the circuit did, and when."""

    mode: EscapeMode
    #: First giant fiber spike, ms from trial start. None if it never fired.
    gf_spike_ms: float | None
    #: First tergotrochanteral motor neuron spike, ms from trial start.
    ttm_spike_ms: float | None
    #: Takeoff time relative to the moment of collision. Negative means the fly went
    #: before contact, which is the only case that counts as an escape.
    latency_to_collision_ms: float | None
    gf_spike_count: int
    ttm_spike_count: int
    dn_spike_counts: dict[str, int]

    @property
    def escaped(self) -> bool:
        return (
            self.mode is not EscapeMode.NONE
            and self.latency_to_collision_ms is not None
            and self.latency_to_collision_ms < 0.0
        )


def _first(times: np.ndarray | None) -> float | None:
    if times is None or len(times) == 0:
        return None
    return float(np.min(times))


def decode(
    spike_times: dict[int, np.ndarray],
    *,
    gf_indices: np.ndarray,
    ttm_indices: np.ndarray,
    collision_ms: float,
    dn_indices: dict[str, np.ndarray] | None = None,
) -> TakeoffEvent:
    """Classify one trial from recorded spike trains.

    ``spike_times`` maps dense neuron index -> spike times in ms, as returned by the neuron
    model when those neurons are in its record set.
    """

    def gather(indices: np.ndarray) -> np.ndarray:
        collected = [spike_times[int(i)] for i in indices if int(i) in spike_times]
        return np.concatenate(collected) if collected else np.zeros(0)

    gf_times = gather(gf_indices)
    ttm_times = gather(ttm_indices)
    gf_first = _first(gf_times)
    ttm_first = _first(ttm_times)

    dn_counts: dict[str, int] = {}
    for name, indices in (dn_indices or {}).items():
        dn_counts[name] = int(len(gather(indices)))

    # A single GF spike suffices for a takeoff, so its arrival is the takeoff time for the
    # short mode. Otherwise the readout falls to the motor neuron itself.
    if gf_first is not None:
        mode = EscapeMode.SHORT
        takeoff = gf_first
    elif ttm_first is not None:
        mode = EscapeMode.LONG
        takeoff = ttm_first
    else:
        mode = EscapeMode.NONE
        takeoff = None

    return TakeoffEvent(
        mode=mode,
        gf_spike_ms=gf_first,
        ttm_spike_ms=ttm_first,
        latency_to_collision_ms=None if takeoff is None else takeoff - collision_ms,
        gf_spike_count=int(len(gf_times)),
        ttm_spike_count=int(len(ttm_times)),
        dn_spike_counts=dn_counts,
    )

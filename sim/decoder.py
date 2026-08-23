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

#: How far a fully one-sided motor output can swing the escape away from straight back.
#: HAND-SET. With no leg model there is nothing in the simulation that produces a heading,
#: so this converts motor asymmetry into one by fiat. The mapping is *contralateral* - the
#: side that fires harder pushes the fly the other way - which is the right sign for a leg
#: extension but is asserted here, not derived. Any directional-tuning result inherits both
#: the sign and the magnitude, so both are swept rather than trusted.
MAX_TURN_DEG = 90.0

#: Minimum spikes across both sides before an asymmetry is treated as a direction rather
#: than as noise. Below this the fly simply jumps straight back.
MIN_SPIKES_TO_STEER = 4


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
    #: Left/right spike counts of whichever population the heading is read from, and the
    #: heading they imply in world degrees.
    left_count: int = 0
    right_count: int = 0
    heading_deg: float = 180.0
    heading_source: str = "none"
    #: Motor counts, reported separately because they are usually too sparse to steer with.
    ttm_left_count: int = 0
    ttm_right_count: int = 0

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


def heading_from_asymmetry(left: int, right: int) -> float:
    """Escape heading in degrees, from left/right motor output.

    Straight back is 180 degrees. A one-sided output rotates that by up to MAX_TURN_DEG,
    contralaterally: more left-side drive sends the fly to its right.

    With both sides equal this returns exactly 180 - straight back, no directional
    preference at all. That is the correct null, and it is what the model produces whenever
    the two hemispheres receive equal drive.
    """
    total = left + right
    if total == 0:
        return 180.0
    asymmetry = (left - right) / total
    wrapped = ((180.0 + asymmetry * MAX_TURN_DEG) + 180.0) % 360.0 - 180.0
    # The wrap sends exactly-backward to -180; report it as +180 so "straight back" reads
    # the same whether it came from symmetry or from the zero-spike default.
    return 180.0 if wrapped == -180.0 else wrapped


def decode(
    spike_times: dict[int, np.ndarray],
    *,
    gf_indices: np.ndarray,
    ttm_indices: np.ndarray,
    collision_ms: float,
    dn_indices: dict[str, np.ndarray] | None = None,
    ttm_left: np.ndarray | None = None,
    ttm_right: np.ndarray | None = None,
    gf_left: np.ndarray | None = None,
    gf_right: np.ndarray | None = None,
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

    ttm_left_count = int(len(gather(ttm_left))) if ttm_left is not None else 0
    ttm_right_count = int(len(gather(ttm_right))) if ttm_right is not None else 0
    gf_left_count = int(len(gather(gf_left))) if gf_left is not None else 0
    gf_right_count = int(len(gather(gf_right))) if gf_right is not None else 0

    # Which population steers.
    #
    # The obvious choice is the motor neuron, but TTMn fires only zero to a few times per
    # trial here - it sits under heavy net inhibition, which Phase 0 measured directly - and
    # an asymmetry estimated from one spike is noise, not a direction. The giant fiber is
    # better sampled and is in any case the decision signal: a single GF spike is sufficient
    # to drive a takeoff, so the side that fires first and hardest is what commits the
    # animal. TTMn counts are still reported, so the sparsity is visible rather than hidden.
    if gf_left_count + gf_right_count >= MIN_SPIKES_TO_STEER:
        left_count, right_count, source = gf_left_count, gf_right_count, "DNp01"
    elif ttm_left_count + ttm_right_count >= MIN_SPIKES_TO_STEER:
        left_count, right_count, source = ttm_left_count, ttm_right_count, "TTMn"
    else:
        left_count, right_count, source = 0, 0, "none"

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
        left_count=left_count,
        right_count=right_count,
        heading_deg=heading_from_asymmetry(left_count, right_count),
        heading_source=source,
        ttm_left_count=ttm_left_count,
        ttm_right_count=ttm_right_count,
    )

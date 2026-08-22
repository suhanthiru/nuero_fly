"""Cell type and neuropil lookups, and the escape-pathway registry.

The build spec supplied a candidate list of escape cell types and instructed us to verify
it against the annotations rather than trust it. Verification against MaleCNS v1.0 found:

* LC4 (126), LC6 (124), LC22 (73), LPLC2 (185) - all present, all bilateral.
* DNp01 (2), DNp02 (2), DNp04 (2), DNp11 (2) - one per hemisphere each, as expected for
  large individually-identified descending neurons. DNp01 carries the instance label
  ``DNp01(GF)_R`` / ``_L`` and the hemibrain type ``Giant Fiber``.
* TTM is annotated as ``TTMn`` (the tergotrochanteral motor neuron itself), not "TTM".
  A separate ``STTMm`` type exists and is *not* the same cell.
* The spec's descending list is a subset: MaleCNS annotates 480 distinct descending types
  across 1314 bodies. DNp02/04/11 are kept because the spec named them, but the slow-mode
  pathway is not established to be limited to those and the lesion sweep should not assume
  it is.
"""

from __future__ import annotations

import fnmatch
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:  # avoid a circular import at runtime
    from .loader import Connectome

# Ordered by position in the circuit, which is also the order the palette follows:
# visual input -> descending -> motor.
ESCAPE_PATHWAY: dict[str, tuple[str, ...]] = {
    # Looming-sensitive visual projection neurons. These are where the analytic encoder
    # injects current, and therefore the boundary of what v1 actually simulates.
    "visual_projection": ("LC4", "LC6", "LC22", "LPLC2"),
    # DNp01 is the giant fiber. The rest are the descending neurons the spec implicated in
    # the slower, directionally-tuned escape mode.
    "descending": ("DNp01", "DNp02", "DNp04", "DNp11"),
    # TTMn drives the mesothoracic leg extension of the short-mode escape. DLMn types drive
    # the wing depression associated with the longer mode. Glob because MaleCNS splits DLMn
    # into muscle-specific subtypes ("DLMn c-f" and friends).
    "motor": ("TTMn", "DLMn*"),
}

# Convenience: every pattern, flattened, in circuit order.
ESCAPE_PATTERNS: tuple[str, ...] = tuple(
    pattern for patterns in ESCAPE_PATHWAY.values() for pattern in patterns
)


def matching_types(connectome: "Connectome", patterns) -> list[str]:
    """Cell type labels present in the connectome that match any fnmatch pattern."""
    if isinstance(patterns, str):
        patterns = [patterns]
    present = pd.unique(connectome.annotations["cell_type"])
    present = [t for t in present if t]
    out: list[str] = []
    for pattern in patterns:
        hits = sorted(fnmatch.filter(present, pattern)) if "*" in pattern or "?" in pattern \
            else ([pattern] if pattern in present else [])
        for hit in hits:
            if hit not in out:
                out.append(hit)
    return out


def ids_for(connectome: "Connectome", patterns, *, side: str | None = None) -> np.ndarray:
    """Body ids whose cell type matches any pattern, optionally restricted to one side."""
    types = matching_types(connectome, patterns)
    frame = connectome.annotations
    mask = frame["cell_type"].isin(types)
    if side is not None:
        mask &= frame["side"] == side
    return np.asarray(frame.index[mask])


def indices_for(connectome: "Connectome", patterns, *, side: str | None = None) -> np.ndarray:
    """Dense matrix indices for the same selection."""
    return connectome.indices_of(ids_for(connectome, patterns, side=side))


def types_of(connectome: "Connectome", body_ids) -> pd.Series:
    """Cell type label for each body id."""
    return connectome.annotations.loc[np.asarray(body_ids), "cell_type"]


def escape_report(connectome: "Connectome") -> pd.DataFrame:
    """Per-escape-cell-type counts, for the Phase 0 exit criterion.

    ``syn_in_raw`` / ``syn_out_raw`` come from the unfiltered release and are the columns to
    compare against published figures. ``syn_in_graph`` / ``syn_out_graph`` are what actually
    survives into the simulated matrix after the synapse threshold and after modulatory
    edges are zeroed; the gap between the two is expected and is itself worth reading.
    """
    frame = connectome.annotations
    rows = []
    for stage, patterns in ESCAPE_PATHWAY.items():
        for cell_type in matching_types(connectome, patterns):
            mask = (frame["cell_type"] == cell_type).to_numpy()
            idx = np.flatnonzero(mask)
            sides = frame["side"].to_numpy()[idx]
            rows.append(
                {
                    "stage": stage,
                    "cell_type": cell_type,
                    "n": idx.size,
                    "n_L": int((sides == "L").sum()),
                    "n_R": int((sides == "R").sum()),
                    "n_other": int(((sides != "L") & (sides != "R")).sum()),
                    "syn_in_raw": float(connectome.raw_in_synapses[idx].sum()),
                    "syn_out_raw": float(connectome.raw_out_synapses[idx].sum()),
                    "syn_in_graph": float(
                        np.abs(connectome.weights[idx].data).sum()
                    ),
                    "syn_out_graph": float(
                        np.abs(connectome.weights[:, idx].data).sum()
                    ),
                }
            )
    missing = [
        pattern
        for pattern in ESCAPE_PATTERNS
        if not matching_types(connectome, pattern)
    ]
    report = pd.DataFrame(rows)
    report.attrs["missing_patterns"] = missing
    return report


def connection_between(
    connectome: "Connectome", pre_patterns, post_patterns
) -> tuple[float, int]:
    """Total signed weight and edge count from one cell-type group to another.

    Used to assert that DNp01 -> TTMn connectivity is non-zero, which is the whole reason
    the project moved to a dataset with an intact neck connective.
    """
    pre = connectome.indices_of(ids_for(connectome, pre_patterns))
    post = connectome.indices_of(ids_for(connectome, post_patterns))
    if pre.size == 0 or post.size == 0:
        return 0.0, 0
    block = connectome.weights[post][:, pre]
    return float(block.sum()), int(block.nnz)

"""Lesioning and subgraph extraction.

Built in Phase 0 rather than retrofitted, because the lesion sweep - silence a cell type,
re-run, measure the behavioural delta, across every type - is the main planned use of this
codebase and it is embarrassingly parallel. It needs a clean hook, deterministic runs and
structured output, and retrofitting any of those later is painful.

Two ways to silence, which are not equivalent:

``silence_mask``
    The preferred one. Returns a boolean mask the neuron model uses to suppress spiking.
    The neuron still integrates its inputs, it simply never fires. This matches what an
    optogenetic or genetic silencing experiment does, costs nothing to apply, and needs no
    matrix rebuild - so a sweep over hundreds of cell types reuses one loaded connectome.

``silence_structural``
    Rebuilds the matrix with the cell type's outgoing edges removed. Useful when something
    downstream wants a genuinely modified graph rather than a runtime mask. More expensive,
    and it discards the silenced cells' own subthreshold dynamics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

from .cell_types import ids_for

if TYPE_CHECKING:
    from .loader import Connectome


def silence_mask(connectome: "Connectome", patterns) -> np.ndarray:
    """Boolean mask, True where the neuron should be prevented from spiking."""
    mask = np.zeros(connectome.n_neurons, dtype=bool)
    mask[connectome.indices_of(ids_for(connectome, patterns))] = True
    return mask


def silence_structural(connectome: "Connectome", patterns) -> "Connectome":
    """A copy of the connectome with the matched cell types' outgoing edges removed."""
    from .loader import Connectome  # local import keeps this module import-cycle free

    victims = connectome.indices_of(ids_for(connectome, patterns))
    # weights[post, pre]: a neuron's outgoing edges are its *column*.
    weights = connectome.weights.tolil(copy=True)
    weights[:, victims] = 0
    weights = weights.tocsr()
    weights.eliminate_zeros()

    meta = dict(connectome.meta)
    meta["lesion"] = {
        "mode": "structural",
        "patterns": list(patterns) if not isinstance(patterns, str) else [patterns],
        "n_neurons_silenced": int(victims.size),
    }
    return Connectome(
        weights=weights,
        ids=connectome.ids,
        annotations=connectome.annotations,
        raw_in_synapses=connectome.raw_in_synapses,
        raw_out_synapses=connectome.raw_out_synapses,
        meta=meta,
    )


def subgraph(connectome: "Connectome", body_ids) -> "Connectome":
    """Extract an induced subgraph over the given body ids, preserving id order."""
    from .loader import Connectome

    keep = np.sort(connectome.indices_of(body_ids))
    weights = connectome.weights[keep][:, keep].tocsr()
    meta = dict(connectome.meta)
    meta["subgraph_of"] = meta.get("dataset")
    meta["n_neurons"] = int(keep.size)
    meta["n_edges_final"] = int(weights.nnz)
    return Connectome(
        weights=weights,
        ids=connectome.ids[keep],
        annotations=connectome.annotations.iloc[keep],
        raw_in_synapses=connectome.raw_in_synapses[keep],
        raw_out_synapses=connectome.raw_out_synapses[keep],
        meta=meta,
    )


def neighbourhood(connectome: "Connectome", body_ids, *, hops: int = 1) -> np.ndarray:
    """Body ids reachable within `hops` synapses of the seed set, in either direction."""
    frontier = set(connectome.indices_of(body_ids).tolist())
    seen = set(frontier)
    weights = connectome.weights
    transposed = weights.T.tocsr()
    for _ in range(hops):
        nxt: set[int] = set()
        for idx in frontier:
            nxt.update(weights.indices[weights.indptr[idx] : weights.indptr[idx + 1]].tolist())
            nxt.update(
                transposed.indices[
                    transposed.indptr[idx] : transposed.indptr[idx + 1]
                ].tolist()
            )
        frontier = nxt - seen
        seen |= nxt
        if not frontier:
            break
    return connectome.ids[np.sort(np.fromiter(seen, dtype=int))]

"""Build a signed sparse connectivity matrix from a connectome release.

Orientation, stated once and relied on everywhere::

    weights[post, pre]

Rows are postsynaptic, columns presynaptic. Two things fall out of that choice:

* the per-timestep input current is one CSR matvec, ``I = weights @ spikes``;
* "which neurons drive this one" is a single CSR row slice, which is exactly the query
  the click-to-explain inspector makes.

Weight is ``synapse_count * neurotransmitter_sign``. Synapse count is a *proxy* for
synaptic strength, not a measurement of it - see README.md.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .neurotransmitters import NT_SIGN, sign_for
from .sources import ConnectomeSource, get_source

CACHE = Path(__file__).resolve().parent / "cache"

# Bump when the meaning of the cached arrays changes, so stale caches cannot be read back.
LOADER_VERSION = 1

# Codex publishes its connection table already thresholded at 5 synapses, and the FlyWire
# modelling literature uses the same floor. Applied to both datasets for comparability.
DEFAULT_MIN_SYNAPSES = 5


@dataclass(frozen=True)
class Connectome:
    """A signed, sparse, annotated wiring diagram."""

    weights: sp.csr_matrix          # (N, N) signed, weights[post, pre]
    ids: np.ndarray                 # (N,) body/root id for each dense index
    annotations: pd.DataFrame       # indexed by id, in the same order as `ids`
    # Per-neuron synapse totals from the *unfiltered* release: before the synapse-count
    # threshold and before transmitter sign. Published per-cell-type synapse counts are
    # reported this way, so the Phase 0 exit criterion must compare against these and not
    # against the filtered graph, or it would be comparing two different quantities.
    raw_in_synapses: np.ndarray = field(default_factory=lambda: np.empty(0))
    raw_out_synapses: np.ndarray = field(default_factory=lambda: np.empty(0))
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_neurons(self) -> int:
        return int(self.weights.shape[0])

    @property
    def n_edges(self) -> int:
        return int(self.weights.nnz)

    def index_of(self, body_id: int) -> int:
        """Dense index for one body id. Raises KeyError if absent."""
        slot = int(np.searchsorted(self.ids, body_id))
        if slot >= self.ids.size or self.ids[slot] != body_id:
            raise KeyError(f"body id {body_id} is not in this connectome")
        return slot

    def indices_of(self, body_ids) -> np.ndarray:
        """Dense indices for many body ids, silently dropping any that are absent."""
        wanted = np.asarray(body_ids)
        slot = np.searchsorted(self.ids, wanted)
        slot[slot >= self.ids.size] = 0
        return slot[self.ids[slot] == wanted]

    def upstream(self, body_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Presynaptic partners of one neuron and their signed weights.

        One CSR row slice, because of the weights[post, pre] orientation.
        """
        row = self.weights[self.index_of(body_id)]
        return self.ids[row.indices], row.data

    def downstream(self, body_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Postsynaptic partners. Costs a column slice, so prefer upstream() in hot paths."""
        column = self.weights.T.tocsr()[self.index_of(body_id)]
        return self.ids[column.indices], column.data


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _cache_key(source: ConnectomeSource, min_synapses: int, neurons_only: bool) -> str:
    """Content hash over everything that can change the result.

    The transmitter sign mapping is deliberately part of the key: editing a sign must
    invalidate every cached matrix, or a stale cache would silently outlive the change.
    """
    digest = hashlib.sha256()
    digest.update(f"v{LOADER_VERSION}|{source.key}|{min_synapses}|{neurons_only}".encode())
    digest.update(json.dumps(NT_SIGN, sort_keys=True).encode())
    for path in sorted(Path(source.root).glob("*")):
        stat = path.stat()
        digest.update(f"{path.name}:{stat.st_size}".encode())
    return digest.hexdigest()[:16]


def _dense_index(ids: np.ndarray, body: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Dense index for each body id, plus a mask of which ones actually exist."""
    slot = np.searchsorted(ids, body)
    np.clip(slot, 0, ids.size - 1, out=slot)
    return slot, ids[slot] == body


def build_connectome(
    dataset: str = "malecns-1.0",
    *,
    min_synapses: int = DEFAULT_MIN_SYNAPSES,
    neurons_only: bool = True,
) -> Connectome:
    """Read a release from disk and assemble the signed sparse matrix. No caching."""
    source = get_source(dataset)

    annotations = source.load_annotations()
    annotations = annotations[~annotations.index.duplicated(keep="first")]
    n_bodies = len(annotations)
    if neurons_only:
        annotations = annotations[source.is_neuron(annotations)]
    annotations = annotations.sort_index()
    ids = np.asarray(annotations.index)

    # Stream the edge list. The MaleCNS release is ~152M edges; holding it whole costs
    # 3.6 GB before temporaries, and the mask and searchsorted intermediates that follow
    # exhaust memory. Accumulating per chunk keeps peak usage flat.
    n = ids.size
    raw_in = np.zeros(n)
    raw_out = np.zeros(n)
    n_edges_raw = n_after_threshold = n_on_graph = n_modulatory = 0
    parts_pre: list[np.ndarray] = []
    parts_post: list[np.ndarray] = []
    parts_weight: list[np.ndarray] = []

    for chunk in source.iter_edges():
        n_edges_raw += len(chunk)

        # Sign lookup over the small transmitter vocabulary. strict=True by default, so an
        # unrecognised label raises here rather than quietly becoming zero.
        signs = np.array([sign_for(label) for label in chunk.nt_vocab], dtype=np.int8)

        # Raw per-neuron synapse totals, before threshold and before sign. An edge counts
        # towards a neuron whenever that neuron is an endpoint, regardless of whether the
        # other endpoint survived neuron filtering - that is how published per-cell-type
        # counts are reported, and therefore what the exit criterion must compare against.
        for body, accumulator in ((chunk.post, raw_in), (chunk.pre, raw_out)):
            slot, hit = _dense_index(ids, body)
            accumulator += np.bincount(
                slot[hit], weights=chunk.syn_count[hit], minlength=n
            )

        keep = chunk.syn_count >= min_synapses
        n_after_threshold += int(keep.sum())
        if not keep.any():
            continue
        pre, post = chunk.pre[keep], chunk.post[keep]
        syn_count, nt_code = chunk.syn_count[keep], chunk.nt_code[keep]

        # Discard edges touching a body we are not simulating (glia, orphan fragments,
        # anything filtered out above).
        pre_idx, pre_ok = _dense_index(ids, pre)
        post_idx, post_ok = _dense_index(ids, post)
        on_graph = pre_ok & post_ok
        n_on_graph += int(on_graph.sum())
        if not on_graph.any():
            continue

        weight = syn_count[on_graph].astype(np.float32) * signs[nt_code[on_graph]]

        # Modulatory transmitters carry sign 0 and therefore no current. Dropping them
        # keeps the matrix smaller; the count is recorded so the loss is visible.
        nonzero = weight != 0
        n_modulatory += int((~nonzero).sum())
        parts_pre.append(pre_idx[on_graph][nonzero].astype(np.int32))
        parts_post.append(post_idx[on_graph][nonzero].astype(np.int32))
        parts_weight.append(weight[nonzero])

    empty32 = np.empty(0, dtype=np.int32)
    weights = sp.coo_matrix(
        (
            np.concatenate(parts_weight) if parts_weight else np.empty(0, np.float32),
            (
                np.concatenate(parts_post) if parts_post else empty32,
                np.concatenate(parts_pre) if parts_pre else empty32,
            ),
        ),
        shape=(n, n),
        dtype=np.float32,
    ).tocsr()
    weights.sum_duplicates()

    meta = {
        "dataset": source.key,
        "citation": source.citation,
        "loader_version": LOADER_VERSION,
        "git_sha": _git_sha(),
        "min_synapses": min_synapses,
        "neurons_only": neurons_only,
        "nt_sign": dict(NT_SIGN),
        "orientation": "weights[post, pre]",
        "n_bodies_in_release": n_bodies,
        "n_neurons": int(n),
        "n_edges_raw": n_edges_raw,
        "n_edges_after_threshold": n_after_threshold,
        "n_edges_on_graph": n_on_graph,
        "n_edges_dropped_modulatory": n_modulatory,
        "n_edges_final": int(weights.nnz),
        "n_synapses_final": float(np.abs(weights.data).sum()),
    }
    return Connectome(
        weights=weights,
        ids=ids,
        annotations=annotations,
        raw_in_synapses=raw_in,
        raw_out_synapses=raw_out,
        meta=meta,
    )


def load_connectome(
    dataset: str = "malecns-1.0",
    *,
    min_synapses: int = DEFAULT_MIN_SYNAPSES,
    neurons_only: bool = True,
    use_cache: bool = True,
) -> Connectome:
    """Build, or reload from the .npz cache if inputs and parameters are unchanged."""
    source = get_source(dataset)
    key = _cache_key(source, min_synapses, neurons_only)
    matrix_path = CACHE / f"{source.key}-{key}.npz"
    annotation_path = CACHE / f"{source.key}-{key}-annotations.feather"

    if use_cache and matrix_path.exists() and annotation_path.exists():
        blob = np.load(matrix_path, allow_pickle=False)
        weights = sp.csr_matrix(
            (blob["data"], blob["indices"], blob["indptr"]), shape=tuple(blob["shape"])
        )
        annotations = pd.read_feather(annotation_path).set_index("id")
        meta = json.loads(matrix_path.with_suffix(".json").read_text())
        return Connectome(
            weights=weights,
            ids=blob["ids"],
            annotations=annotations,
            raw_in_synapses=blob["raw_in"],
            raw_out_synapses=blob["raw_out"],
            meta=meta,
        )

    connectome = build_connectome(
        dataset, min_synapses=min_synapses, neurons_only=neurons_only
    )
    if use_cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            matrix_path,
            data=connectome.weights.data,
            indices=connectome.weights.indices,
            indptr=connectome.weights.indptr,
            shape=np.asarray(connectome.weights.shape),
            ids=connectome.ids,
            raw_in=connectome.raw_in_synapses,
            raw_out=connectome.raw_out_synapses,
        )
        connectome.annotations.reset_index().to_feather(annotation_path)
        matrix_path.with_suffix(".json").write_text(json.dumps(connectome.meta, indent=2))
    return connectome


def _main() -> None:
    import argparse

    from .cell_types import connection_between, escape_report

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="malecns-1.0")
    parser.add_argument("--min-synapses", type=int, default=DEFAULT_MIN_SYNAPSES)
    parser.add_argument("--report-escape-types", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    connectome = load_connectome(
        args.dataset, min_synapses=args.min_synapses, use_cache=not args.no_cache
    )

    print("=" * 78)
    print(f"{connectome.meta['dataset']}   {connectome.meta['citation']}")
    print("=" * 78)
    for key in (
        "orientation",
        "min_synapses",
        "n_bodies_in_release",
        "n_neurons",
        "n_edges_raw",
        "n_edges_after_threshold",
        "n_edges_on_graph",
        "n_edges_dropped_modulatory",
        "n_edges_final",
        "n_synapses_final",
    ):
        print(f"  {key:<28} {connectome.meta[key]:,}" if isinstance(
            connectome.meta[key], (int, float)
        ) else f"  {key:<28} {connectome.meta[key]}")
    print(f"  nt_sign                      {connectome.meta['nt_sign']}")

    if not args.report_escape_types:
        return

    report = escape_report(connectome)
    print("\n" + "=" * 78)
    print("ESCAPE PATHWAY - Phase 0 exit criterion")
    print("=" * 78)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(report.to_string(index=False))

    missing = report.attrs.get("missing_patterns", [])
    print(f"\nunmatched patterns from the spec: {missing or 'none'}")

    print("\n--- the connection the dataset switch was made for ---")
    for pre, post in (
        ("DNp01", "TTMn"),
        ("DNp01", "DLMn*"),
        ("LPLC2", "DNp01"),
        ("LC4", "DNp01"),
        ("LC6", "DNp01"),
        ("LC22", "DNp01"),
    ):
        total, count = connection_between(connectome, pre, post)
        print(f"  {pre:>7} -> {post:<7} signed weight {total:>10,.0f}  over {count:>3} edges")


if __name__ == "__main__":
    _main()

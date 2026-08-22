"""Dataset adapters.

Every dataset-specific fact - file layout, column spelling, transmitter vocabulary,
what counts as a neuron - lives behind ``ConnectomeSource``. Nothing outside this module
may branch on which dataset is loaded. Adding a third connectome should mean adding one
class here and touching nothing else.

Two adapters exist:

``MaleCNSSource``
    The primary substrate. One male fly's entire CNS: central brain, both optic lobes and
    the full ventral nerve cord, with an intact neck connective. Transmitter predictions are
    published *per neuron*, not per synapse, so the sign of an edge is determined by its
    presynaptic cell.

``FlyWire783Source``
    Used only to reproduce Shiu et al. in Phase 1, on the data they ran. Transmitter
    predictions are published *per edge*.

Edges are yielded in chunks rather than returned whole. The MaleCNS edge list is
151,856,684 rows; materialising it as int64 columns costs 3.6 GB before a single temporary,
and the masking and searchsorted intermediates that follow push a naive implementation past
available memory. Streaming keeps peak usage flat and independent of release size.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pyarrow.ipc as ipc

from .neurotransmitters import normalise

RAW = Path(__file__).resolve().parent / "raw"

# Canonical annotation columns every adapter must produce, indexed by body/root id.
ANNOTATION_COLUMNS = (
    "cell_type",     # primary cell type label, e.g. "LPLC2", "DNp01", "TTMn"
    "side",          # "L" | "R" | "M" | ""
    "superclass",    # coarse role, e.g. "descending_neuron", "visual_projection"
    "klass",         # finer class where the dataset provides one
    "status",        # proofreading status, dataset-specific vocabulary
    "flywire_type",  # FlyWire cell type correspondence where known ("" otherwise)
)


@dataclass(frozen=True)
class EdgeChunk:
    """A slice of the edge list.

    ``nt_code`` indexes into ``nt_vocab``, which holds names already normalised to the
    vocabulary of :mod:`data.neurotransmitters`. Carrying codes rather than strings keeps
    the per-chunk work to integer indexing; the loader turns the small vocabulary into
    signs once and never touches a string in the hot path.
    """

    pre: np.ndarray
    post: np.ndarray
    syn_count: np.ndarray
    nt_code: np.ndarray
    nt_vocab: np.ndarray

    def __len__(self) -> int:
        return int(self.pre.size)


class ConnectomeSource(ABC):
    """One connectome release, in whatever shape its publishers chose."""

    key: str
    citation: str
    root: Path

    @abstractmethod
    def load_annotations(self) -> pd.DataFrame:
        """Per-neuron annotations indexed by body/root id, with ANNOTATION_COLUMNS."""

    @abstractmethod
    def iter_edges(self) -> Iterator[EdgeChunk]:
        """Yield the edge list in chunks, transmitter already normalised."""

    @abstractmethod
    def is_neuron(self, annotations: pd.DataFrame) -> pd.Series:
        """Boolean mask selecting rows that are actual proofread neurons.

        Glia, orphan fragments and out-of-scope bodies are excluded here rather than by
        a magic filter buried in the loader.
        """


class MaleCNSSource(ConnectomeSource):
    key = "malecns-1.0"
    citation = (
        "MaleCNS v1.0, Janelia FlyEM, CC-BY. https://male-cns.janelia.org/download/"
    )

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or (RAW / "malecns")

    def load_annotations(self) -> pd.DataFrame:
        frame = feather.read_table(self.root / "body-annotations.feather").to_pandas()
        out = pd.DataFrame(index=pd.Index(frame["bodyId"].to_numpy(), name="id"))
        out["cell_type"] = frame["type"].fillna("").astype(str).to_numpy()
        out["side"] = frame["somaSide"].fillna("").astype(str).to_numpy()
        out["superclass"] = frame["superclass"].fillna("").astype(str).to_numpy()
        out["klass"] = frame["class"].fillna("").astype(str).to_numpy()
        out["status"] = frame["status"].fillna("").astype(str).to_numpy()
        out["flywire_type"] = frame["flywireType"].fillna("").astype(str).to_numpy()
        out["instance"] = frame["instance"].fillna("").astype(str).to_numpy()
        # somaLocation is a list<int64> [x, y, z] in 8 nm voxels, kept for the Phase 4 point
        # cloud. Absent for a substantial minority of bodies, hence nullable float.
        coords = np.full((len(frame), 3), np.nan)
        for i, value in enumerate(frame["somaLocation"].to_numpy()):
            if value is not None and len(value) == 3:
                coords[i] = value
        out[["soma_x", "soma_y", "soma_z"]] = coords
        return out

    def _transmitter_by_body(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sorted body ids, their transmitter codes, and the shared vocabulary.

        MaleCNS publishes several columns of increasing aggregation. ``consensus_nt`` is the
        reconciled call and is preferred; where absent we fall back to the cell-type level
        prediction, then the per-body prediction, then explicit "unknown". Falling back
        rather than dropping keeps counts honest: a neuron with no transmitter call still
        exists, it simply contributes sign 0.
        """
        table = feather.read_table(
            self.root / "body-neurotransmitters.feather",
            columns=["body", "consensus_nt", "celltype_predicted_nt", "predicted_nt"],
        ).to_pandas()
        call = (
            table["consensus_nt"]
            .fillna(table["celltype_predicted_nt"])
            .fillna(table["predicted_nt"])
            .fillna("unknown")
            .astype(str)
            .map(normalise)
            .to_numpy()
        )
        bodies = table["body"].to_numpy()
        vocab, codes = np.unique(call, return_inverse=True)
        if "unknown" not in vocab:
            vocab = np.append(vocab, "unknown")
        order = np.argsort(bodies)
        return bodies[order], codes.astype(np.int8)[order], vocab

    def iter_edges(self) -> Iterator[EdgeChunk]:
        bodies, body_code, vocab = self._transmitter_by_body()
        unknown = int(np.flatnonzero(vocab == "unknown")[0])

        with (self.root / "connectome-weights.feather").open("rb") as fh:
            reader = ipc.open_file(fh)
            for i in range(reader.num_record_batches):
                batch = reader.get_batch(i)
                pre = batch.column("body_pre").to_numpy(zero_copy_only=False)
                post = batch.column("body_post").to_numpy(zero_copy_only=False)
                weight = batch.column("weight").to_numpy(zero_copy_only=False)

                # An edge inherits the transmitter of its presynaptic cell.
                slot = np.searchsorted(bodies, pre)
                np.clip(slot, 0, bodies.size - 1, out=slot)
                found = bodies[slot] == pre
                code = np.where(found, body_code[slot], unknown).astype(np.int8)

                yield EdgeChunk(
                    pre=pre,
                    post=post,
                    syn_count=weight.astype(np.int32),
                    nt_code=code,
                    nt_vocab=vocab,
                )

    def is_neuron(self, annotations: pd.DataFrame) -> pd.Series:
        # 'Traced' excludes Glia, Orphan, Unimportant, Assign and Anchor, which are the
        # other values MaleCNS uses in this field.
        return annotations["status"] == "Traced"


class FlyWire783Source(ConnectomeSource):
    key = "flywire-783"
    citation = (
        "FlyWire FAFB release 783 (Codex flat files); Dorkenwald et al. 2024, "
        "Schlegel et al. 2024. See https://flywire.ai/guidelines"
    )

    def __init__(self, root: Path | None = None) -> None:
        self.root = root or (RAW / "flywire783")

    def load_annotations(self) -> pd.DataFrame:
        classification = pd.read_csv(
            self.root / "classification.csv.gz", dtype={"root_id": np.uint64}
        )
        types = pd.read_csv(
            self.root / "consolidated_cell_types.csv.gz", dtype={"root_id": np.uint64}
        )
        merged = classification.merge(types, on="root_id", how="left")

        out = pd.DataFrame(index=pd.Index(merged["root_id"].to_numpy(), name="id"))
        out["cell_type"] = merged["primary_type"].fillna("").astype(str).to_numpy()
        side = merged["side"].fillna("").astype(str).str.upper().str[:1]
        out["side"] = side.replace({"C": "M"}).to_numpy()  # 'center' -> midline
        out["superclass"] = merged["super_class"].fillna("").astype(str).to_numpy()
        out["klass"] = merged["class"].fillna("").astype(str).to_numpy()
        out["status"] = "proofread"  # every row in the Codex release is proofread
        out["flywire_type"] = out["cell_type"]
        out["instance"] = ""
        out[["soma_x", "soma_y", "soma_z"]] = np.nan
        return out

    def iter_edges(self) -> Iterator[EdgeChunk]:
        # ~3M rows: small enough to read whole, but chunked anyway so both adapters take
        # the same path through the loader.
        for frame in pd.read_csv(
            self.root / "connections.csv.gz",
            usecols=["pre_root_id", "post_root_id", "syn_count", "nt_type"],
            dtype={
                "pre_root_id": np.uint64,
                "post_root_id": np.uint64,
                "syn_count": np.int32,
                "nt_type": str,
            },
            chunksize=2_000_000,
        ):
            nt = frame["nt_type"].fillna("unknown").astype(str).map(normalise).to_numpy()
            vocab, codes = np.unique(nt, return_inverse=True)
            yield EdgeChunk(
                pre=frame["pre_root_id"].to_numpy(),
                post=frame["post_root_id"].to_numpy(),
                syn_count=frame["syn_count"].to_numpy(),
                nt_code=codes.astype(np.int8),
                nt_vocab=vocab,
            )

    def is_neuron(self, annotations: pd.DataFrame) -> pd.Series:
        return pd.Series(True, index=annotations.index)


SOURCES: dict[str, type[ConnectomeSource]] = {
    MaleCNSSource.key: MaleCNSSource,
    FlyWire783Source.key: FlyWire783Source,
}


def get_source(key: str) -> ConnectomeSource:
    if key not in SOURCES:
        raise KeyError(f"unknown dataset {key!r}; available: {sorted(SOURCES)}")
    return SOURCES[key]()

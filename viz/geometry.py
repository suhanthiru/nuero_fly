"""Fetch and parse neuroglancer precomputed geometry from the MaleCNS public bucket.

MaleCNS publishes everything we need for the 3D view - compartment shells, neuron meshes,
skeletons - as neuroglancer precomputed data on an anonymous bucket. The shells and the
compartment meshes use ``neuroglancer_legacy_mesh``, which is a pleasantly simple format:

    uint32                       vertex count
    float32[3 * vertex_count]    xyz, in nanometres
    uint32[3 * triangle_count]   triangle indices

Coordinates are nanometres, the same frame as ``somaLocation`` in the annotations once its
8 nm voxel scale is applied. :func:`soma_positions_nm` does that conversion so there is one
place where the two are reconciled.

Neuron meshes live under ``segmentation/multi-res-meshes`` in sharded Draco format, which is
considerably harder to read. ``segmentation/single-res-meshes`` is legacy format and is the
route to prefer for the handful of neurons that get real geometry.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

BUCKET = "https://storage.googleapis.com/flyem-male-cns"

# Soma coordinates in the annotations are in 8 nm voxels.
SOMA_VOXEL_NM = 8.0

COMPARTMENTS = "rois/malecns-major-compartments-v2"
BRAIN_SHELL = "rois/brain-shell-v2.2"
VNC_SHELL = "rois/vnc-shell-v2"


@dataclass(frozen=True)
class Mesh:
    """A triangle mesh in nanometres."""

    name: str
    positions: np.ndarray   # (V, 3) float32
    indices: np.ndarray     # (T, 3) uint32

    @property
    def n_vertices(self) -> int:
        return int(self.positions.shape[0])

    @property
    def n_triangles(self) -> int:
        return int(self.indices.shape[0])

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.positions.min(axis=0), self.positions.max(axis=0)


def parse_ngmesh(blob: bytes, name: str = "") -> Mesh:
    """Decode one ``neuroglancer_legacy_mesh`` fragment."""
    (n_vertices,) = struct.unpack("<I", blob[:4])
    vertex_bytes = 12 * n_vertices
    positions = np.frombuffer(
        blob, dtype="<f4", count=3 * n_vertices, offset=4
    ).reshape(n_vertices, 3)
    indices = np.frombuffer(blob, dtype="<u4", offset=4 + vertex_bytes)
    if indices.size % 3:
        raise ValueError(f"{name}: index buffer is not a multiple of 3")
    return Mesh(name=name, positions=positions.copy(), indices=indices.reshape(-1, 3).copy())


def segment_labels(source: str, *, client: httpx.Client | None = None) -> dict[str, str]:
    """Map segment id -> human label, from a precomputed segment_properties block."""
    owns = client is None
    client = client or httpx.Client(timeout=60.0)
    try:
        info = client.get(f"{BUCKET}/{source}/segment_properties/info").json()
        inline = info["inline"]
        labels = next(
            p["values"] for p in inline["properties"] if p["type"] == "label"
        )
        return dict(zip(inline["ids"], labels))
    finally:
        if owns:
            client.close()


def fetch_mesh(
    source: str, segment: str, *, client: httpx.Client | None = None, name: str = ""
) -> Mesh:
    """Fetch one segment's mesh: read its manifest, then concatenate its fragments."""
    owns = client is None
    client = client or httpx.Client(timeout=300.0)
    try:
        manifest = client.get(f"{BUCKET}/{source}/mesh/{segment}:0").json()
        meshes = []
        for fragment in manifest["fragments"]:
            blob = client.get(f"{BUCKET}/{source}/mesh/{fragment}").content
            meshes.append(parse_ngmesh(blob, name=fragment))
        return _concatenate(meshes, name=name or segment)
    finally:
        if owns:
            client.close()


def fetch_compartments(source: str = COMPARTMENTS) -> list[Mesh]:
    """All labelled compartment meshes: CentralBrain, Optic(L/R), CV, VNC."""
    with httpx.Client(timeout=300.0) as client:
        labels = segment_labels(source, client=client)
        return [
            fetch_mesh(source, segment, client=client, name=label)
            for segment, label in labels.items()
        ]


def _concatenate(meshes: list[Mesh], name: str) -> Mesh:
    if len(meshes) == 1:
        return Mesh(name=name, positions=meshes[0].positions, indices=meshes[0].indices)
    positions, indices, offset = [], [], 0
    for mesh in meshes:
        positions.append(mesh.positions)
        indices.append(mesh.indices + offset)
        offset += mesh.n_vertices
    return Mesh(
        name=name,
        positions=np.concatenate(positions),
        indices=np.concatenate(indices),
    )


def soma_positions_nm(annotations) -> tuple[np.ndarray, np.ndarray]:
    """Soma coordinates in nanometres, and a mask of which neurons actually have one.

    Roughly a third of bodies carry no soma location. They are dropped from the point cloud
    rather than drawn at the origin, which would put a spurious blob in the middle of the
    brain.
    """
    xyz = annotations[["soma_x", "soma_y", "soma_z"]].to_numpy(dtype=np.float64)
    have = np.isfinite(xyz).all(axis=1)
    return (xyz * SOMA_VOXEL_NM).astype(np.float32), have


def write_binary(path: Path, arrays: dict[str, np.ndarray]) -> dict[str, dict]:
    """Write several arrays into one flat binary blob, returning a layout manifest.

    Deliberately not glTF. The frontend wants typed arrays it can hand straight to the GPU,
    and a hand-rolled layout is far less code than a glTF writer plus a glTF parser.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    layout: dict[str, dict] = {}
    offset = 0
    with path.open("wb") as fh:
        for key, array in arrays.items():
            data = np.ascontiguousarray(array)
            blob = data.tobytes()
            layout[key] = {
                "offset": offset,
                "length": len(blob),
                "dtype": data.dtype.name,
                "shape": list(data.shape),
            }
            fh.write(blob)
            offset += len(blob)
    return layout


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))

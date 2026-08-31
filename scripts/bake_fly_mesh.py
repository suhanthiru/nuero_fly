"""Bake the flybody model into one posed mesh for the viewer.

flybody is a kinematic tree of 67 parts. The viewer needs a single mesh in a resting pose,
so this loads the model in MuJoCo, runs forward kinematics once at the default joint
configuration, reads each visual geom's world transform, and bakes the vertices into that
frame. Using MuJoCo to do it rather than walking the tree by hand means the pose is exactly
the model's own, with no chance of a transform composed in the wrong order.

**This is geometry, not biomechanics.** The escape in this project is a rigid-body impulse;
the baked mesh is a body to look at while that impulse plays out. Nothing here actuates a
leg, and the viewer says so. Rendering an articulated fly whose joints never move would
imply leg mechanics we do not simulate, which is worse than the ellipsoid it replaces.

flybody: Vaxenburg et al., Nature 2025. Apache-2.0, via google-deepmind/mujoco_menagerie.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

from viz import geometry

RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "flybody"
OUT = Path(__file__).resolve().parent.parent / "viz" / "frontend" / "public" / "scene"

#: Target body length in millimetres, to match the arena's rigid fly.
TARGET_LENGTH_MM = 2.5

#: Triangle budget. The raw model is far heavier than a 2.5 mm object on screen needs.
MAX_TRIANGLES = 60_000


def bake(model_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return world-space vertices and faces for every visual mesh geom, at qpos0."""
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    groups = {}
    for geom in range(model.ngeom):
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
            groups.setdefault(int(model.geom_group[geom]), 0)
            groups[int(model.geom_group[geom])] += 1
    print(f"  mesh geoms by group: {groups}")

    # flybody puts its visual shell and its collision primitives in different groups; the
    # visual one is whichever group holds the bulk of the mesh geoms.
    visual_group = max(groups, key=lambda key: groups[key])
    print(f"  taking group {visual_group}")

    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0
    for geom in range(model.ngeom):
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if int(model.geom_group[geom]) != visual_group:
            continue
        mesh = int(model.geom_dataid[geom])
        vadr, vnum = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
        fadr, fnum = int(model.mesh_faceadr[mesh]), int(model.mesh_facenum[mesh])

        local = model.mesh_vert[vadr : vadr + vnum].reshape(-1, 3)
        face = model.mesh_face[fadr : fadr + fnum].reshape(-1, 3)

        rotation = data.geom_xmat[geom].reshape(3, 3)
        translation = data.geom_xpos[geom]
        vertices.append(local @ rotation.T + translation)
        faces.append(face + offset)
        offset += vnum

    return np.concatenate(vertices), np.concatenate(faces)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=RAW / "fruitfly.xml")
    args = parser.parse_args()

    print(f"loading {args.model.name} ...")
    vertices, faces = bake(args.model)
    print(f"  baked {len(vertices):,} vertices, {len(faces):,} triangles")

    # Centre and scale to the arena's fly, which is 2.5 mm long along +X.
    #
    # Scale by the X extent specifically, not by the largest axis. The model's widest
    # dimension is Y - that is the leg span, not the body - so scaling by the maximum would
    # squeeze the fly to half size. The head sits at +X and the abdomen at -X, so the
    # anatomical forward axis already matches the arena and no rotation is needed.
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    extent = high - low
    print(f"  native extent (body X, legs Y, height Z): {np.round(extent, 2)}")
    vertices = vertices - (low + high) / 2.0
    vertices = vertices * (TARGET_LENGTH_MM / extent[0])

    mesh = geometry.Mesh(
        name="fly",
        positions=vertices.astype(np.float32),
        indices=faces.astype(np.uint32),
    )
    mesh = geometry.decimate(mesh, MAX_TRIANGLES)
    low, high = mesh.bounds()
    span = high - low
    print(f"  after scaling and decimation: {mesh.n_triangles:,} triangles, "
          f"body {span[0]:.2f} mm long, {span[1]:.2f} mm leg span, {span[2]:.2f} mm tall")

    layout = geometry.write_binary(
        OUT / "fly.bin", {"position": mesh.positions, "index": mesh.indices}
    )
    geometry.write_manifest(
        OUT / "fly.json",
        {
            "file": "fly.bin",
            "layout": layout,
            "n_triangles": mesh.n_triangles,
            "length_mm": float((high - low)[0]),
            "leg_span_mm": float((high - low)[1]),
            "source": "flybody (Vaxenburg et al., Nature 2025), Apache-2.0, "
                      "via google-deepmind/mujoco_menagerie",
            "note": "geometry only - the escape is a rigid-body impulse, no leg mechanics",
        },
    )
    print(f"\nwrote {OUT / 'fly.bin'} ({(OUT / 'fly.bin').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()

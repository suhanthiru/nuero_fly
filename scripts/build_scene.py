"""Export the 3D scene the frontend loads: soma point cloud plus compartment shells.

Runs offline and writes into ``viz/frontend/public/scene``. Nothing here touches the
simulation - geometry and activity are deliberately separate, so the viewer can be built
and iterated on before there are any spikes to show.

Coordinates: everything is converted to micrometres and recentred on the midpoint of the
whole CNS, so the model orbits about itself rather than about the corner of the EM volume.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx
import numpy as np

from data.cell_types import ids_for
from data.loader import load_connectome
from viz import geometry
from viz.palette import (
    BACKGROUND,
    CELL_TYPE_COLOR,
    COMPARTMENT_COLOR,
    COMPARTMENT_OPACITY,
    STAGE_COLOR,
    colors_for,
    hex_to_rgb,
    stage_of,
)

OUT = Path(__file__).resolve().parent.parent / "viz" / "frontend" / "public" / "scene"
NM_PER_UM = 1000.0

# Which cells get real morphology, and in what form.
#
# Skeletons for the populations: LC4 and LPLC2 are the giant fiber's two real inputs, and
# what matters visually is the bundle, not any individual axon. Meshes for the cells whose
# individual spiking is read out - the giant fiber itself and the motor neuron it drives -
# because there are four of them in total and each one is an object worth seeing.
#
# LC6 and LC22 are deliberately absent. Phase 0 found they make zero synapses onto DNp01,
# even unthresholded, so drawing them converging on it would be a lie.
MORPHOLOGY: dict[str, str] = {
    "LC4": "skeleton",
    "LPLC2": "skeleton",
    "DNp01": "mesh",
    "TTMn": "mesh",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="malecns-1.0")
    args = parser.parse_args()

    print("loading connectome ...")
    connectome = load_connectome(args.dataset)
    annotations = connectome.annotations

    print("fetching compartment meshes ...")
    meshes = geometry.fetch_compartments()
    for mesh in meshes:
        low, high = mesh.bounds()
        print(
            f"  {mesh.name:<14} {mesh.n_vertices:>7,} verts  {mesh.n_triangles:>7,} tris  "
            f"bounds {np.round(low / NM_PER_UM).astype(int)} .. "
            f"{np.round(high / NM_PER_UM).astype(int)} um"
        )

    # One shared transform for every piece of geometry, derived from the compartments
    # because they define the true extent of the volume.
    all_points = np.concatenate([m.positions for m in meshes])
    low = all_points.min(axis=0)
    high = all_points.max(axis=0)
    centre = (low + high) / 2.0
    extent_um = (high - low) / NM_PER_UM
    print(f"  volume extent: {np.round(extent_um).astype(int)} um")

    # The volume's native frame has Z increasing brain -> cervical connective -> VNC, so the
    # animal would hang brain-down in a Z-up viewer. Correct it with a 180 degree rotation
    # about X: (x, y, z) -> (x, -y, -z). A rotation, deliberately, and not a Z flip - a flip
    # mirrors the volume and silently swaps left and right, which would quietly invalidate
    # every directional-tuning result the project is aiming at.
    def to_um(points: np.ndarray) -> np.ndarray:
        centred = (points - centre) / NM_PER_UM
        centred[:, 1] *= -1.0
        centred[:, 2] *= -1.0
        return centred.astype(np.float32)

    # ---- somata -------------------------------------------------------------------
    positions_nm, have_soma = geometry.soma_positions_nm(annotations)
    print(
        f"somata: {have_soma.sum():,} of {len(annotations):,} neurons have a soma location"
    )
    kept = annotations[have_soma]
    positions = to_um(positions_nm[have_soma])
    rgb, is_pathway = colors_for(kept)

    cell_types = kept["cell_type"].to_numpy()
    vocabulary, type_index = np.unique(cell_types, return_inverse=True)
    stages = np.array([stage_of(t) for t in vocabulary])

    somata_layout = geometry.write_binary(
        OUT / "somata.bin",
        {
            "position": positions,
            "color": rgb,
            "pathway": is_pathway.astype(np.uint8),
            "type_index": type_index.astype(np.uint16),
            "body_id": np.asarray(kept.index, dtype=np.uint64),
        },
    )

    # ---- compartment shells -------------------------------------------------------
    shell_arrays: dict[str, np.ndarray] = {}
    shells = []
    for mesh in meshes:
        shell_arrays[f"{mesh.name}/position"] = to_um(mesh.positions)
        shell_arrays[f"{mesh.name}/index"] = mesh.indices.astype(np.uint32)
        shells.append(
            {
                "name": mesh.name,
                "color": hex_to_rgb(COMPARTMENT_COLOR.get(mesh.name, "#7a8496")),
                "opacity": COMPARTMENT_OPACITY.get(mesh.name, 0.045),
                "n_vertices": mesh.n_vertices,
                "n_triangles": mesh.n_triangles,
            }
        )
    shell_layout = geometry.write_binary(OUT / "compartments.bin", shell_arrays)

    # ---- morphology: skeletons for populations, meshes for identified cells ---------
    print("\nfetching morphology ...")
    morphology_arrays: dict[str, np.ndarray] = {}
    groups = []
    for cell_type, kind in MORPHOLOGY.items():
        body_ids = ids_for(connectome, cell_type)
        color = hex_to_rgb(CELL_TYPE_COLOR.get(cell_type, "#cccccc"))

        if kind == "skeleton":
            skeletons = geometry.fetch_skeletons(body_ids)
            source, target, owner = geometry.skeletons_to_segments(skeletons)
            morphology_arrays[f"{cell_type}/source"] = to_um(source)
            morphology_arrays[f"{cell_type}/target"] = to_um(target)
            morphology_arrays[f"{cell_type}/owner"] = owner
            groups.append(
                {
                    "name": cell_type,
                    "kind": "skeleton",
                    "color": color,
                    "n_neurons": len(skeletons),
                    "n_requested": len(body_ids),
                    "n_segments": int(owner.size),
                }
            )
            print(
                f"  {cell_type:<7} {len(skeletons):>3}/{len(body_ids):<3} skeletons, "
                f"{owner.size:>7,} segments"
            )
        else:
            bodies = []
            with httpx.Client(timeout=300.0) as client:
                for body_id in body_ids:
                    mesh = geometry.fetch_neuron_mesh(int(body_id), client=client)
                    if mesh is None:
                        continue
                    morphology_arrays[f"{cell_type}/{body_id}/position"] = to_um(
                        mesh.positions
                    )
                    morphology_arrays[f"{cell_type}/{body_id}/index"] = mesh.indices.astype(
                        np.uint32
                    )
                    bodies.append(
                        {
                            "body_id": int(body_id),
                            "n_vertices": mesh.n_vertices,
                            "n_triangles": mesh.n_triangles,
                        }
                    )
            groups.append(
                {"name": cell_type, "kind": "mesh", "color": color, "bodies": bodies}
            )
            total_tris = sum(b["n_triangles"] for b in bodies)
            print(f"  {cell_type:<7} {len(bodies)} meshes, {total_tris:>7,} triangles")

    morphology_layout = geometry.write_binary(OUT / "morphology.bin", morphology_arrays)

    geometry.write_manifest(
        OUT / "manifest.json",
        {
            "dataset": connectome.meta["dataset"],
            "citation": connectome.meta["citation"],
            "units": "micrometres, recentred on the CNS midpoint",
            "orientation": (
                "rotated 180 degrees about X from the native volume frame, so +Z is "
                "brain-ward and the animal stands upright. Handedness is preserved: "
                "+X is the Optic(L) side."
            ),
            "extent_um": [round(float(v), 1) for v in extent_um],
            "background": BACKGROUND,
            "somata": {
                "count": int(have_soma.sum()),
                "file": "somata.bin",
                "layout": somata_layout,
                "cell_types": vocabulary.tolist(),
                "stages": stages.tolist(),
                "pathway_count": int(is_pathway.sum()),
            },
            "compartments": {
                "file": "compartments.bin",
                "layout": shell_layout,
                "shells": shells,
            },
            "morphology": {
                "file": "morphology.bin",
                "layout": morphology_layout,
                "groups": groups,
            },
            "palette": {
                "stage": STAGE_COLOR,
                "cell_type": CELL_TYPE_COLOR,
            },
        },
    )

    total = sum(p.stat().st_size for p in OUT.glob("*"))
    print(f"\nwrote {OUT} ({total / 1e6:.1f} MB)")
    print(f"  escape-pathway somata: {is_pathway.sum():,}")
    for stage in ("visual_projection", "descending", "motor"):
        n = int(sum(1 for t in cell_types if stage_of(t) == stage))
        print(f"    {stage:<20} {n:,}")


if __name__ == "__main__":
    main()

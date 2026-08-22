"""Colour assignment.

Two rules, from the design decisions, and they are load-bearing:

* **Colour encodes cell type. Brightness encodes activity.** Activity never touches hue -
  map it to hue and the anatomy disappears.
* **Each stage of the circuit owns a hue family.** Blue-cyan for visual input, amber-orange
  for descending, red-magenta for motor. The escape then reads as a colour wave travelling
  down the animal, which is the story the whole thing is telling. Within a family, types
  are separated by lightness and small hue shifts.

Everything that is not escape pathway is context: desaturated slate, never bright, tinted
only enough by region that the brain does not read as a flat grey cloud.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BACKGROUND = "#08090b"

# Ordered by position in the circuit. The order is the palette's meaning.
STAGE_ORDER = ("visual_projection", "descending", "motor", "context")

STAGE_COLOR: dict[str, str] = {
    "visual_projection": "#4a9eff",
    "descending": "#ff8a3d",
    "motor": "#f43f5e",
    "context": "#64748b",
}

# Per-type colours within each family.
CELL_TYPE_COLOR: dict[str, str] = {
    # visual input - blue / cyan
    "LC4": "#4a9eff",
    "LC6": "#38bdf8",
    "LC22": "#22d3ee",
    "LPLC2": "#67e8f9",
    # descending - amber / orange
    "DNp01": "#ff8a3d",
    "DNp02": "#fbbf24",
    "DNp04": "#f59e0b",
    "DNp11": "#fdba74",
    # motor - red / magenta
    "TTMn": "#f43f5e",
    "DLMn a, b": "#fb7185",
    "DLMn c-f": "#fda4af",
    # the giant-fiber-coupled VNC interneurons. Not in the original spec list, but the
    # Phase 0 probe found GFC2 drives TTMn harder than the giant fiber does directly,
    # so they are part of the story and get to be visible.
    "GFC2": "#ffb27a",
    "GFC4": "#ffc9a3",
    "PSI": "#ff9f6e",
}

# Context tints by superclass. Near-neutral slate, varied just enough to separate the optic
# lobes from the central brain from the nerve cord. These must never compete with the
# escape palette for attention.
SUPERCLASS_COLOR: dict[str, str] = {
    "ol_intrinsic": "#5b6a80",
    "ol_sensory": "#556377",
    # Visual projection neurons that are *not* LC4/LPLC2 stay context-coloured: being in
    # the superclass is not the same as being in the escape pathway.
    "visual_projection": "#5f6d84",
    "visual_centrifugal": "#5c6b81",
    "cb_intrinsic": "#6b7280",
    "cb_sensory": "#6a7078",
    "cb_motor": "#7a6f74",
    "vnc_intrinsic": "#6d7a8a",
    "vnc_sensory": "#66727f",
    "vnc_motor": "#7d7078",
    "descending_neuron": "#77736f",
    "ascending_neuron": "#6f7580",
}

DEFAULT_CONTEXT = "#5a6473"


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def stage_of(cell_type: str) -> str:
    """Which circuit stage a cell type belongs to, or 'context'."""
    from data.cell_types import ESCAPE_PATHWAY

    import fnmatch

    for stage, patterns in ESCAPE_PATHWAY.items():
        for pattern in patterns:
            if cell_type == pattern or fnmatch.fnmatch(cell_type, pattern):
                return stage
    if cell_type in ("GFC2", "GFC4", "PSI"):
        return "descending"
    return "context"


def colors_for(annotations: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """RGB (uint8, N x 3) and a boolean mask marking escape-pathway neurons.

    The mask is what the renderer uses to decide who is allowed to be bright: context
    neurons are drawn dim and small, pathway neurons drawn larger and lit.
    """
    cell_type = annotations["cell_type"].to_numpy()
    superclass = annotations["superclass"].to_numpy()

    rgb = np.empty((len(annotations), 3), dtype=np.uint8)
    is_pathway = np.zeros(len(annotations), dtype=bool)

    # Resolve per distinct label rather than per neuron: 165k rows, a few hundred labels.
    context_cache = {
        key: hex_to_rgb(SUPERCLASS_COLOR.get(key, DEFAULT_CONTEXT))
        for key in set(superclass)
    }
    for i, (ctype, sclass) in enumerate(zip(cell_type, superclass)):
        if ctype in CELL_TYPE_COLOR:
            rgb[i] = hex_to_rgb(CELL_TYPE_COLOR[ctype])
            is_pathway[i] = True
        else:
            rgb[i] = context_cache[sclass]
    return rgb, is_pathway


# Compartment shells. Desaturated and near-neutral by explicit decision: the tints exist to
# orient the viewer, not to carry information, and hue belongs to cell type.
COMPARTMENT_COLOR: dict[str, str] = {
    "CentralBrain": "#2e3d52",
    "Optic(L)": "#27374b",
    "Optic(R)": "#27374b",
    "CV": "#54432f",   # cervical connective - the neck. Warmed, because the giant fiber
                       # descending through it is the most legible single event here.
    "VNC": "#293850",
}

# Only back faces are drawn, so these are read once per pixel rather than accumulating
# through both walls of the hull. That keeps the silhouette readable without the shell
# brightening past the neurons it is supposed to sit behind.
COMPARTMENT_OPACITY: dict[str, float] = {
    "CentralBrain": 0.30,
    "Optic(L)": 0.26,
    "Optic(R)": 0.26,
    "CV": 0.16,
    "VNC": 0.28,
}

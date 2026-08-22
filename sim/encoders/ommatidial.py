"""Ommatidial encoder - not implemented.

This is the documented hook that brings the optic lobe to life. Version 1 uses
:class:`sim.encoders.analytic.AnalyticLoomingEncoder`, which computes angular size and
expansion rate from scene geometry and drives LC4 and LPLC2 directly. That bypasses the
optic lobe, so its ~50,000 neurons carry no activity and appear only as dim anatomical
context in the viewer. This is deliberate and documented, not an oversight.

Implementing this class means: rendering the scene into the fly's ommatidial array,
driving photoreceptors R1-R8, and letting the lamina, medulla, lobula and lobula plate
compute the looming response themselves - at which point LC4 and LPLC2 activity becomes a
prediction of the model rather than an input to it, and the optic lobe lights up.

Because everything downstream depends only on :class:`SensoryEncoder`, that is a module
swap. Nothing in ``sim/`` outside ``analytic.py`` knows that LC neurons are where drive
currently enters.
"""

from __future__ import annotations

from .base import SceneState, SensoryEncoder

_MESSAGE = """\
OmmatidialEncoder is not implemented.

Version 1 drives LC4 and LPLC2 directly from analytic scene geometry, which is why the
optic lobe is silent. Implementing this encoder requires:

  1. a projection from scene geometry onto the ommatidial lattice (~800 columns per eye;
     MaleCNS publishes column assignments per neuron in the annotations as assignedOlHex1
     and assignedOlHex2, and column ROI meshes under rois/ME(R)-columns-v8 and friends);
  2. a photoreceptor model driving R1-R8 - note that histamine is inhibitory, so the sign
     convention in data/neurotransmitters.py already handles the sign flip at the first
     synapse;
  3. removing LC4/LPLC2 from the drive path entirely, so their activity becomes a
     prediction rather than an input.

Use AnalyticLoomingEncoder for now.
"""


class OmmatidialEncoder(SensoryEncoder):
    """Phase 2 upgrade path. Raises on construction."""

    name = "ommatidial"

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError(_MESSAGE)

    def encode(self, scene: SceneState) -> dict[int, float]:
        raise NotImplementedError(_MESSAGE)

    def target_ids(self) -> list[int]:
        raise NotImplementedError(_MESSAGE)

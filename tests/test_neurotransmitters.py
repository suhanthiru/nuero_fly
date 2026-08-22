"""The sign mapping is the most dangerous code in the project. Pin it hard."""

from __future__ import annotations

import pytest

from data.neurotransmitters import (
    NT_SIGN,
    UnknownNeurotransmitter,
    normalise,
    sign_for,
)


def test_excitatory_and_inhibitory_signs():
    assert sign_for("acetylcholine") == +1
    assert sign_for("gaba") == -1
    assert sign_for("glutamate") == -1
    assert sign_for("histamine") == -1


def test_glutamate_is_inhibitory():
    """Guards the one call the build spec left ambiguous.

    The spec listed glutamate on both the excitatory and inhibitory side. It is inhibitory
    because GluCl-alpha dominates at fly central synapses, and because Shiu et al. make the
    same choice - flipping it would break the Phase 1 reproduction. If this test is ever
    changed, the Phase 1 gate must be re-run.
    """
    assert NT_SIGN["glutamate"] == -1


def test_modulators_carry_no_current():
    for modulator in ("dopamine", "octopamine", "serotonin"):
        assert sign_for(modulator) == 0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ACH", "acetylcholine"),      # FlyWire Codex spelling
        ("GABA", "gaba"),
        ("GLUT", "glutamate"),
        ("DA", "dopamine"),
        ("SER", "serotonin"),
        ("acetylcholine", "acetylcholine"),  # MaleCNS spelling
        ("  GABA  ", "gaba"),
        ("", "unknown"),
        (None, "unknown"),
    ],
)
def test_dataset_spellings_normalise(raw, expected):
    assert normalise(raw) == expected


def test_flywire_and_malecns_spellings_agree():
    """The two datasets spell the same transmitter differently; signs must still match."""
    for short, long in (("ACH", "acetylcholine"), ("GLUT", "glutamate"), ("GABA", "gaba")):
        assert sign_for(short) == sign_for(long)


def test_unknown_label_is_fatal_not_silent():
    """Defaulting an unrecognised transmitter would silently drop or invent edges."""
    with pytest.raises(UnknownNeurotransmitter):
        sign_for("kryptonite")
    assert sign_for("kryptonite", strict=False) == 0


def test_every_sign_is_in_range():
    assert set(NT_SIGN.values()) <= {-1, 0, +1}

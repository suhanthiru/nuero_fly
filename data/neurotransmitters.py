"""Neurotransmitter -> synaptic sign.

This is the most dangerous module in the project. Getting one neurotransmitter class
backwards produces a network that still runs, still spikes, and still looks plausible,
while being wrong in a way no smoke test catches. It is deliberately isolated here,
kept tiny, and covered by a dedicated test against a known-inhibitory pathway.

Mapping (following Shiu, Sterne et al., "Transforming a head direction signal into a
behavioral sequence: a connectome-constrained model of the Drosophila brain",
and the standard assignment used across the FlyWire modelling literature):

    acetylcholine  -> +1   principal fast excitatory transmitter in the fly CNS
    GABA           -> -1   Rdl / GABA-B, chloride and K+ conductances
    glutamate      -> -1   see note below
    histamine      -> -1   photoreceptor transmitter; ort / HisCl1 are chloride channels
    dopamine       ->  0   modulatory; not represented in a current-based LIF
    octopamine     ->  0   modulatory
    serotonin      ->  0   modulatory

Note on glutamate. The build spec listed glutamate on both the excitatory and the
inhibitory side. It is assigned inhibitory here because GluCl-alpha, a glutamate-gated
chloride channel, is the dominant postsynaptic receptor at fly central synapses. This is
the same choice Shiu et al. make, so it is also what the Phase 1 reproduction requires:
flipping it would break agreement with their reference implementation, which is precisely
why that reproduction is the gate for everything downstream.

Note on the modulators. Setting them to 0 removes those edges from the graph entirely.
That is a real, documented limitation and not a rounding decision - see the neuromodulation
hazard in README.md. It is what Shiu et al. do and it keeps us comparable to them.
"""

from __future__ import annotations

# Canonical sign per normalised neurotransmitter name. One dict, one place.
NT_SIGN: dict[str, int] = {
    "acetylcholine": +1,
    "gaba": -1,
    "glutamate": -1,
    "histamine": -1,
    "dopamine": 0,
    "octopamine": 0,
    "serotonin": 0,
    "unknown": 0,
}

# Dataset-specific spellings normalised onto the keys above. FlyWire Codex uses short
# uppercase codes; MaleCNS uses full lowercase names. Extend here rather than adding
# branches at call sites.
_ALIASES: dict[str, str] = {
    # FlyWire Codex nt_type
    "ach": "acetylcholine",
    "acetylcholin": "acetylcholine",
    "glut": "glutamate",
    "gaba": "gaba",
    "da": "dopamine",
    "dopamin": "dopamine",
    "oct": "octopamine",
    "ser": "serotonin",
    "5ht": "serotonin",
    "his": "histamine",
    # explicit "we do not know"
    "": "unknown",
    "unk": "unknown",
    "none": "unknown",
    "nan": "unknown",
    "unclear": "unknown",
}


class UnknownNeurotransmitter(ValueError):
    """Raised when a dataset contains a transmitter label we have no sign for.

    Deliberately fatal. Silently defaulting an unrecognised label to 0 would drop
    edges without anyone noticing; defaulting it to +1 would invent excitation.
    """


def normalise(nt: object) -> str:
    """Fold a raw dataset transmitter label onto a canonical NT_SIGN key."""
    if nt is None:
        return "unknown"
    text = str(nt).strip().lower()
    text = _ALIASES.get(text, text)
    return text


def sign_for(nt: object, *, strict: bool = True) -> int:
    """Signed multiplier for one transmitter label.

    strict=True (default) raises on an unrecognised label. Pass strict=False only in
    exploratory code, never in the loader path.
    """
    key = normalise(nt)
    if key not in NT_SIGN:
        if strict:
            raise UnknownNeurotransmitter(
                f"no sign defined for neurotransmitter {nt!r} (normalised: {key!r}). "
                f"Add it to NT_SIGN in data/neurotransmitters.py with a citation, "
                f"rather than defaulting it."
            )
        return 0
    return NT_SIGN[key]


def sign_table(labels) -> dict[str, int]:
    """Map an iterable of raw labels to their signs, for vectorised lookup."""
    return {label: sign_for(label) for label in set(labels)}

"""Lesion hook and subgraph extraction - the machinery the lesion sweep will lean on."""

from __future__ import annotations

import numpy as np

from data.cell_types import ids_for
from data.subset import neighbourhood, silence_mask, silence_structural, subgraph
from tests.conftest import requires_malecns

pytestmark = requires_malecns


def test_silence_mask_selects_exactly_the_named_type(malecns):
    mask = silence_mask(malecns, "DNp01")
    assert mask.sum() == 2
    assert set(malecns.annotations["cell_type"].to_numpy()[mask]) == {"DNp01"}


def test_silence_mask_leaves_the_connectome_untouched(malecns):
    before = malecns.weights.nnz
    silence_mask(malecns, "LC4")
    assert malecns.weights.nnz == before


def test_structural_silencing_removes_outgoing_not_incoming(malecns):
    lesioned = silence_structural(malecns, "LC4")
    lc4 = lesioned.indices_of(ids_for(lesioned, "LC4"))

    # weights[post, pre]: outgoing edges are the column, incoming are the row.
    assert lesioned.weights[:, lc4].nnz == 0
    assert lesioned.weights[lc4].nnz > 0, "LC4 should still receive input"
    assert lesioned.meta["lesion"]["n_neurons_silenced"] == len(lc4)


def test_structural_silencing_cuts_the_pathway_it_should(malecns):
    gf_ids = ids_for(malecns, "DNp01")
    intact = malecns.weights[malecns.indices_of(gf_ids)][
        :, malecns.indices_of(ids_for(malecns, "LC4"))
    ].sum()
    assert intact > 0

    lesioned = silence_structural(malecns, "LC4")
    after = lesioned.weights[lesioned.indices_of(gf_ids)][
        :, lesioned.indices_of(ids_for(lesioned, "LC4"))
    ].sum()
    assert after == 0


def test_subgraph_is_induced_and_order_preserving(malecns):
    wanted = np.concatenate([ids_for(malecns, t) for t in ("LC4", "LPLC2", "DNp01")])
    sub = subgraph(malecns, wanted)

    assert sub.n_neurons == len(np.unique(wanted))
    assert np.all(np.diff(sub.ids) > 0)
    assert np.array_equal(np.asarray(sub.annotations.index), sub.ids)

    # LC4 -> DNp01 must survive extraction with its weight intact.
    full = malecns.weights[malecns.indices_of(ids_for(malecns, "DNp01"))][
        :, malecns.indices_of(ids_for(malecns, "LC4"))
    ].sum()
    cut = sub.weights[sub.indices_of(ids_for(sub, "DNp01"))][
        :, sub.indices_of(ids_for(sub, "LC4"))
    ].sum()
    assert cut == full


def test_neighbourhood_reaches_known_partners(malecns):
    gf_id = int(ids_for(malecns, "DNp01", side="R")[0])
    around = neighbourhood(malecns, [gf_id], hops=1)
    types = set(malecns.annotations.loc[around, "cell_type"])
    assert "LC4" in types      # presynaptic
    assert "TTMn" in types     # postsynaptic
    assert len(around) > 100

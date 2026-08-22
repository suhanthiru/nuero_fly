"""Loader tests: counts, round-trips, orientation, sign, threshold, determinism."""

from __future__ import annotations

import numpy as np
import pytest

from data.cell_types import ids_for, matching_types
from data.loader import build_connectome, load_connectome
from tests.conftest import requires_malecns

pytestmark = requires_malecns


class TestCounts:
    def test_total_neuron_count_matches_published(self, malecns):
        """MaleCNS v1.0 is published as 'more than 166,000' neurons."""
        assert 150_000 < malecns.n_neurons < 175_000

    def test_glia_are_excluded(self, malecns):
        assert (malecns.annotations["status"] == "Traced").all()

    @pytest.mark.parametrize(
        ("cell_type", "expected"),
        [
            # Large individually-identified cells: exactly one per hemisphere. These are
            # textbook counts, not estimates, so they are exact assertions.
            ("DNp01", 2),   # the giant fiber
            ("DNp02", 2),
            ("DNp04", 2),
            ("DNp11", 2),
            ("TTMn", 2),    # tergotrochanteral motor neuron
        ],
    )
    def test_identified_neuron_counts_are_exact(self, malecns, cell_type, expected):
        assert len(ids_for(malecns, cell_type)) == expected

    @pytest.mark.parametrize(
        ("cell_type", "low", "high"),
        [
            # Population types, per hemisphere, bracketed by published counts from FAFB
            # (55 LC4 / 108 LPLC2) and hemibrain (71 LC4 / 85 LPLC2).
            ("LC4", 45, 85),
            ("LPLC2", 75, 120),
            ("LC6", 45, 80),
            ("LC22", 25, 55),
        ],
    )
    def test_population_counts_are_in_published_range(self, malecns, cell_type, low, high):
        for side in ("L", "R"):
            n = len(ids_for(malecns, cell_type, side=side))
            assert low <= n <= high, f"{cell_type}_{side} n={n} outside [{low}, {high}]"

    def test_every_escape_type_is_bilateral(self, malecns):
        for cell_type in ("LC4", "LC6", "LC22", "LPLC2", "DNp01", "TTMn"):
            left = len(ids_for(malecns, cell_type, side="L"))
            right = len(ids_for(malecns, cell_type, side="R"))
            assert left > 0 and right > 0, f"{cell_type} is not bilateral: L={left} R={right}"


class TestRoundTrip:
    def test_id_to_index_to_id(self, malecns):
        sample = malecns.ids[:: max(1, malecns.n_neurons // 500)]
        for body_id in sample:
            assert malecns.ids[malecns.index_of(body_id)] == body_id

    def test_id_to_cell_type_round_trip(self, malecns):
        for cell_type in ("DNp01", "TTMn", "LC4", "LPLC2"):
            ids = ids_for(malecns, cell_type)
            assert len(ids) > 0
            assert (malecns.annotations.loc[ids, "cell_type"] == cell_type).all()

    def test_ids_are_sorted_so_searchsorted_is_valid(self, malecns):
        """index_of and indices_of both rely on this; a silent violation would corrupt
        every lookup rather than fail loudly."""
        assert np.all(np.diff(malecns.ids) > 0)

    def test_annotations_align_with_ids(self, malecns):
        assert np.array_equal(np.asarray(malecns.annotations.index), malecns.ids)

    def test_missing_id_raises(self, malecns):
        with pytest.raises(KeyError):
            malecns.index_of(-1)


class TestOrientation:
    def test_weights_are_post_by_pre(self, malecns):
        """LC4 is presynaptic to the giant fiber, not the other way round.

        If the matrix were transposed this assertion would flip, and every downstream
        result would be wrong while still looking plausible.
        """
        lc4 = malecns.indices_of(ids_for(malecns, "LC4"))
        gf = malecns.indices_of(ids_for(malecns, "DNp01"))
        forward = malecns.weights[gf][:, lc4].sum()      # rows=GF(post), cols=LC4(pre)
        backward = malecns.weights[lc4][:, gf].sum()     # rows=LC4(post), cols=GF(pre)
        assert forward > 0
        assert backward == 0

    def test_upstream_returns_presynaptic_partners(self, malecns):
        gf_id = int(ids_for(malecns, "DNp01", side="R")[0])
        partners, weights = malecns.upstream(gf_id)
        types = set(malecns.annotations.loc[partners, "cell_type"])
        assert "LC4" in types
        assert weights.size == partners.size


class TestSignConvention:
    def test_cholinergic_projection_is_excitatory(self, malecns):
        """LC4 and LPLC2 are cholinergic; their drive onto the GF must be positive."""
        gf = malecns.indices_of(ids_for(malecns, "DNp01"))
        for cell_type in ("LC4", "LPLC2"):
            pre = malecns.indices_of(ids_for(malecns, cell_type))
            assert malecns.weights[gf][:, pre].sum() > 0

    def test_a_gabaergic_cell_type_is_inhibitory(self, malecns):
        """The spec's first named hazard. A known inhibitory population must come out
        negative; if the sign table were inverted this is where it would show."""
        inhibitory = [
            t for t in matching_types(malecns, "IN13A*") if t
        ]  # 13A hemilineage is GABAergic
        assert inhibitory, "no 13A interneurons found to test against"
        idx = malecns.indices_of(ids_for(malecns, inhibitory))
        outgoing = malecns.weights[:, idx]
        negative_share = (outgoing.data < 0).mean()
        assert negative_share > 0.9, f"only {negative_share:.0%} of 13A output is inhibitory"

    def test_both_signs_are_present(self, malecns):
        assert (malecns.weights.data > 0).any()
        assert (malecns.weights.data < 0).any()

    def test_no_zero_weights_stored(self, malecns):
        """Modulatory edges are dropped, not stored as explicit zeros."""
        assert (malecns.weights.data != 0).all()


class TestThreshold:
    def test_higher_threshold_removes_edges(self):
        loose = load_connectome("malecns-1.0", min_synapses=1)
        tight = load_connectome("malecns-1.0", min_synapses=20)
        assert tight.n_edges < loose.n_edges
        assert tight.n_neurons == loose.n_neurons  # thresholding edges, not neurons

    def test_threshold_is_recorded_in_metadata(self, malecns):
        assert malecns.meta["min_synapses"] == 5
        assert malecns.meta["orientation"] == "weights[post, pre]"
        assert malecns.meta["nt_sign"]["glutamate"] == -1


class TestDeterminism:
    def test_rebuild_is_bit_identical(self):
        """Non-negotiable: the lesion sweep compares runs against each other."""
        a = build_connectome("malecns-1.0")
        b = build_connectome("malecns-1.0")
        assert np.array_equal(a.ids, b.ids)
        assert np.array_equal(a.weights.indptr, b.weights.indptr)
        assert np.array_equal(a.weights.indices, b.weights.indices)
        assert np.array_equal(a.weights.data, b.weights.data)

    def test_cache_matches_fresh_build(self, malecns):
        fresh = build_connectome("malecns-1.0")
        assert np.array_equal(fresh.ids, malecns.ids)
        assert np.array_equal(fresh.weights.data, malecns.weights.data)
        assert np.array_equal(fresh.weights.indices, malecns.weights.indices)

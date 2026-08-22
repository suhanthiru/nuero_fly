"""Shared fixtures.

Building the MaleCNS matrix takes ~23 s from cold and streams a 1 GB file, so it is built
once per session and shared. Tests must treat the fixture as read-only.
"""

from __future__ import annotations

import pytest

from data.loader import load_connectome
from data.sources import MaleCNSSource


def _have(source) -> bool:
    return source.root.exists() and any(source.root.iterdir())


requires_malecns = pytest.mark.skipif(
    not _have(MaleCNSSource()),
    reason="MaleCNS flat files absent; run scripts/fetch_data.sh",
)


@pytest.fixture(scope="session")
def malecns():
    return load_connectome("malecns-1.0")

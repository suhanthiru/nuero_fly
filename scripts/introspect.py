"""Phase 0.1 - report the actual schemas of the downloaded flat files.

Nothing in the loader may assume a column name that has not been seen here first.
Reads the 1 GB edge list via the Arrow IPC footer so it never loads the whole table.
"""

from __future__ import annotations

import gzip
import itertools
from pathlib import Path

import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def feather_schema(path: Path, *, peek_rows: int = 3) -> None:
    """Schema + first batch only. Does not materialise the full table."""
    rule(f"{path.name}  ({path.stat().st_size / 1e6:.0f} MB)")
    with path.open("rb") as fh:
        reader = ipc.open_file(fh)
        print(f"rows={reader.num_record_batches} batches")
        for field in reader.schema:
            print(f"  {field.name:<34} {field.type}")
        if reader.num_record_batches:
            batch = reader.get_batch(0)
            print(f"\nfirst {peek_rows} rows of batch 0 (batch has {batch.num_rows} rows):")
            print(batch.slice(0, peek_rows).to_pandas().to_string())


def feather_full(path: Path, *, peek_rows: int = 5) -> pa.Table:
    rule(f"{path.name}  ({path.stat().st_size / 1e6:.0f} MB)  [full read]")
    table = feather.read_table(path)
    print(f"shape: {table.num_rows} rows x {table.num_columns} cols")
    for field in table.schema:
        print(f"  {field.name:<34} {field.type}")
    print(f"\nfirst {peek_rows} rows:")
    print(table.slice(0, peek_rows).to_pandas().to_string())
    return table


def csv_gz_head(path: Path, n: int = 4) -> None:
    rule(f"{path.name}  ({path.stat().st_size / 1e6:.0f} MB)")
    with gzip.open(path, "rt") as fh:
        for line in itertools.islice(fh, n):
            print("  " + line.rstrip())


def value_counts(table: pa.Table, column: str, top: int = 25) -> None:
    if column not in table.column_names:
        print(f"  [no column {column!r}]")
        return
    series = table.column(column).to_pandas()
    counts = series.value_counts(dropna=False)
    print(f"\n{column}: {series.nunique(dropna=True)} distinct, top {top}:")
    print(counts.head(top).to_string())


def main() -> None:
    mcns = RAW / "malecns"
    fw = RAW / "flywire783"

    feather_schema(mcns / "connectome-weights.feather")
    ann = feather_full(mcns / "body-annotations.feather")
    feather_schema(mcns / "body-neurotransmitters.feather")

    rule("MaleCNS annotation vocabularies")
    for col in ("superclass", "class", "side", "somaSide", "consensusNt", "predictedNt"):
        value_counts(ann, col)

    for name in ("connections", "classification", "consolidated_cell_types", "coordinates"):
        csv_gz_head(fw / f"{name}.csv.gz")


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# Fetch the public connectome flat files for Phase 0.
#
# All sources are anonymous public buckets; no credentials, no CAVE, no login.
#   MaleCNS v1.0  - Janelia FlyEM, CC-BY. https://male-cns.janelia.org/download/
#   FlyWire 783   - Codex flat files.     https://codex.flywire.ai/
#
# Citation obligations for both datasets are recorded in README.md.
# Downloads resume (curl -C -), so re-running is cheap and idempotent.

set -euo pipefail

RAW="$(cd "$(dirname "$0")/.." && pwd)/data/raw"
MCNS_BASE="https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
FW_BASE="https://storage.googleapis.com/flywire-data/codex/data/fafb/783"

mkdir -p "$RAW/malecns" "$RAW/flywire783"

get() {
  local url="$1" dest="$2"
  if [ -s "$dest" ]; then
    echo "  have  $(basename "$dest")"
    return 0
  fi
  echo "  get   $(basename "$dest")"
  curl -fsSL -C - --retry 3 --retry-delay 2 -o "$dest" "$url"
}

echo "== MaleCNS v1.0 =="
get "$MCNS_BASE/connectome-weights-male-cns-v1.0-minconf-0.5.feather" \
    "$RAW/malecns/connectome-weights.feather"
get "$MCNS_BASE/body-annotations-male-cns-v1.0-minconf-0.5.feather" \
    "$RAW/malecns/body-annotations.feather"
get "$MCNS_BASE/body-neurotransmitters-male-cns-v1.0.feather" \
    "$RAW/malecns/body-neurotransmitters.feather"

echo "== FlyWire 783 (Codex) =="
for f in connections classification consolidated_cell_types coordinates cell_stats; do
  get "$FW_BASE/$f.csv.gz" "$RAW/flywire783/$f.csv.gz"
done

echo "== sizes =="
du -h "$RAW"/*/* | sort -k2

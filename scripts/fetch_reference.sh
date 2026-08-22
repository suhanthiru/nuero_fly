#!/usr/bin/env bash
# Fetch the Shiu et al. reference model's own connectivity tables.
#
# Phase 1 reproduces their published sugar -> proboscis result. To make a mismatch
# unambiguous, our LIF is first run on *their* preprocessed graph rather than ours, which
# isolates the neuron model from the loader. Our loader is then compared against the same
# tables separately.
#
# Source: https://github.com/philshiu/Drosophila_brain_model  (~104 MB)

set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/data/raw/shiu"
BASE_LFS="https://media.githubusercontent.com/media/philshiu/Drosophila_brain_model/main"
BASE_RAW="https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main"

mkdir -p "$DEST"

get() {
  local name="$1" dest="$DEST/$1"
  if [ -s "$dest" ]; then
    echo "  have  $name"
    return 0
  fi
  echo "  get   $name"
  # Large files in that repo are served through git-lfs; fall back to plain raw.
  curl -fsSL --retry 3 -o "$dest" "$BASE_LFS/$name" \
    || curl -fsSL --retry 3 -o "$dest" "$BASE_RAW/$name"
}

# Release 630 is what their paper ran, and their published neuron ID lists are 630 root
# ids - FlyWire root ids do not survive across releases. Reproducing the paper therefore
# means reproducing it on 630; 783 is kept for the loader comparison.
get 2023_03_23_completeness_630_final.csv
get 2023_03_23_connectivity_630_final.parquet

get Completeness_783.csv
get Connectivity_783.parquet

ls -lh "$DEST"

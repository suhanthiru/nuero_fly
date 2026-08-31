#!/usr/bin/env bash
# Fetch the flybody Drosophila body model.
#
# An anatomically-detailed MuJoCo model of Drosophila melanogaster - 67 body parts, 66
# joints - from Google DeepMind and HHMI Janelia (Vaxenburg et al., Nature 2025). Apache-2.0.
#
#   https://github.com/google-deepmind/mujoco_menagerie/tree/main/flybody
#   https://github.com/TuragaLab/flybody          (source of truth)
#
# Used here for GEOMETRY ONLY. The escape is still a rigid-body impulse; this supplies a
# body to look at, not leg mechanics. scripts/bake_fly_mesh.py bakes it into a single posed
# mesh for the viewer, and the viewer labels it as a shell.
#
# ~140 MB of OBJ meshes. Resumable, so re-running is cheap.

set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/data/raw/flybody"
BASE="https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/flybody"
API="https://api.github.com/repos/google-deepmind/mujoco_menagerie/contents/flybody/assets"

mkdir -p "$DEST/assets"

get() {
  local url="$1" dest="$2"
  [ -s "$dest" ] && return 0
  curl -fsSL --retry 3 -o "$dest" "$url"
}

echo "== model =="
for f in fruitfly.xml scene.xml LICENSE README.md; do
  echo "  $f"
  get "$BASE/$f" "$DEST/$f"
done

echo "== meshes =="
python3 - "$API" <<'PY' > "$DEST/.asset-list"
import json, sys, urllib.request
with urllib.request.urlopen(sys.argv[1]) as response:
    for item in json.load(response):
        print(item["name"])
PY

count=$(wc -l < "$DEST/.asset-list")
echo "  $count meshes"
# Modest parallelism: enough to be quick, not enough to look like abuse.
xargs -a "$DEST/.asset-list" -P 8 -I {} \
  curl -fsSL --retry 3 -o "$DEST/assets/{}" "$BASE/assets/{}" -z "$DEST/assets/{}"

du -sh "$DEST"

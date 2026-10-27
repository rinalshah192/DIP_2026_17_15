#!/usr/bin/env bash
# Downloads the AITEX Fabric Image Database (https://www.aitex.es/afid/) into data/raw.
# Needs curl and 7z (macOS: brew install p7zip; Ubuntu: apt install p7zip-full).
# Please cite: Silvestre-Blanes et al., AUTEX Research Journal 19(4), 2019.
set -euo pipefail

BASE="https://www.aitex.es/wp-content/uploads/2019/07"
DEST="${1:-$(cd "$(dirname "$0")/../.." && pwd)/data/raw}"
mkdir -p "$DEST"
cd "$DEST"

for name in NODefect_images Defect_images Mask_images; do
  if [ -d "$name" ]; then
    echo "$name: already present, skipped"
    continue
  fi
  curl -fL --retry 3 -o "$name.7z" "$BASE/$name.7z"
  7z x -y "$name.7z" > /dev/null
  rm "$name.7z"
  echo "$name: extracted"
done

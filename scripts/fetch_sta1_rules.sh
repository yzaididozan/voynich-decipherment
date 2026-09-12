#!/usr/bin/env bash
set -euo pipefail

DEST="${1:-data/reference/sta1}"
BASE="https://www.voynich.nu/software/bitrans"

mkdir -p "$DEST"

for name in \
  STA-Eva_def.bit \
  STA-EvaT_def.bit \
  STA-v101_def.bit
do
  echo "Fetching $name"
  curl -fL --retry 3 \
    "$BASE/$name" \
    -o "$DEST/$name"
done

echo
echo "SHA-256:"
if command -v shasum >/dev/null 2>&1; then
  shasum -a 256 "$DEST"/*.bit
else
  sha256sum "$DEST"/*.bit
fi

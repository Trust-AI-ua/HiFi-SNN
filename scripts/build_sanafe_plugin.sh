#!/usr/bin/env bash
# Build src/sanafe_plugin/loihi_lif_delayed.cpp against the headers of SANA-FE v2.2.9 (the
# version pinned in requirements-neuro.txt). Output: src/sanafe_plugin/build/ (git-ignored).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
B="$REPO/src/sanafe_plugin/build"
mkdir -p "$B"
if [ ! -d "$B/sanafe-src-v2.2.9" ]; then
  git clone -q --depth 1 --branch v2.2.9 https://github.com/SLAM-Lab/SANA-FE.git "$B/sanafe-src-v2.2.9"
fi
g++ -std=c++17 -O2 -shared -fPIC -Wall -I "$B/sanafe-src-v2.2.9/src" \
    "$REPO/src/sanafe_plugin/loihi_lif_delayed.cpp" -o "$B/libloihi_lif_delayed.so"
echo "built $B/libloihi_lif_delayed.so"

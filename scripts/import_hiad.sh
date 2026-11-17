#!/usr/bin/env bash
# Merge the exact public upstream history, retaining CSR changes and license.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
upstream=https://github.com/cnulab/HiAD.git
revision=54f7c2e49956cffa231d1ec6e19a373c68a6254f
if git merge-base --is-ancestor "$revision" HEAD 2>/dev/null; then
  echo "HiAD already imported"
  exit 0
fi
git fetch --no-tags "$upstream" "$revision"
git merge --allow-unrelated-histories --no-edit "$revision"

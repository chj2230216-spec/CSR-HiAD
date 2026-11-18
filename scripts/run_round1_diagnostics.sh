#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
gpu="${GPU:-0}"
batch="${BATCH_SIZE:-2}"
out="${OUT_ROOT:-results/round1_shared}"
data="${DATA_ROOT:-data/MVTec-2K}"
# Override SEEDS="42 2025 2026" for frozen-seed confirmation.
read -r -a seeds <<< "${SEEDS:-42}"
mkdir -p "$out"
for seed in "${seeds[@]}"; do
  for cat in bottle capsule; do
    dir="$out/$cat/seed$seed"
    log="$out/${cat}_seed${seed}.log"
    if [ -e "$dir" ] || [ -e "$log" ]; then
      echo "Refusing overwrite: $dir or $log" >&2
      exit 1
    fi
    python runs/run_csr_diagnostics.py --data_root "$data" --category "$cat" \
      --gpu "$gpu" --seed "$seed" --batch_size "$batch" --audit_alignment \
      --out "$dir" "$@" 2>&1 | tee "$log"
  done
done
python scripts/summarize_csr_diagnostics.py --root "$out" --out "$out/summary"

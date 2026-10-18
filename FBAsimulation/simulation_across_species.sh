#!/bin/bash

set -euo pipefail

for seed in {0..0}; do
    echo "Running seed=${seed}..."

    mapfile -t flux_limits < <(
        python - "$seed" <<'PY'
import sys
import numpy as np

seed = int(sys.argv[1])
rng = np.random.default_rng(seed)

flux_limits = rng.uniform(
    1000,
    1000,
    size=205,
)

for value in flux_limits:
    print(value)
PY
    )

    python FBAsimulation/simulation_across_species_by_model.py \
        --n 1 \
        --solver gurobi \
        --medium-count 423 \
        --seed-base "$seed" \
        --flux-limits "${flux_limits[@]}" \
        --output-dir FBAsolution

    # python FBAsimulation/simulation_across_species_by_model.py \
    #     --n 40 \
    #     --solver gurobi \
    #     --medium-count 60 \
    #     --seed-base "$seed" \
    #     --flux-limits "${flux_limits[@]}" \
    #     --output-dir FBAsolution/Perturb_P
done
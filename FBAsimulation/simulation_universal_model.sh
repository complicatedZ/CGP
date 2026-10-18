#!/bin/bash

set -euo pipefail

for seed in {2..20}; do
    echo "Running seed=${seed}..."

    python plot_figure/p_w_test/simulation_universal_model_by_file.py "$seed"

done
#!/bin/bash
# Check all 9 models run on five timesteps, in the same order as the full campaign.
# Runs on CPU and uses no GPU credits. Scores are not meaningful.
# Replaces any earlier cache/small-test and results/small-test.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/../repro.env"
source "$MLSTEP_CONDA_SH"
conda activate "$MLSTEP_CONDA_ENV"
cd "$MLSTEP_ROOT"

export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

cache=cache/small-test
results=results/small-test
split=(--train-timesteps 2:4 --validation-timesteps 5:6)
rm -rf "$cache" "$results"

echo "== Prepare the data"
for method in physical stretch128; do
    python -m mlstep.cache --data-dir "$MLSTEP_DATA_DIR" --cache-dir "$cache/$method" \
        --preprocess "$method" "${split[@]}"
done
python -m mlstep.cache --preprocess ple --physical-cache "$cache/physical" --cache-dir "$cache/ple_all"
python -m mlstep.cache --preprocess ple --physical-cache "$cache/physical" \
    --cache-dir "$cache/ple_tracer_residual" --embedding-groups tracer residual

echo "== Train XGBoost"
python -m mlstep.xgb train --data-dir "$MLSTEP_DATA_DIR" --output-dir "$results/xgb" "${split[@]}" \
    --device cpu --rounds 10 --patience 5 --threads 4 --recalls 0.90 0.95

echo "== Prepare the teacher targets"
python -m mlstep.xgb targets --model "$results/xgb/model.ubj" --cache-dir "$cache/physical" \
    --output-dir "$cache/teacher" --device cpu

for variant in physical stretch128 ple_all ple_tracer_residual; do
    method=$variant
    [[ $variant == ple_* ]] && method=ple
    for mode in supervised distilled; do
        echo "== Train and score: $mode $variant"
        teacher=()
        [[ $mode == distilled ]] && teacher=(--teacher-dir "$cache/teacher")
        output="$results/$mode/$variant"
        python -m mlstep.student --cache-dir "$cache/$variant" --output-dir "$output" \
            --device cpu --epochs 1 --patience 1 "${teacher[@]}"
        python -m mlstep.generalization operating-points \
            --model "$output/model_${method}_seed0.pt" --cache-dir "$cache/$variant" \
            --device cpu --recalls 0.90 0.95 --output "$output/validation.json"
    done
done

echo "Done. Results are in $results/"

#!/usr/bin/env bash
set -euo pipefail

download() {
    model=$1
    weights=$2
    shift 2
    for step in "$@"; do
        echo "$model step$step"
        hf download "$model" \
            --revision "step$step" \
            --include "$weights" \
            --include "*.json" \
            --include "tokenizer*" \
            --include "*.txt" \
            --include "*.model"
    done
}

case "${1:-}" in
    experiment1-1b)
        download EleutherAI/pythia-1b "*.safetensors" \
            256 512 1000 2000 5000 8000 18000 50000 143000
        ;;
    experiment1-410m)
        seed=${2:-0}
        if [ "$seed" = 0 ]; then
            download EleutherAI/pythia-410m "*.safetensors" \
                256 512 1000 2000 5000 8000 18000 50000 143000
        else
            download "EleutherAI/pythia-410m-seed$seed" "pytorch_model*.bin" \
                256 512 1000 2000 5000 8000 18000 50000 143000
        fi
        ;;
    experiment1-outliers)
        for seed in 3 4; do
            download "EleutherAI/pythia-410m-seed$seed" "pytorch_model*.bin" \
                53000 63000 73000 83000 93000 103000 113000 123000 133000
        done
        ;;
    experiment2-routing)
        download EleutherAI/pythia-1b "*.safetensors" \
            1000 2000 5000 8000 18000 50000 70000 143000
        ;;
    experiment2-final)
        download EleutherAI/pythia-1b "*.safetensors" 143000
        ;;
    *)
        echo "usage:"
        echo "  $0 experiment1-1b"
        echo "  $0 experiment1-410m SEED   # SEED=0,...,9"
        echo "  $0 experiment1-outliers"
        echo "  $0 experiment2-routing"
        echo "  $0 experiment2-final"
        exit 1
        ;;
esac

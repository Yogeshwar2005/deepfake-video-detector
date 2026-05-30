#!/bin/bash

echo "Starting all training runs..."

for seed in 1 2 3 4 5
do
    echo "Running seed $seed"

    python3 training.py \
        --epochs 10 \
        --seed $seed \
        --augment \
        --loss bce \
        --pos-weight 0.15736747005

done

echo "All runs completed."
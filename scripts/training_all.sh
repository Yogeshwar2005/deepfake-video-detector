#!/bin/bash

echo "Starting all training runs..."

for aug in 0 1; do
    for sampler in 0 1; do
        for pw in 0 1; do
            echo "=========================================="
            echo "augment=$aug sampler=$sampler pos_weight=$pw"
            echo "=========================================="
            python3 training.py -e 10 -a $aug -s $sampler -pw $pw
        done
    done
done

echo "All training runs complete."
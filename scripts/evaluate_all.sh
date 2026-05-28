#!/bin/bash

for checkpoint in ../checkpoints/best__*.pth; do
    echo ""
    ./evaluate_one.sh "$checkpoint"
done
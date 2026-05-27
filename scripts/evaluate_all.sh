#!/bin/bash

for checkpoint in ../checkpoints/best__*.pth; do
    echo ""
    echo "########################################## "
    echo "MODEL: $checkpoint"
    echo "##########################################"
    ./evaluate_one.sh "$checkpoint"
done
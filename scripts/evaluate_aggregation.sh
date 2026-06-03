#!/bin/bash

CHECKPOINT="../models/efficientnet_b0/results/best__seed-3_e-10_augment-on_sampler-off_loss-bce_pw-0.15736747005.pth"

for AGG in max mean median topk
do
for FRAMES in 8 16 32
do
echo "========================================="
echo "Aggregation: $AGG | Frames: $FRAMES"
echo "========================================="


    python3 evaluate_videos.py \
        --validate \
        --checkpoint "$CHECKPOINT" \
        --aggregation "$AGG" \
        --num-frames "$FRAMES"

done


done

#!/bin/bash

MODEL=$1


echo "########################################## "
echo "MODEL: $MODEL"
echo "##########################################"

echo
echo "=========================================="
echo "Clean test set, checkpoint threshold"
echo "=========================================="
python3 ../src/evaluate.py -l "$MODEL" --test

echo
echo "=========================================="
echo "Clean test set, threshold = 0.5"
echo "=========================================="
python3 ../src/evaluate.py -l "$MODEL" -t 0.5 --test

echo
echo "=========================================="
echo "Compressed test set, checkpoint threshold"
echo "=========================================="
python3 ../src/evaluate.py -l "$MODEL" -c 1 --test

echo
echo "=========================================="
echo "Compressed test set, threshold = 0.5"
echo "=========================================="
python3 ../src/evaluate.py -l "$MODEL" -t 0.5 -c 1 --test

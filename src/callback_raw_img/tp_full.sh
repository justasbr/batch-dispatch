#!/bin/bash
# Basic range in for loop
for value in $(seq 33 39); do
  for frameworkName in tf keras torch; do
    for convNet in alexnet; do
      OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 python throughput.py --framework $frameworkName --num_batches 1000 --batch_size $value --model $convNet
    done
  done
done
echo All done

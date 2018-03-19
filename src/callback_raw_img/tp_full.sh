#!/bin/bash
# Basic range in for loop
for value in $(seq 1 3); do
  for frameworkName in keras torch tf; do
    for convNet in resnet vgg inception alexnet; do
      python throughput.py --framework $frameworkName --num_batches 10 --batch_size $value --model $convNet
    done
  done
done
echo All done

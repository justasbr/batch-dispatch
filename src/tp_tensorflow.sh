#!/bin/bash
# Basic range in for loop
for value in $(seq 1 10); do
  for frameworkName in keras torch; do
    for convNet in resnet vgg inception; do
      python throughput.py --framework $frameworkName --num_batches 10000 --batch_size $value --model $convNet
    done
  done
done
for value in $(seq 7 10); do
  for frameworkName in tf; do
    for convNet in resnet vgg inception; do
      python throughput.py --framework $frameworkName --num_batches 10000 --batch_size $value --model $convNet
    done
  done
done
echo All done

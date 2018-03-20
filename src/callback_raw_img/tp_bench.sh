#!/bin/bash
# Basic range in for loop
for value in $(seq 1 8); do
  for frameworkName in keras; do
    for convNet in resnet inception; do # alexnet resnet vgg inception; do
      # MKL_NUM_THREADS=8 
      python throughput.py --framework $frameworkName --num_batches 1000 --batch_size $value --model $convNet
    done
  done
  
  #for frameworkName in tf; do
  #  for convNet in alexnet; do
  #    python throughput.py --framework $frameworkName --num_batches 1000 --batch_size $value --model $convNet
  #  done
  #done
done
#for value in $(seq 4 8); do
#  for frameworkName in tf keras; do
#    for convNet in alexnet resnet vgg inception; do
#      #MKL_NUM_THREADS=8 
#      pythonthroughput.py --framework $frameworkName --num_batches 1000 --batch_size $value --model $convNet
#    done
#  done
#done
echo All done

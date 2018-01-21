#!/bin/bash
# Basic range in for loop
for value in $(seq 1 10)
do
echo $value
python throughput.py --framework torch --num_batches 10000 --batch_size $value --model alexnet
done
echo All done

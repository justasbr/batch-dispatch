for value in 1 2 4 8 16 32 64 128 256 512 1024; do
  for frameworkName in tf; do
    for convNet in alexnet; do
      python benchmark.py --framework $frameworkName --num_batches 100 --batch_size $value --model $convNet
    done
  done
done

echo All Done

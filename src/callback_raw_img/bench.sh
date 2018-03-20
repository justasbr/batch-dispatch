for frameworkName in tf torch trt; do
  for convNet in alexnet; do # vgg resnet inception; do
    for value in 256 512 1024; do #1 2 4 8 16 32 64 128; do
      python benchmark.py --framework $frameworkName --num_batches 321 --batch_size $value --model $convNet
    done
  done
done
echo All Done

for frameworkName in tf; do #tf keras torch trt; do # tf keras torch trt; do #tf keras torch; do #f; do #keras tf; do # torch; do # tf keras torch; do
  for convNet in alexnet; do #alexnet vgg resnet inception; do #inference; do # vgg resnet; do
    for value in 28 32 36; do #5 6 8 10 12; do # 6 8 10 12 16 20 24; do
      python throughput.py --framework $frameworkName --num_batches 2000 --batch_size $value --model $convNet
    done
  done
done
echo All Done

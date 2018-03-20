for model in inception; do # alexnet vgg resnet inception; do
  for num in 1 2 4 8 16 32 64 128 256; do #256 512; do #1 2 4 8 16 32 64 128; do
    python benchmark.py --framework trt --model $model --batch_size $num -n 100
  done
done

#python benchmark.py --framework trt --model alexnet --batch_size 512 -n 100
#python benchmark.py --framework trt --model alexnet --batch_size 1024 -n 100

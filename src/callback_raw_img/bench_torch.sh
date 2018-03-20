for model in alexnet vgg resnet inception; do
  for num in 1 2 4 8 16 32 64; do
    python torch_benchmark.py --framework torch --model $model --batch_size $num -n 1 --trace True
  done
done


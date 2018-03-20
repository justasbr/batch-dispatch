for frameworkName in tf keras; do #tf keras torch; do #f; do #keras tf; do # torch; do # tf keras torch; do 
  for convNet in inception; do #inference; do # vgg resnet; do
    for value in 1 2; do
      DINTRA=16 OMP_NUM_THREADS=16 KMP_BLOCKTIME=100 python throughput.py --framework $frameworkName --num_batches 1000 --batch_size $value --model $convNet
    done
  done
  #for convNet in alexnet; do
  #  for value in 1 2 3 4 6 8 12 16 20 25 32 40 50; do
  #    DINTRA=16 OMP_NUM_THREADS=16 KMP_BLOCKTIME=100 python throughput.py --framework $frameworkName --num_batches 1000 --batch_size $value --model $convNet
 #   done
  #done
done
echo All Done

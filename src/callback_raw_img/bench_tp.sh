for numThreads in 16; do
	for frameworkName in tf keras torch; do #f; do #keras tf; do # torch; do # tf keras torch; do 
	  for convNet in resnet inception; do
	    for value in 64 128; do
	      DINTRA=$numThreads OMP_NUM_THREADS=$numThreads KMP_BLOCKTIME=100 python benchmark.py --framework $frameworkName --num_batches 153 --batch_size $value --model $convNet
	    done
	  done
       done
done
echo All Done

for numThreads in 1 2; do
	for frameworkName in tf keras torch; do #f; do #keras tf; do # torch; do # tf keras torch; do 
	   for convNet in resnet inception; do
	     for value in 1 2 4 8 16 32; do
	       DINTRA=$numThreads OMP_NUM_THREADS=$numThreads KMP_BLOCKTIME=100 python benchmark.py --framework $frameworkName --num_batches 11 --batch_size $value --model $convNet
	     done
	   done

        done
done
for numThreads in 4 8 16; do
	for frameworkName in tf keras torch; do #f; do #keras tf; do # torch; do # tf keras torch; do 
	   for convNet in resnet inception; do
	     for value in 1 2 4 8 16 32; do
	       DINTRA=$numThreads OMP_NUM_THREADS=$numThreads KMP_BLOCKTIME=100 python benchmark.py --framework $frameworkName --num_batches 23 --batch_size $value --model $convNet
	     done
	   done

        done
done
echo All Done

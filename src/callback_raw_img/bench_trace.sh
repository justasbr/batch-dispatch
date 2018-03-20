# : <<'END'
for value in 128; do # 2 4 8 16 32 64 128; do
  for frameworkName in keras; do #tf keras torch; do
    for convNet in alexnet vgg resnet inception; do
      MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark.py --framework $frameworkName --num_batches 2 --batch_size $value --model $convNet --trace True
    done
  done
done

for value in 256 512 1024; do
  for frameworkName in keras; do # tf keras torch; do
    for convNet in alexnet; do
      MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark.py --framework $frameworkName --num_batches 2 --batch_size $value --model $convNet --trace True
    done
  done
done
# END
mv bench_logs/keraslog.out bench_logs/keraslog.out_single
# : <<'END'
for value in 1 2 4 8 16 32 64 128; do
  for frameworkName in keras; do # keras torch; do
    for convNet in alexnet vgg resnet inception; do
      echo Performance
      MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark.py --framework $frameworkName --num_batches 52 --batch_size $value --model $convNet
    done
  done
done

for value in 256 512 1024; do
  for frameworkName in keras; do
    for convNet in alexnet; do
      echo Performance
      MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python benchmark.py --framework $frameworkName --num_batches 52 --batch_size $value --model $convNet
    done
  done
done
# END

echo All Done

for x in /sys/devices/system/cpu/cpu*/online; do
  echo 0>"$x"
done

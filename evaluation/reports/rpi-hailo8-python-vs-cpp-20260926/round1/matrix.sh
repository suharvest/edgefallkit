#!/bin/bash
# full matrix; run as root
B=/home/harvest/hailo-bench
trap 'systemctl stop hailort.service; docker rm -f hbench-A >/dev/null 2>&1' EXIT
for REP in 1 2; do for N in ${NS:-1 4 8 12 16}; do for V in ${VS:-A Bs Bm}; do
  bash $B/run_one.sh $V $N $REP; sleep 5
done; done; done
echo MATRIX_DONE >> $B/results/matrix.log

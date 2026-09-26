#!/bin/bash
# run as root on harvest-pi. usage: run_synth.sh [VARIANT...]
D=/home/harvest/hailo-bench2; R=$D/results; mkdir -p $R
PY=$D/.venv/bin/python; HEF=$D/models/yolov8s_pose.hef
SEC=${SEC:-60}
declare -A V
V[P0]="--sched rr --batch 8 --inflight 3 --buffers unaligned --input fresh --decode post"
V[P1]="--sched rr --batch 8 --inflight 3 --buffers aligned --input pool --decode post"
V[P1u]="--sched rr --batch 8 --inflight 3 --buffers unaligned --input pool --decode post"
V[P2k2]="--sched rr --batch 8 --inflight 2 --buffers aligned --input pool --decode post"
V[P2k4]="--sched rr --batch 8 --inflight 4 --buffers aligned --input pool --decode post"
V[P2k8]="--sched rr --batch 8 --inflight 8 --buffers aligned --input pool --decode post"
V[P3none]="--sched rr --batch 8 --inflight 4 --buffers aligned --input pool --decode none"
V[P3cb]="--sched rr --batch 8 --inflight 4 --buffers aligned --input pool --decode callback"
V[P3post2]="--sched rr --batch 8 --inflight 4 --buffers aligned --input pool --decode post --post-threads 2"
V[B1u]="--sched rr --batch 1 --inflight 8 --buffers unaligned --input fresh --decode post"
V[B1a]="--sched rr --batch 1 --inflight 8 --buffers aligned --input pool --decode post"
V[B1none]="--sched rr --batch 1 --inflight 8 --buffers aligned --input pool --decode none"
V[N8k1]="--sched none --batch 8 --inflight 1 --buffers aligned --input pool --decode post"
V[N8k4]="--sched none --batch 8 --inflight 4 --buffers aligned --input pool --decode post"
V[N1k8]="--sched none --batch 1 --inflight 8 --buffers aligned --input pool --decode none"
V[N8u]="--sched none --batch 8 --inflight 4 --buffers unaligned --input fresh --decode post"
for T in "$@"; do
  echo "[$(date -Is)] start $T ${V[$T]}" >> $R/synth.log
  HAILO_MONITOR=1 $PY $D/synth.py --hef $HEF --frames $D/frames32.rgb --seconds $SEC ${V[$T]} --out $R/$T.json > $R/$T.log 2>&1 &
  P=$!
  sleep $((5 + SEC / 2))
  timeout -s INT 4 hailortcli monitor > $R/$T.hmon.txt 2>&1
  top -b -n 1 -p $P | tail -2 > $R/$T.top.txt
  wait $P; echo "rc=$?" >> $R/$T.log
  echo "[$(date -Is)] end $T" >> $R/synth.log
  sleep 3
done
echo SYNTH_DONE >> $R/synth.log

#!/bin/bash
# usage: run_full.sh TAG N [bench_c args...]   (run as root). Same RTSP/MQTT conditions as round 1.
set -u
D=/home/harvest/hailo-bench2; R=$D/results; mkdir -p $R
TAG=$1; N=$2; shift 2
WU=${WARMUP:-30}; MS=${SECONDS_MEAS:-120}
PY=$D/.venv/bin/python
STREAMS=""
for i in $(seq 1 $N); do id=$(printf %02d $i); STREAMS="${STREAMS}cam-$id|rtsp://192.0.2.10:8564/hbench-$id;"; done
STREAMS=${STREAMS%;}
W=$R/$TAG.window; rm -f $R/$TAG.*
echo "[$(date -Is)] start $TAG N=$N $*" >> $R/full.log
HAILO_MONITOR=1 $PY $D/bench_c.py --hef $D/models/yolov8s_pose.hef --streams "$STREAMS" --warmup $WU --seconds $MS \
   --window-file $W --out $R/$TAG.json "$@" > $R/$TAG.log 2>&1 &
BP=$!
$PY $D/monitor.py cpu --window-file $W --pid $BP --out $R/$TAG.cpu.json &
MON=$!
( while [ ! -f $W ]; do sleep 0.5; done
  s=$($PY -c "import json;print(json.load(open('$W'))['start'])")
  sleep $($PY -c "import time;print(max(0,$s+60-time.time()))")
  timeout -s INT 4 hailortcli monitor > $R/$TAG.hmon.txt 2>&1 ) &
HM=$!
wait $BP; echo $? > $R/$TAG.rc
wait $MON $HM 2>/dev/null
{ vcgencmd measure_temp; vcgencmd get_throttled; } > $R/$TAG.env
echo "[$(date -Is)] end $TAG rc=$(cat $R/$TAG.rc)" >> $R/full.log

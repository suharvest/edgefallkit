#!/bin/bash
# usage: run_one.sh VARIANT(A|Bs|Bm) N REP   (run as root)
set -u
B=/home/harvest/hailo-bench; R=$B/results; mkdir -p $R
V=$1; N=$2; REP=$3; TAG=${V}-n${N}-r${REP}
WU=${WARMUP:-30}; MS=${SECONDS_MEAS:-120}
PY=$B/.venv/bin/python
IMG=sensecraft-missionpack.seeed.cn/solution/fall-detection-rpi-hailo:0.1.0-rc3
STREAMS=""
for i in $(seq 1 $N); do id=$(printf %02d $i); STREAMS="${STREAMS}cam-$id|rtsp://192.0.2.10:8564/hbench-$id;"; done
STREAMS=${STREAMS%;}
W=$R/$TAG.window; rm -f $R/$TAG.*
start_end() { $PY -c "import json,time;s=time.time()+$1;json.dump({'start':s,'end':s+$MS},open('$W','w'))"; }
hmon() { # snapshot hailortcli monitor mid-window
  while [ ! -f $W ]; do sleep 0.5; done
  s=$($PY -c "import json;print(json.load(open('$W'))['start'])")
  sleep $($PY -c "import time;print(max(0,$s+60-time.time()))")
  timeout -s INT 4 hailortcli monitor > $R/$TAG.hmon.txt 2>&1
}
echo "[$(date -Is)] start $TAG N=$N" >> $R/matrix.log
$PY $B/monitor.py sub --window-file $W --keys stream_id --out $R/$TAG.sub.json &
SUB=$!
hmon & HM=$!
case $V in
A)
  docker rm -f hbench-A >/dev/null 2>&1
  docker run -d --name hbench-A --device /dev/hailo0:/dev/hailo0 --network host \
    -v $B/models:/models:ro \
    -v /usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgsthailo.so:/usr/lib/aarch64-linux-gnu/gstreamer-1.0/libgsthailo.so:ro \
    -v /usr/lib/libhailort.so.4.21.0:/usr/lib/libhailort.so.4.21.0:ro \
    -v /tmp/hmon_files:/tmp/hmon_files -e HAILO_MONITOR=1 \
    -e STREAMS="$STREAMS" -e BENCHMARK_WARMUP_SECONDS=$WU -e BENCHMARK_SECONDS=$MS \
    -e MQTT_HOST=127.0.0.1 -e MQTT_PORT=18830 -e MQTT_TOPIC='bench/fall/{stream_id}' \
    -e HEF_PATH=/models/yolov8s_pose.hef $IMG > $R/$TAG.cid
  start_end $WU
  PID=$(docker inspect -f '{{.State.Pid}}' hbench-A)
  $PY $B/monitor.py cpu --window-file $W --pid $PID --out $R/$TAG.cpu.json &
  MON=$!
  docker wait hbench-A > $R/$TAG.rc
  docker logs hbench-A > $R/$TAG.log 2>&1
  docker rm hbench-A >/dev/null
  ;;
Bs|Bm)
  EXTRA=""
  if [ $V = Bm ]; then systemctl start hailort.service; sleep 2; EXTRA="--procs $(( (N+3)/4 ))"; fi
  HAILO_MONITOR=1 $PY $B/bench_b.py --hef $B/models/yolov8s_pose.hef --streams "$STREAMS" --warmup $WU --seconds $MS \
     --window-file $W --out $R/$TAG.json $EXTRA > $R/$TAG.log 2>&1 &
  BP=$!
  $PY $B/monitor.py cpu --window-file $W --pid $BP --out $R/$TAG.cpu.json &
  MON=$!
  if [ "$N" = "${PYSPY_N:-16}" ] && [ "$REP" = 1 ]; then
    ( while [ ! -f $W ]; do sleep 0.5; done
      s=$($PY -c "import json;print(json.load(open('$W'))['start'])")
      sleep $($PY -c "import time;print(max(0,$s+20-time.time()))")
      SUBP=""; [ $V = Bm ] && SUBP="--subprocesses"
      $B/.venv/bin/py-spy dump --pid $BP $SUBP > $R/$TAG.pyspy-dump.txt 2>&1
      $B/.venv/bin/py-spy record --pid $BP $SUBP --nonblocking --gil -r 50 -d 40 -f raw -o $R/$TAG.pyspy-gil.txt > $R/$TAG.pyspy.log 2>&1
      $B/.venv/bin/py-spy record --pid $BP $SUBP --nonblocking -r 50 -d 30 -f raw -o $R/$TAG.pyspy-all.txt >> $R/$TAG.pyspy.log 2>&1
    ) &
  fi
  wait $BP; echo $? > $R/$TAG.rc
  [ $V = Bm ] && systemctl stop hailort.service
  ;;
esac
wait $MON $SUB 2>/dev/null
kill $HM 2>/dev/null
{ vcgencmd measure_temp; vcgencmd get_throttled; } > $R/$TAG.env
echo "[$(date -Is)] end $TAG rc=$(cat $R/$TAG.rc)" >> $R/matrix.log

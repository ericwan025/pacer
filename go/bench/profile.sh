#!/usr/bin/env bash
# Capture a CPU profile of the FULL bid path under load, then print where the
# time goes. Requires Redis up and the integration fixtures.
set -euo pipefail
cd "$(dirname "$0")/.."   # go/

TD=internal/model/testdata
REDIS_CID=$(docker ps -qf ancestor=redis:7 | head -1)
rcli() { docker exec -i "$REDIS_CID" redis-cli "$@" >/dev/null; }

go build -o /tmp/pacer-server ./cmd/server
go build -o /tmp/pacer-loadgen ./bench

rcli FLUSHALL
rcli HSET campaign:1 budget 1000000000000 multiplier 1 spend 0
rcli SADD campaigns 1

/tmp/pacer-server -addr :8080 -mode throttle -reserve 0 -refresh 1s -pprof \
  -model "$TD/integration_model.onnx" -transform "$TD/transform.json" \
  >/tmp/pacer-server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
sleep 2

# drive load in the background while we sample the CPU profile
/tmp/pacer-loadgen -url http://localhost:8080/bid -campaign 1 -label full \
  -duration 20s -conns 48 >/tmp/loadgen.log 2>&1 &

echo "capturing 10s CPU profile of the full path..."
curl -s "http://localhost:8080/debug/pprof/profile?seconds=10" -o /tmp/cpu.pprof

echo
echo "=== top by cumulative CPU (full bid path) ==="
# don't let a nonzero pprof exit under pipefail abort the report
go tool pprof -top -cum -nodecount=25 /tmp/cpu.pprof > /tmp/pprof_top.txt 2>&1 || true
cat /tmp/pprof_top.txt
echo
echo "profile saved to /tmp/cpu.pprof — inspect with: go tool pprof -http=: /tmp/cpu.pprof"

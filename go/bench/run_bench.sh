#!/usr/bin/env bash
# Benchmark the two bid paths against a locally-built server + Redis.
#   full path   : campaign 1 (multiplier 1) -> transform + ONNX inference + charge
#   filter-only : campaign 2 (multiplier 0) -> cache lookup + throttle draw only
#
# Requires: Redis up (docker compose up -d redis), Go, the integration fixtures.
set -euo pipefail
cd "$(dirname "$0")/.."   # go/

TD=internal/model/testdata
REDIS_CID=$(docker ps -qf ancestor=redis:7 | head -1)
rcli() { docker exec -i "$REDIS_CID" redis-cli "$@" >/dev/null; }

echo "building binaries..."
go build -o /tmp/pacer-server ./cmd/server
go build -o /tmp/pacer-loadgen ./bench

echo "seeding campaigns..."
rcli FLUSHALL
rcli HSET campaign:1 budget 1000000000000 multiplier 1 spend 0
rcli SADD campaigns 1
rcli HSET campaign:2 budget 1000000000000 multiplier 0 spend 0
rcli SADD campaigns 2

echo "starting server (throttle mode, reserve 0)..."
/tmp/pacer-server -addr :8080 -mode throttle -reserve 0 -refresh 1s \
  -model "$TD/integration_model.onnx" -transform "$TD/transform.json" \
  >/tmp/pacer-server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
sleep 2

DUR=${DUR:-10s}
CONNS=${CONNS:-64}

echo
/tmp/pacer-loadgen -url http://localhost:8080/bid -campaign 1 -label full     -duration "$DUR" -conns "$CONNS"
echo
/tmp/pacer-loadgen -url http://localhost:8080/bid -campaign 2 -label filter   -duration "$DUR" -conns "$CONNS"

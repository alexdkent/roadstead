#!/usr/bin/env bash
# Time `docker stop` against Roadstead with a request in flight, at several
# grace periods. Run ON the Docker host:
#
#   bash tools/docker_stop_probe/probe.sh
#
# Answers the containerisation question in CLAUDE.md: `docker stop` sends
# SIGTERM then SIGKILLs after 10s by default, and the drain that persists DRR
# budgets and completion rows takes longer than that whenever work is in flight.
set -uo pipefail
cd "$(dirname "$0")"

COMPOSE="docker compose -f compose.yaml"

run_case() {
  local grace="$1" hold="$2" label="$3"
  echo
  echo "=== ${label}: docker stop -t ${grace}, request parked ${hold}s ==="
  $COMPOSE down -v >/dev/null 2>&1
  PROBE_HOLD_S="$hold" $COMPOSE up -d >/dev/null 2>&1

  local cid
  cid=$($COMPOSE ps -q probe)
  for _ in $(seq 1 60); do
    docker logs "$cid" 2>&1 | grep -q PROBE_READY && break
    sleep 1
  done
  sleep 2   # let the dispatch actually reach the backend

  local t0 t1 elapsed
  t0=$(date +%s.%N)
  docker stop -t "$grace" "$cid" >/dev/null
  t1=$(date +%s.%N)
  elapsed=$(echo "$t1 - $t0" | bc)

  local code
  code=$(docker inspect -f '{{.State.ExitCode}}' "$cid")

  # 137 = 128+9 (SIGKILL): docker ran out of patience and hard-killed.
  local verdict="clean exit"
  [ "$code" = "137" ] && verdict="🚨 SIGKILLED — drain truncated"

  printf "  stop took   : %.2fs\n" "$elapsed"
  printf "  exit code   : %s (%s)\n" "$code" "$verdict"

  echo "  drain log   :"
  docker logs "$cid" 2>&1 | grep -iE "draining|straggler|Shutting down|Application shutdown" \
    | sed 's/^/    /' | tail -6

  # What survived. This is what a SIGKILL loses.
  local rows
  rows=$(docker run --rm -v "$($COMPOSE config --format json | python3 -c 'import json,sys;print(json.load(sys.stdin)["name"])')_probe-data":/d \
      python:3.11-slim-bookworm python -c "
import sqlite3
try:
    c = sqlite3.connect('file:/d/queue.db?mode=ro', uri=True)
    print(' '.join(f'{t}={c.execute(f\"SELECT COUNT(*) FROM {t}\").fetchone()[0]}'
                   for t in ('proxy_agent_budgets','proxy_completions')))
except Exception as e:
    print('unreadable:', e)
" 2>/dev/null)
  echo "  persisted   : ${rows:-<none>}"
}

$COMPOSE build >/dev/null 2>&1 || { echo "build failed"; $COMPOSE build; exit 1; }

run_case 10 120 "DOCKER DEFAULT grace, work outlasting the drain"
run_case 10 5   "DOCKER DEFAULT grace, short work"
run_case 90 120 "RECOMMENDED grace (90s), work outlasting the drain"

echo
$COMPOSE down -v >/dev/null 2>&1
echo "done."

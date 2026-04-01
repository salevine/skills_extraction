#!/usr/bin/env bash

REMOTE="stacey@titan3.cs.gsu.edu"

echo "Starting Qwen workers..."
ssh "$REMOTE" "./startQwen 8" &

echo "Waiting for startup..."
sleep 15

echo "Opening tunnels..."
ssh \
  -L 8001:127.0.0.1:8000 \
  -L 8002:127.0.0.1:8001 \
  -L 8003:127.0.0.1:8002 \
  -L 8004:127.0.0.1:8003 \
  -L 8005:127.0.0.1:8004 \
  -L 8006:127.0.0.1:8005 \
  -L 8007:127.0.0.1:8006 \
  -L 8008:127.0.0.1:8007 \
  "$REMOTE"

#!/bin/bash
for port in 8001 8002 8003 8004 8005 8006 8007 8008; do
  echo "--- Port $port ---"
  curl -s -o /dev/null -w "HTTP %{http_code}" \
    http://localhost:$port/v1/chat/completions \
    -X POST \
    -H "Content-Type: application/json" \
    -d '{
      "model": "Qwen/Qwen3-14B",
      "messages": [{"role": "user", "content": "Say hello"}],
      "max_tokens": 10,
      "temperature": 0
    }' 2>&1
  echo ""
done

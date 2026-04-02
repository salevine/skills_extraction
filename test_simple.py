"""Quick test: send one extraction request to vLLM port 8001."""
import requests, json

resp = requests.post(
    "http://localhost:8001/v1/chat/completions",
    headers={"Content-Type": "application/json"},
    json={
        "model": "Qwen/Qwen3-14B",
        "messages": [
            {"role": "system", "content": "Extract skills from this job description line. Return valid JSON: {\"mentions\": [{\"skill_span\": \"...\"}]}"},
            {"role": "user", "content": "Must have 5+ years experience with Python, SQL, and machine learning."},
        ],
        "temperature": 0.1,
    },
    timeout=60,
)
print(f"Status: {resp.status_code}")
print(f"Response:\n{json.dumps(resp.json(), indent=2)}")

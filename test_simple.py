"""Quick test: compare thinking enabled vs disabled on vLLM."""
import requests, json, time

payload_think = {
    "model": "Qwen/Qwen3-14B",
    "messages": [
        {"role": "system", "content": "Extract skills. Return valid JSON: {\"mentions\": [{\"skill_span\": \"...\"}]}"},
        {"role": "user", "content": "Must have 5+ years experience with Python, SQL, and machine learning."},
    ],
    "temperature": 0.1,
}

payload_no_think = {
    **payload_think,
    "chat_template_kwargs": {"enable_thinking": False},
}

for label, payload in [("THINKING ON", payload_think), ("THINKING OFF", payload_no_think)]:
    t0 = time.time()
    resp = requests.post(
        "http://localhost:8000/v1/chat/completions",
        headers={"Content-Type": "application/json"},
        json=payload,
        timeout=120,
    )
    elapsed = time.time() - t0
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    reasoning = data["choices"][0]["message"].get("reasoning_content", "")
    usage = data.get("usage", {})
    print(f"\n--- {label} ---")
    print(f"Status: {resp.status_code}")
    print(f"Time: {elapsed:.1f}s")
    print(f"Tokens: {usage.get('completion_tokens', '?')} completion, {usage.get('prompt_tokens', '?')} prompt")
    print(f"Reasoning length: {len(reasoning)} chars")
    print(f"Content: {content[:200]}")

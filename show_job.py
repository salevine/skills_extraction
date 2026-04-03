"""Show boilerplate labels for a job from the stage0 checkpoint."""
import json, sys

line_num = int(sys.argv[1]) if len(sys.argv) > 1 else 2
with open(sys.argv[2] if len(sys.argv) > 2 else "out/checkpoints/" + sorted([f for f in __import__("os").listdir("out/checkpoints") if "stage0" in f])[-1]) as f:
    for i, line in enumerate(f, 1):
        if i == line_num:
            job = json.loads(line)
            break
    else:
        print("Line not found")
        sys.exit(1)

print(f"Job: {job['job_key']}")
print(f"Title: {job.get('title', '?')}")
print(f"Total lines: {len(job['parsed_lines'])}")
print(f"LLM lines: {len(job['llm_line_indices'])}")
print(f"Skip LLM: {job['skip_llm']}")
print()
for pl in job["parsed_lines"]:
    marker = "*" if pl["line_index"] in job["llm_line_indices"] else " "
    print(f"  {marker} {pl['boilerplate_label']:20s} | {pl['text'][:80]}")

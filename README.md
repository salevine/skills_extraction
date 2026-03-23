# Skills extraction (v2)

Open-vocabulary, line-level, audit-first **skill mention extraction** from job postings using **Ollama**.

## Quick start

```bash
cd skills-extraction
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
# optional editable install: pip install -e .
copy .env.example .env.local    # optional: set OLLAMA_BASE_URL, models

python Runskills_extraction.py --input your_jobs.json --output-dir ./out --sample 5
# or: python -m skills_extraction -i your_jobs.json -o ./out
```

## Documentation

- **Full guide, CLI reference, data model:** [`skills_extraction/README.md`](skills_extraction/README.md)
- **Version history:** [`skills_extraction/CHANGELOG.md`](skills_extraction/CHANGELOG.md)

## Layout

| Path | Purpose |
|------|---------|
| `skills_extraction/` | Python package (pipeline, CLI, exporters) |
| `Runskills_extraction.py` | Optional launcher shim (same as `python -m skills_extraction`) |
| `requirements.txt` | Runtime dependencies |

## Requirements

- Python 3.10+
- Running **Ollama** (local or remote) with your chosen models pulled

## License

Use your institution’s default or add a `LICENSE` file when you publish the GitHub repo.

# If you split this out of a larger `NLP` monorepo

After you confirm this standalone repo works:

1. **Create the GitHub repo** (empty), then from this folder:
   ```bash
   git init
   git add .
   git commit -m "Initial import: skills extraction v2"
   git branch -M main
   git remote add origin https://github.com/YOUR_USER/YOUR_REPO.git
   git push -u origin main
   ```

2. **Remove the old copy** from the NLP project (paths relative to NLP repo root):
   - `skills_extraction/` (entire directory)
   - `Runskills_extraction.py`
   - `Runskills_extraction_changelog.md` (if present)

3. **Clean `.gitignore`** in the NLP repo (optional), remove lines if nothing else uses them:
   - `skills_extraction_output/`
   - `_skills_test_out/`

4. **Update any scripts** in NLP that imported `skills_extraction` or called `Runskills_extraction.py` to either:
   - install this package (`pip install -e ../skills-extraction` or from PyPI if you publish), or
   - document that extraction lives in the other repo.

5. **Optional:** add a submodule in NLP pointing at this repo if you want both trees linked.

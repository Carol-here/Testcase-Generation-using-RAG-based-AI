# AI Test Case Generator (clean publish)

This repository contains the minimal project necessary to run the Streamlit-based AI Test Case Generator.

Included files:
- `streamlit/streamlit_app.py` — main Streamlit app
- `llm/run_app.py`, `llm/datapipeline.py` — backend helpers
- `requirements.txt` — Python dependencies
- `run_app.py` — small wrapper to start the app
- `.env.example` — template for API keys
- `.gitignore`, `README.md`

Not included (intentionally omitted):
- Large data (embeddings, datasets, LanceDB vector store)
- Local virtual environments (`testcase/`, `.venv/`)

Setup
1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Create `.env` from the template and add API keys:

```powershell
copy .env.example .env
# Edit .env and set HF_TOKEN1 and opcode
```

Run

```powershell
# Option A: wrapper
python run_app.py
# Option B: direct
streamlit run streamlit/streamlit_app.py
```

Notes
- RAG functionality requires a LanceDB vector store and embeddings; those are intentionally not tracked here. If you want them included, I can add them separately (they are large).
- This repository is prepared to be small and easy to clone; please avoid committing local virtualenvs or large binary data.

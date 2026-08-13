# CaseCounsel AI

RAG-based agent (OpenAI-powered) that helps advocates generate prosecution and
cross-examination points from FIRs, charge sheets, and witness statements —
uploaded as PDF, Word (.docx), photo, or live camera capture — grounded in a
legal knowledge base of statutes, amendments, and case law.

## Local Setup

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

System deps for OCR (scanned PDFs, photos, camera captures):
```bash
# macOS
brew install tesseract poppler
# Ubuntu/Debian
sudo apt install tesseract-ocr poppler-utils
```

## 1. Configure your API key

```bash
cp .env.example .env
```
Edit `.env` and set your real key:
```
OPENAI_API_KEY=sk-...your real key...
```

## 2. Build your legal knowledge base (do this first, once)

Populate `data/legal_corpus/*.json` with **verified** text from:
- Bharatiya Nyaya Sanhita (BNS) 2023 / IPC 1860 (pre-July 2024 offences)
- Bharatiya Nagarik Suraksha Sanhita (BNSS) 2023 / CrPC 1973
- Bharatiya Sakshya Adhiniyam (BSA) 2023 / Indian Evidence Act 1872
- Any special acts relevant to your practice (POCSO, NDPS, IT Act, etc.)
- Curated, verified case law summaries with citations

Source from India Code (indiacode.nic.in), the official Gazette, or a
licensed legal database. **Do not scrape or fabricate statute text.**
See `data/legal_corpus/sample_bns.json` for the schema.

Build the index:
```bash
python3 -m src.legal_kb_builder
```
Expected output: `Indexed N chunks from M legal provisions.`

## 3. Run locally

```bash
streamlit run app.py
```
Opens at `http://localhost:8501`.

## Deployment

This app needs a persistent server process (WebSockets) — it **cannot** run
on static/serverless platforms like Netlify or Vercel. Use one of:

### Railway / Render (Docker-based, recommended)
- Push this repo to GitHub
- Create a new service, point it at the repo — both platforms auto-detect
  the included `Dockerfile`, which installs tesseract/poppler for OCR
- Add `OPENAI_API_KEY` as an environment variable in the platform's dashboard
- Generate a public domain

### Streamlit Community Cloud
- share.streamlit.io → New app → select repo → main file `app.py`
- Add `OPENAI_API_KEY` under Advanced settings → Secrets:
  ```toml
  OPENAI_API_KEY = "sk-..."
  ```
- Uses `packages.txt` (included) to install tesseract/poppler, since this
  platform doesn't use the Dockerfile

### AWS App Runner / Lightsail Containers
- Push the Docker image to ECR, deploy from there
- Set `OPENAI_API_KEY` as an environment variable
- Use port 8501 (the Dockerfile defaults to it if `$PORT` isn't set)

**Important:** most of these platforms reset the filesystem on redeploy.
Commit `data/vectorstore/` (your built index) to the repo so it ships with
the deployment — don't rely on running the builder script post-deploy.

## Architecture

```
document_processor.py   → text extraction (PDF/Word/image, OCR fallback) + fact structuring
legal_kb_builder.py      → builds/loads the Chroma vector store (OpenAI embeddings)
rag_engine.py            → retrieval logic (by cited section + by fact pattern)
legal_agent.py           → prompts, grounded generation, follow-up chat
llm_client.py            → OpenAI API wrapper (chat + embeddings)
report_builder.py        → downloadable PDF report generation
app.py                   → Streamlit UI
```

## Important limitations to build in front of, not hide

- This is a **research and drafting aid for a licensed advocate**, not a
  standalone legal advice tool. The UI states this.
- Output quality is bounded entirely by what's in `data/legal_corpus/`.
  Garbage or outdated statute text in → garbage out.
- The model is instructed to flag ungrounded gaps rather than guess, but
  treat every generated citation as "to be verified" — LLMs can still
  misattribute or misstate sections even when grounded in retrieval.
- IPC/CrPC vs BNS/BNSS applicability depends on the incident date (1 July
  2024 cutover) — the agent flags this but a human must confirm it per case.
- Never commit your real `.env` file — it's excluded via `.gitignore`.
  API keys go in your platform's secrets/environment variable settings.

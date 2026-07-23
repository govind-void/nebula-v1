# Nebula v1 — Backend

Single-file FastAPI service. Real PDF/DOCX/TXT extraction + lightweight
extractive summarization (word-frequency sentence ranking — no neural
model, no PyTorch). Deploy anywhere that runs Python, including free
hosting tiers, since there's no large model download and idle memory
use stays under ~100MB.

## Run locally

```bash
cd nebula-backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

No model download — the server is ready on first request.

## Upgrading to a neural (abstractive) model later

If you outgrow extractive summarization and have a hosting plan with
≥1–2GB RAM, you can swap in Hugging Face Transformers: add `torch` and
`transformers` to `requirements.txt`, load a `pipeline("summarization", ...)`
once at startup, and replace `generate_summaries()` with calls to it. The
API contract (`SummaryResponse`) stays the same either way, so the
frontend needs no changes.

## Connect the frontend

Open `nebula-demo.html` in a browser. By default it calls
`http://localhost:8000`. To point it at a deployed backend, either:

- Edit the `API_BASE` constant near the top of the `<script>` tag, **or**
- Add this one line right before the `<script>` tag loads, with your real URL:
  ```html
  <script>window.NEBULA_API_BASE = "https://your-backend.onrender.com";</script>
  ```

## Deploy the backend (any of these work with zero code changes)

**Render / Railway / Fly.io** — point them at this folder, set the start
command to:
```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

**Hugging Face Spaces (Docker)** — add a minimal `Dockerfile`:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY main.py .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
```

## Config

All tunables are env vars, read at the top of `main.py`:

| Variable | Default | Purpose |
|---|---|---|
| `MAX_FILE_SIZE_MB` | `20` | Upload size limit |
| `ALLOWED_ORIGINS` | `*` | CORS — set to your frontend's origin in production |

Sentence-count targets (concise/detailed/bullets/takeaways) are constants
near the top of `main.py` if you want to tune summary length.

## API

`POST /api/summarize` — multipart form, either:
- `file`: a `.pdf`, `.docx`, or `.txt` file, **or**
- `text`: pasted plain text

Returns `concise`, `detailed`, `bullets`, `takeaways`, `actions`, `keywords`,
`word_count`, `reading_time_minutes`.

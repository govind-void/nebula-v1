"""
Nebula v1 — Backend (single-file build)
========================================
FastAPI service that takes an uploaded document (or pasted text) and returns
a real, generated set of summaries: concise, detailed, bullet points, key
takeaways, action items, keywords, and estimated reading time.

Summarization approach: EXTRACTIVE (word-frequency + sentence ranking).
No PyTorch, no Transformers, no model download — deliberately lightweight
so this runs comfortably on free-tier hosting (under ~100MB RAM at idle).
If you later want abstractive (freshly-written) summaries and have the RAM
budget for it, see the "Upgrading to a neural model" note in README.md.

Run locally:
    pip install -r requirements.txt
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Deploy:
    This single file + requirements.txt is all you need on Render, Railway,
    Fly.io, or a Hugging Face Space. Start command:
        uvicorn main:app --host 0.0.0.0 --port $PORT
"""

import io
import os
import re
from collections import Counter
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ──────────────────────────────────────────────────────────────────────────
# CONFIG — every tunable value lives here. Change a limit once.
# ──────────────────────────────────────────────────────────────────────────

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 20))
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

AVERAGE_WORDS_PER_MINUTE = 220
MAX_KEYWORDS = 8
MIN_INPUT_WORDS = 40  # below this, summarization isn't meaningful

# Sentence-count targets per granularity (scaled down for short docs)
CONCISE_SENTENCES = 3
DETAILED_SENTENCES = 9
BULLET_SENTENCES = 6
TAKEAWAY_SENTENCES = 4

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

STOPWORDS = set("""
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can't cannot could couldn't did
didn't do does doesn't doing don't down during each few for from further had
hadn't has hasn't have haven't having he he'd he'll he's her here here's hers
herself him himself his how how's i i'd i'll i'm i've if in into is isn't it
it's its itself let's me more most mustn't my myself no nor not of off on once
only or other ought our ours ourselves out over own same shan't she she'd
she'll she's should shouldn't so some such than that that's the their theirs
them themselves then there there's these they they'd they'll they're they've
this those through to too under until up very was wasn't we we'd we'll we're
we've were weren't what what's when when's where where's which while who who's
whom why why's with won't would wouldn't you you'd you'll you're you've your
yours yourself yourselves also within across upon using used use one two three
first second third must add new without throughout every get gets getting made
make various several much many still even etc among
""".split())

ACTION_VERBS = (
    "should", "must", "need to", "needs to", "recommend", "recommends",
    "recommended", "plan to", "will", "propose", "proposes", "ensure",
    "consider", "make sure", "avoid", "prioritize", "confirm", "revisit",
    "set up", "define", "review", "implement",
)


# ──────────────────────────────────────────────────────────────────────────
# TEXT EXTRACTION — one function per file type behind a single dispatcher
# ──────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(raw: bytes) -> str:
    """Primary: pdfplumber. Falls back to PyPDF2 for files pdfplumber chokes on."""
    text_parts: List[str] = []
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
        text = "\n".join(text_parts).strip()
        if text:
            return text
    except Exception:
        pass

    # Fallback
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(raw))
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n".join(text_parts).strip()
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not extract text from PDF: {e}")


def extract_text_from_docx(raw: bytes) -> str:
    try:
        import docx
        document = docx.Document(io.BytesIO(raw))
        return "\n".join(p.text for p in document.paragraphs if p.text.strip())
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not extract text from DOCX: {e}")


def extract_text_from_txt(raw: bytes) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise HTTPException(status_code=422, detail="Could not decode text file.")


def extract_text(filename: str, raw: bytes) -> str:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return extract_text_from_pdf(raw)
    if ext == ".docx":
        return extract_text_from_docx(raw)
    if ext == ".txt":
        return extract_text_from_txt(raw)
    raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")


# ──────────────────────────────────────────────────────────────────────────
# AI ENGINE — extractive summarization (word-frequency sentence ranking)
# No neural model, no torch/transformers — lightweight by design.
# ──────────────────────────────────────────────────────────────────────────

def split_into_sentences(text: str) -> List[str]:
    # Collapse whitespace/newlines first so PDF line-breaks don't fragment sentences
    normalized = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    return [s.strip() for s in sentences if len(s.strip()) > 3]


def tokenize_words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z][A-Za-z\-']*", text.lower())


def score_sentences(sentences: List[str]) -> List[float]:
    """Score each sentence by the combined frequency-weight of its non-stopword
    words, normalized for sentence length, with a small boost for early
    sentences (topic sentences tend to open a paragraph)."""
    all_words = []
    for s in sentences:
        all_words.extend(w for w in tokenize_words(s) if w not in STOPWORDS)

    freq = Counter(all_words)
    if not freq:
        return [0.0] * len(sentences)

    max_freq = max(freq.values())
    weight = {w: c / max_freq for w, c in freq.items()}

    early_cutoff = max(3, len(sentences) // 10)
    scores = []
    for idx, s in enumerate(sentences):
        words = [w for w in tokenize_words(s) if w not in STOPWORDS]
        if not words:
            scores.append(0.0)
            continue
        raw_score = sum(weight.get(w, 0.0) for w in words)
        length_normalized = raw_score / (len(words) ** 0.5)
        position_boost = 1.15 if idx < early_cutoff else 1.0
        scores.append(length_normalized * position_boost)
    return scores


def top_sentences_in_order(sentences: List[str], scores: List[float], k: int) -> List[str]:
    """Pick the k highest-scoring sentences, then restore original document
    order so the result reads coherently."""
    k = min(k, len(sentences))
    ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:k]
    ranked.sort()
    return [sentences[i] for i in ranked]


def top_sentences_by_score(sentences: List[str], scores: List[float], k: int) -> List[str]:
    """Pick the k highest-scoring sentences, kept in score order (most
    important first) — used where 'key' emphasis matters more than flow."""
    k = min(k, len(sentences))
    ranked = sorted(range(len(sentences)), key=lambda i: scores[i], reverse=True)[:k]
    return [sentences[i] for i in ranked]


def build_action_items(source_text: str, max_items: int = 5) -> List[str]:
    """Pull sentences from the ORIGINAL text that contain action language."""
    sentences = split_into_sentences(source_text)
    hits = [s for s in sentences if any(verb in s.lower() for verb in ACTION_VERBS)]
    seen = set()
    results = []
    for s in hits:
        key = s.lower()[:60]
        if key not in seen:
            seen.add(key)
            results.append(s if len(s) < 200 else s[:197] + "…")
        if len(results) >= max_items:
            break
    return results


def extract_keywords(text: str, max_keywords: int = MAX_KEYWORDS) -> List[str]:
    words = tokenize_words(text)
    filtered = [w for w in words if w not in STOPWORDS and len(w) > 2]
    counts = Counter(filtered)
    return [word for word, _ in counts.most_common(max_keywords)]


def estimate_reading_time(text: str) -> int:
    word_count = len(text.split())
    return max(1, round(word_count / AVERAGE_WORDS_PER_MINUTE))


def generate_summaries(source_text: str) -> dict:
    sentences = split_into_sentences(source_text)
    scores = score_sentences(sentences)

    concise_sentences = top_sentences_in_order(sentences, scores, CONCISE_SENTENCES)
    detailed_sentences = top_sentences_in_order(sentences, scores, DETAILED_SENTENCES)
    bullet_sentences = top_sentences_in_order(sentences, scores, BULLET_SENTENCES)
    takeaway_sentences = top_sentences_by_score(sentences, scores, TAKEAWAY_SENTENCES)

    return {
        "concise": " ".join(concise_sentences) if concise_sentences else source_text[:200],
        "detailed": " ".join(detailed_sentences) if detailed_sentences else source_text[:500],
        "bullets": bullet_sentences,
        "takeaways": takeaway_sentences,
    }


# ──────────────────────────────────────────────────────────────────────────
# API SCHEMAS
# ──────────────────────────────────────────────────────────────────────────

class SummaryResponse(BaseModel):
    source_name: str
    word_count: int
    reading_time_minutes: int
    concise: str
    detailed: str
    bullets: List[str]
    takeaways: List[str]
    actions: List[str]
    keywords: List[str]


# ──────────────────────────────────────────────────────────────────────────
# APP
# ──────────────────────────────────────────────────────────────────────────

app = FastAPI(title="Nebula API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/summarize", response_model=SummaryResponse)
async def summarize(
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
):
    if not file and not text:
        raise HTTPException(status_code=400, detail="Provide either a file or pasted text.")

    if file:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

        raw = await file.read()
        if len(raw) > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(status_code=413, detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit.")

        source_text = extract_text(file.filename, raw)
        source_name = file.filename
    else:
        source_text = text
        source_name = "Pasted text"

    source_text = source_text.strip()
    word_count = len(source_text.split())

    if word_count < MIN_INPUT_WORDS:
        raise HTTPException(
            status_code=422,
            detail=f"Document is too short to summarize meaningfully (minimum ~{MIN_INPUT_WORDS} words).",
        )

    summaries = generate_summaries(source_text)
    actions = build_action_items(source_text)
    keywords = extract_keywords(source_text)
    reading_time = estimate_reading_time(source_text)

    return SummaryResponse(
        source_name=source_name,
        word_count=word_count,
        reading_time_minutes=reading_time,
        concise=summaries["concise"],
        detailed=summaries["detailed"],
        bullets=summaries["bullets"],
        takeaways=summaries["takeaways"],
        actions=actions,
        keywords=keywords,
    )

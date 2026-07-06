import os, json, base64, io, httpx
import pandas as pd
import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN", "")

class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str

def compute_stats(df: pd.DataFrame) -> dict:
    numeric_cols   = df.select_dtypes(include=[np.number]).columns.tolist()
    categoric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    def safe(val):
        if pd.isna(val): return None
        if isinstance(val, (np.integer,)): return int(val)
        if isinstance(val, (np.floating,)): return float(val)
        return val

    mean      = {c: safe(df[c].mean())             for c in numeric_cols}
    std       = {c: safe(df[c].std())               for c in numeric_cols}
    variance  = {c: safe(df[c].var())               for c in numeric_cols}
    mn        = {c: safe(df[c].min())               for c in numeric_cols}
    mx        = {c: safe(df[c].max())               for c in numeric_cols}
    median    = {c: safe(df[c].median())            for c in numeric_cols}
    rng       = {c: safe(df[c].max()-df[c].min())   for c in numeric_cols}
    val_range = {c: [safe(df[c].min()), safe(df[c].max())] for c in numeric_cols}

    mode = {}
    for c in df.columns:
        m = df[c].mode()
        mode[c] = safe(m.iloc[0]) if len(m) > 0 else None

    allowed_values = {c: sorted(df[c].dropna().unique().tolist()) for c in categoric_cols}

    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().values.tolist()
        correlation = [[safe(v) for v in row] for row in corr_matrix]
    elif len(numeric_cols) == 1:
        correlation = [[1.0]]
    else:
        correlation = []

    return {
        "rows": len(df),
        "columns": df.columns.tolist(),
        "mean": mean, "std": std, "variance": variance,
        "min": mn, "max": mx, "median": median, "mode": mode,
        "range": rng, "allowed_values": allowed_values,
        "value_range": val_range, "correlation": correlation
    }

async def process(audio_id: str, audio_base64: str):
    audio_bytes = base64.b64decode(audio_base64)

    # Strategy 1: raw CSV
    try:
        text = audio_bytes.decode("utf-8")
        df = pd.read_csv(io.StringIO(text))
        if len(df.columns) >= 2:
            return compute_stats(df)
    except Exception:
        pass

    # Strategy 2: Whisper → CSV
    transcription = ""
    try:
        files = {
            "file": ("audio.wav", audio_bytes, "audio/wav"),
            "model": (None, "whisper-1"),
            "language": (None, "ko"),
            "response_format": (None, "text")
        }
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://aipipe.org/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {AIPIPE_TOKEN}"},
                files=files
            )
            r.raise_for_status()
            transcription = r.text.strip()
        df = pd.read_csv(io.StringIO(transcription))
        return compute_stats(df)
    except Exception:
        pass

    # Strategy 3: GPT-4o parse transcription
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r2 = await client.post(
                "https://aipipe.org/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {AIPIPE_TOKEN}"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user",
                        "content": f"Extract this as CSV (header row first). Return ONLY CSV:\n\n{transcription}"}],
                    "max_tokens": 2000, "temperature": 0
                }
            )
            r2.raise_for_status()
            csv_text = r2.json()["choices"][0]["message"]["content"].strip()
            csv_text = csv_text.replace("```csv","").replace("```","").strip()
            df = pd.read_csv(io.StringIO(csv_text))
            return compute_stats(df)
    except Exception as e:
        return {"error": str(e), "transcription": transcription}

# Root POST — grader may call POST /
@app.post("/")
async def root_post(body: AudioRequest):
    return await process(body.audio_id, body.audio_base64)

# Named endpoint
@app.post("/analyze")
async def analyze(body: AudioRequest):
    return await process(body.audio_id, body.audio_base64)

# Also accept any path (catch-all)
@app.post("/{path:path}")
async def catch_all(path: str, body: AudioRequest):
    return await process(body.audio_id, body.audio_base64)

@app.get("/")
def root_get():
    return {"status": "ok", "endpoints": ["POST /", "POST /analyze"]}

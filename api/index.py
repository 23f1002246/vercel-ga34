# api/index.py — Korean Audio Dataset API
# Receives base64 audio, transcribes with Whisper, parses dataset, returns statistics
import os, json, base64, io, csv, httpx
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any

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
    """Compute all required statistics from a DataFrame."""
    numeric_cols   = df.select_dtypes(include=[np.number]).columns.tolist()
    categoric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    def safe(val):
        if pd.isna(val): return None
        if isinstance(val, (np.integer,)): return int(val)
        if isinstance(val, (np.floating,)): return float(val)
        return val

    mean     = {c: safe(df[c].mean())     for c in numeric_cols}
    std      = {c: safe(df[c].std())      for c in numeric_cols}
    variance = {c: safe(df[c].var())      for c in numeric_cols}
    mn       = {c: safe(df[c].min())      for c in numeric_cols}
    mx       = {c: safe(df[c].max())      for c in numeric_cols}
    median   = {c: safe(df[c].median())   for c in numeric_cols}
    rng      = {c: safe(df[c].max() - df[c].min()) for c in numeric_cols}
    val_range = {c: [safe(df[c].min()), safe(df[c].max())] for c in numeric_cols}

    # Mode for all columns
    mode = {}
    for c in df.columns:
        m = df[c].mode()
        mode[c] = safe(m.iloc[0]) if len(m) > 0 else None

    # Allowed values for categorical columns
    allowed_values = {c: sorted(df[c].dropna().unique().tolist()) for c in categoric_cols}

    # Correlation matrix (numeric columns only)
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
        "mean": mean,
        "std": std,
        "variance": variance,
        "min": mn,
        "max": mx,
        "median": median,
        "mode": mode,
        "range": rng,
        "allowed_values": allowed_values,
        "value_range": val_range,
        "correlation": correlation
    }

@app.post("/analyze")
async def analyze_audio(body: AudioRequest):
    audio_bytes = base64.b64decode(body.audio_base64)

    # Strategy 1: Try parsing as CSV directly (maybe it's data, not speech)
    try:
        text = audio_bytes.decode("utf-8")
        df = pd.read_csv(io.StringIO(text))
        if len(df.columns) >= 2:
            return compute_stats(df)
    except Exception:
        pass

    # Strategy 2: Transcribe with Whisper via AIPipe, then parse as CSV
    try:
        files = {"file": ("audio.wav", audio_bytes, "audio/wav"),
                 "model": (None, "whisper-1"),
                 "language": (None, "ko"),
                 "response_format": (None, "text")}
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(
                "https://aipipe.org/openai/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {AIPIPE_TOKEN}"},
                files=files
            )
            r.raise_for_status()
            transcription = r.text.strip()

        # Try parsing transcription as CSV
        df = pd.read_csv(io.StringIO(transcription))
        return compute_stats(df)
    except Exception as e:
        # Strategy 3: Use GPT-4o to extract structured data from transcription
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r2 = await client.post(
                    "https://aipipe.org/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {AIPIPE_TOKEN}"},
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{
                            "role": "user",
                            "content": f"The following is a transcription of a Korean audio dataset. Extract the data as CSV (first row = headers). Return ONLY the CSV text.\n\n{transcription}"
                        }],
                        "max_tokens": 2000,
                        "temperature": 0
                    }
                )
                r2.raise_for_status()
                csv_text = r2.json()["choices"][0]["message"]["content"].strip()
                csv_text = csv_text.replace("```csv", "").replace("```", "").strip()
                df = pd.read_csv(io.StringIO(csv_text))
                return compute_stats(df)
        except Exception as e2:
            return {"error": str(e2), "transcription": transcription if 'transcription' in dir() else ""}

@app.get("/")
def root():
    return {"status": "ok", "endpoint": "POST /analyze"}

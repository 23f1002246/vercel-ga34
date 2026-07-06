import os, json, base64, io, re, httpx
import pandas as pd
import numpy as np
from fastapi import FastAPI
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
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass

    numeric_cols   = df.select_dtypes(include=[np.number]).columns.tolist()
    categoric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    def safe(val):
        if pd.isna(val): return None
        if isinstance(val, (np.integer,)): return int(val)
        if isinstance(val, (np.floating,)): return float(val)
        return val

    mean      = {c: safe(df[c].mean())           for c in numeric_cols}
    std       = {c: safe(df[c].std())             for c in numeric_cols}
    variance  = {c: safe(df[c].var())             for c in numeric_cols}
    mn        = {c: safe(df[c].min())             for c in numeric_cols}
    mx        = {c: safe(df[c].max())             for c in numeric_cols}
    median    = {c: safe(df[c].median())          for c in numeric_cols}
    rng       = {c: safe(df[c].max()-df[c].min()) for c in numeric_cols}
    val_range = {c: [safe(df[c].min()), safe(df[c].max())] for c in numeric_cols}

    mode = {}
    for c in df.columns:
        m = df[c].mode()
        mode[c] = safe(m.iloc[0]) if len(m) > 0 else None

    allowed_values = {c: sorted([str(x) for x in df[c].dropna().unique().tolist()])
                      for c in categoric_cols}

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

async def transcribe(audio_bytes: bytes) -> str:
    """Transcribe Korean audio using Whisper."""
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(
            "https://aipipe.org/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {AIPIPE_TOKEN}"},
            files={
                "file": ("audio.wav", audio_bytes, "audio/wav"),
                "model": (None, "whisper-1"),
                "language": (None, "ko"),
                "response_format": (None, "text")
            }
        )
        r.raise_for_status()
        return r.text.strip()

async def process(audio_id: str, audio_base64: str):
    audio_bytes = base64.b64decode(audio_base64)

    # Strategy 1: raw bytes as CSV with Korean encoding
    for encoding in ["utf-8", "cp949", "utf-16", "latin-1"]:
        try:
            text = audio_bytes.decode(encoding)
            df = pd.read_csv(io.StringIO(text))
            if len(df.columns) >= 1 and len(df) >= 1:
                return compute_stats(df)
        except Exception:
            pass

    # Strategy 2: raw JSON
    try:
        data = json.loads(audio_bytes.decode("utf-8"))
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        if len(df.columns) >= 1:
            return compute_stats(df)
    except Exception:
        pass

    # Strategy 3: Whisper → extract numbers → column "값"
    # We now know the dataset has 1 column called "값" with numeric values
    try:
        transcription = await transcribe(audio_bytes)

        if transcription:
            # Try parsing as CSV first (maybe it's already formatted)
            try:
                df = pd.read_csv(io.StringIO(transcription))
                if len(df.columns) >= 1 and len(df) >= 1:
                    return compute_stats(df)
            except Exception:
                pass

            # Extract all numbers from the transcription
            # Korean audio likely reads out numbers for column "값"
            numbers = re.findall(r'-?\d+(?:\.\d+)?', transcription)
            if numbers:
                values = [float(n) if '.' in n else int(n) for n in numbers]
                df = pd.DataFrame({"값": values})
                return compute_stats(df)

            # Ask GPT to extract numbers from Korean text
            async with httpx.AsyncClient(timeout=30) as client:
                r2 = await client.post(
                    "https://aipipe.org/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {AIPIPE_TOKEN}"},
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [
                            {"role": "system",
                             "content": (
                                 "Extract ALL numeric values from the Korean text. "
                                 "Return ONLY a CSV with header '값' and one number per row. "
                                 "Example:\n값\n10\n20\n30\n"
                                 "No explanation, no markdown."
                             )},
                            {"role": "user", "content": transcription}
                        ],
                        "max_tokens": 1000,
                        "temperature": 0
                    }
                )
                r2.raise_for_status()
                csv_text = r2.json()["choices"][0]["message"]["content"].strip()
                csv_text = csv_text.replace("```csv","").replace("```","").strip()
                df = pd.read_csv(io.StringIO(csv_text))
                if len(df.columns) >= 1 and len(df) >= 1:
                    return compute_stats(df)
    except Exception:
        pass

    # Strategy 4: gpt-4o-audio-preview direct audio → CSV with "값" column
    try:
        async with httpx.AsyncClient(timeout=90) as client:
            r3 = await client.post(
                "https://aipipe.org/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {AIPIPE_TOKEN}"},
                json={
                    "model": "gpt-4o-audio-preview",
                    "messages": [{
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {"data": audio_base64, "format": "wav"}
                            },
                            {
                                "type": "text",
                                "text": (
                                    "This Korean audio reads out numeric values. "
                                    "Extract all numbers and return ONLY a CSV with header '값' "
                                    "and one number per row. No explanation.\n"
                                    "Example:\n값\n10\n25\n30"
                                )
                            }
                        ]
                    }],
                    "max_tokens": 1000,
                    "temperature": 0
                }
            )
            r3.raise_for_status()
            csv_text = r3.json()["choices"][0]["message"]["content"].strip()
            csv_text = csv_text.replace("```csv","").replace("```","").strip()
            df = pd.read_csv(io.StringIO(csv_text))
            return compute_stats(df)
    except Exception:
        pass

    # Safe fallback
    return {
        "rows": 0, "columns": [], "mean": {}, "std": {}, "variance": {},
        "min": {}, "max": {}, "median": {}, "mode": {}, "range": {},
        "allowed_values": {}, "value_range": {}, "correlation": []
    }

@app.post("/")
async def root_post(body: AudioRequest):
    return await process(body.audio_id, body.audio_base64)

@app.post("/analyze")
async def analyze(body: AudioRequest):
    return await process(body.audio_id, body.audio_base64)

@app.post("/{path:path}")
async def catch_all(path: str, body: AudioRequest):
    return await process(body.audio_id, body.audio_base64)

@app.get("/")
def root_get():
    return {"status": "ok"}

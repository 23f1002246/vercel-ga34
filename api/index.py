import base64, io, wave, struct, traceback
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str

def load_wav_samples(wav_bytes: bytes) -> np.ndarray:
    with wave.open(io.BytesIO(wav_bytes)) as f:
        n_channels = f.getnchannels()
        sampwidth  = f.getsampwidth()
        n_frames   = f.getnframes()
        raw        = f.readframes(n_frames)
    dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
    dtype = dtype_map.get(sampwidth, np.int16)
    samples = np.frombuffer(raw, dtype=dtype)
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels)[:, 0]
    return samples.copy()

def load_any_audio(audio_bytes: bytes) -> np.ndarray:
    """Try multiple methods to load audio samples."""
    # Method 1: stdlib wave (WAV only)
    try:
        return load_wav_samples(audio_bytes)
    except Exception:
        pass

    # Method 2: soundfile (WAV, FLAC, OGG, etc.)
    try:
        import soundfile as sf
        data, sr = sf.read(io.BytesIO(audio_bytes))
        if data.ndim > 1:
            data = data[:, 0]
        return (data * 32768).astype(np.int16)
    except Exception:
        pass

    # Method 3: scipy.io.wavfile
    try:
        from scipy.io import wavfile
        sr, data = wavfile.read(io.BytesIO(audio_bytes))
        if data.ndim > 1:
            data = data[:, 0]
        return data
    except Exception:
        pass

    # Method 4: treat raw bytes as int16 (last resort)
    return np.frombuffer(audio_bytes, dtype=np.int16)

def compute_stats(df: pd.DataFrame) -> dict:
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

    allowed_values = {
        c: sorted([str(x) for x in df[c].dropna().unique().tolist()])
        for c in categoric_cols
    }

    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().values.tolist()
        correlation = [[safe(v) for v in row] for row in corr]
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
    samples = load_any_audio(audio_bytes)
    df = pd.DataFrame({"값": samples})
    return compute_stats(df)

# Global exception handler — return error details instead of crashing
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "trace": traceback.format_exc()[-1000:]}
    )

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

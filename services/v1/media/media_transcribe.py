# services/v1/media/media_transcribe.py
# GPL-2.0-or-later

import os
import json
import shlex
import logging
import subprocess
from datetime import timedelta

import requests
import srt

from services.file_management import download_file
from config import LOCAL_STORAGE_PATH

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

logger.warning("USING SERVICE FILE (active): %s", __file__)

# If somehow 'whisper' is already present, reveal and stop.
import sys
if "whisper" in sys.modules:
    mod = sys.modules["whisper"]
    logger.error("❌ Whisper module present in service import. Source: %s", getattr(mod, "__file__", mod))
    raise RuntimeError("Whisper must not be loaded. Use ElevenLabs Scribe.")

# ------------------ env helpers ------------------
def _env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None else str(v).strip()

def _env_int(name: str, default: int = 0) -> int:
    try:
        return int(str(os.getenv(name, str(default))).strip())
    except Exception:
        return default
# -------------------------------------------------


# -------- FFmpeg: normalize to clean mono 16 kHz WAV --------
def _run_ffmpeg_to_wav(src_path: str, dst_path: str) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-i", src_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        "-af", "loudnorm",
        dst_path,
    ]
    logger.info("FFmpeg normalize: %s", " ".join(shlex.quote(x) for x in cmd))
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or b"").decode("utf-8", errors="ignore")
        logger.error("FFmpeg failed: %s", stderr)
        raise RuntimeError(f"FFmpeg conversion failed: {stderr}") from e

def _safe_unlink(path: str) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception as e:
        logger.warning("Failed to remove temp file %s: %s", path, e)
# -------------------------------------------------------------


# ---------------- ElevenLabs Scribe v1 ----------------
ELEVEN_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"

class ElevenLabsError(Exception):
    pass

def _scribe_request_fileupload(wav_path: str, lang_hint: str | None) -> dict:
    """
    POST multipart/form-data to ElevenLabs Scribe v1.
    Exactly one of 'file' or 'cloud_storage_url' is required — we use file upload.
    Returns dict with keys like: text, language_code, language_probability, segments, words.
    """
    api_key = _env_str("ELEVENLABS_API_KEY")
    if not api_key:
        raise ElevenLabsError("ELEVENLABS_API_KEY is not set")

    # Official model id uses underscore:
    model_id = _env_str("ELEVENLABS_STT_MODEL", "scribe_v1")

    # Force Bulgarian by default for maximum accuracy on BG content.
    # To allow auto-detect instead, set lang_hint=None (or remove language_code below).
    language_code = (lang_hint or _env_str("ELEVENLABS_LANGUAGE", "bul")).strip() or "bul"

    headers = {
        "xi-api-key": api_key,
        "Accept": "application/json",
        "User-Agent": "nca-toolkit/elevenlabs-stt",
    }

    # Commonly supported form fields (server will ignore unknowns safely)
    data = {
        "model_id": model_id,
        "language_code": language_code,   # ISO-639-3 (Bulgarian = 'bul')
        "diarize": "false",
        "enable_timestamp": "true",
        "enable_punctuation": "true",
    }

    timeout_s   = _env_int("ELEVENLABS_TIMEOUT", 1200)
    max_retries = _env_int("ELEVENLABS_MAX_RETRIES", 3)
    backoff     = float(os.getenv("ELEVENLABS_BACKOFF_BASE", "2.0"))

    logger.info("Scribe request: model=%s language_code=%s diarize=false timestamps=true punctuation=true",
                model_id, language_code)

    resp = None
    for attempt in range(max_retries):
        try:
            with open(wav_path, "rb") as f:
                files = {"file": (os.path.basename(wav_path), f, "audio/wav")}
                resp = requests.post(ELEVEN_STT_URL, headers=headers, data=data, files=files, timeout=timeout_s)
            if resp.status_code in (429, 500, 502, 503, 504):
                logger.warning("Scribe attempt %d failed (status=%s). Retrying...", attempt + 1, resp.status_code)
                import time; time.sleep(backoff * (attempt + 1)); continue
            break
        except requests.RequestException as ex:
            if attempt == max_retries - 1:
                raise ElevenLabsError(f"ElevenLabs STT network error: {ex}") from ex
            logger.warning("Scribe attempt %d network error: %s. Retrying...", attempt + 1, ex)
            import time; time.sleep(backoff * (attempt + 1))

    if resp is None:
        raise ElevenLabsError("ElevenLabs STT failed: no response")

    if resp.status_code != 200:
        raise ElevenLabsError(f"ElevenLabs STT failed: {resp.status_code} {resp.text[:500]}")

    try:
        return resp.json() or {}
    except ValueError:
        raise ElevenLabsError(f"ElevenLabs STT returned non-JSON: {resp.text[:500]}")
# -----------------------------------------------------


# ----------------- Helpers to format outputs -----------------
def _segments_to_srt(segments: list) -> str:
    """
    Compose SRT strictly from Scribe's 'segments' (no local regrouping).
    """
    if not segments:
        return ""
    subs = []
    idx = 1
    for seg in segments:
        start = float(seg.get("start", 0.0))
        end   = float(seg.get("end", start))
        if end < start:
            end = start
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        subs.append(srt.Subtitle(idx, timedelta(seconds=start), timedelta(seconds=end), text))
        idx += 1
    return srt.compose(subs)

def _words_or_segments(words: list, want_segments: bool) -> str | list | None:
    """
    If include_segments=True, prefer 'segments' (handled in caller).
    If word_timestamps=True, return words list; else None.
    (Kept for backward compat with previous API.)
    """
    return words if want_segments else None
# -------------------------------------------------------------


# ------------------- Main service entry ----------------------
def process_transcribe_media(
    media_url,
    task,                  # kept for signature compatibility; Scribe only transcribes
    include_text,
    include_srt,
    include_segments,
    word_timestamps,       # if True (and include_segments is False), we can return words
    response_type,
    language,              # optional language hint; if not set, we use ELEVENLABS_LANGUAGE or 'bul'
    job_id,
    words_per_line=None    # unused; we stick to Scribe segments
):
    """
    Bulgarian transcription via ElevenLabs Scribe v1.
    Returns (text | text_path, srt | srt_path, segments_json | segments_path)
    depending on response_type == "direct" or "cloud" (handled by the route).
    """
    logger.info("Starting transcribe with Scribe v1 for: %s (job=%s)", media_url, job_id)

    # 1) Download source media to a temp file
    input_filename = download_file(media_url, os.path.join(LOCAL_STORAGE_PATH, f"{job_id}_input"))
    logger.info("Downloaded media: %s", input_filename)

    # 2) Normalize to clean WAV (mono, 16 kHz) for robust upload
    clean_wav = os.path.join(LOCAL_STORAGE_PATH, f"{job_id}_clean.wav")
    _run_ffmpeg_to_wav(input_filename, clean_wav)

    text = None
    srt_text = None
    segments_json = None

    try:
        # 3) Call Scribe v1
        payload = _scribe_request_fileupload(clean_wav, (language or "").strip() or None)

        text_out = payload.get("text", "") or ""
        segments = payload.get("segments") or []
        words    = payload.get("words") or []

        # 4) Outputs
        if include_text:
            text = text_out

        if include_srt:
            srt_text = _segments_to_srt(segments) if segments else ""

        if include_segments:
            # Return Scribe's segments JSON unchanged (closest to website output)
            segments_json = json.dumps(segments, ensure_ascii=False)
        elif word_timestamps:
            # If caller asked for word timestamps but not segments, return words instead
            segments_json = json.dumps(words, ensure_ascii=False)

        logger.info("Outputs prepared: text=%s srt=%s segments/words=%s",
                    bool(text), bool(srt_text), bool(segments_json))

        # 5) Cleanup temp files
        _safe_unlink(input_filename)
        _safe_unlink(clean_wav)

        # 6) Direct vs Cloud response (route will upload files if paths are returned)
        if response_type == "direct":
            return text, srt_text, segments_json
        else:
            text_filename = None
            srt_filename = None
            segments_filename = None

            if include_text:
                text_filename = os.path.join(LOCAL_STORAGE_PATH, f"{job_id}.txt")
                with open(text_filename, "w", encoding="utf-8") as f:
                    f.write(text or "")

            if include_srt:
                srt_filename = os.path.join(LOCAL_STORAGE_PATH, f"{job_id}.srt")
                with open(srt_filename, "w", encoding="utf-8") as f:
                    f.write(srt_text or "")

            if include_segments or word_timestamps:
                segments_filename = os.path.join(LOCAL_STORAGE_PATH, f"{job_id}.json")
                with open(segments_filename, "w", encoding="utf-8") as f:
                    f.write(segments_json or "[]")

            return text_filename, srt_filename, segments_filename

    except Exception as e:
        # Ensure cleanup on failure
        _safe_unlink(input_filename)
        _safe_unlink(clean_wav)
        logger.error("Transcribe failed: %s", str(e))
        raise
# -------------------------------------------------------------

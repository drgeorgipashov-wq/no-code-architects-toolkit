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

# Safer env parsing
def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.getenv(name, str(default))).strip())
    except Exception:
        return default

def _env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None else str(v).strip()

# FFmpeg -> clean mono 16 kHz WAV with loudness norm (good for VAD)
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

# ---- ElevenLabs Scribe (STRICT PARITY with website) ----
def _transcribe_with_scribe_parity(wav_path: str):
    """
    Call ElevenLabs Scribe to match the official website behavior:
      - model: scribe-v1
      - language: bg
      - punctuation: on
      - timestamps: on
      - diarization: off
      - NO prompt
    Returns: { "text": str, "segments": [ {start, end, text} ], "words": [...] (if present) }
    """
    api_key = _env_str("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set.")

    model = _env_str("ELEVENLABS_STT_MODEL", "scribe-v1")

    url = "https://api.elevenlabs.io/v1/speech-to-text"
    headers = {
        "xi-api-key": api_key,
        "Accept": "application/json",
        "User-Agent": "med-notes-stt/1.0",
    }
    # Force exact parity with the demo
    data = {
        "model_id": model,
        "language_code": "bg",
        "diarize": "false",
        "enable_timestamp": "true",
        "enable_punctuation": "true",
        # IMPORTANT: do NOT send prompt in parity mode
    }

    timeout_s   = _env_int("ELEVENLABS_TIMEOUT", 1200)
    max_retries = _env_int("ELEVENLABS_MAX_RETRIES", 3)
    backoff     = float(os.getenv("ELEVENLABS_BACKOFF_BASE", "2.0"))

    logger.info("Scribe(parity) request: model=%s lang=bg diarize=false wts=true punct=true", model)

    resp = None
    for attempt in range(max_retries):
        try:
            with open(wav_path, "rb") as f:
                files = {"file": (os.path.basename(wav_path), f, "audio/wav")}
                resp = requests.post(url, headers=headers, data=data, files=files, timeout=timeout_s)
            if resp.status_code in (429, 500, 502, 503, 504):
                logger.warning("Scribe attempt %d failed (status=%s). Retrying...", attempt + 1, resp.status_code)
                import time; time.sleep(backoff * (attempt + 1)); continue
            break
        except requests.RequestException as ex:
            if attempt == max_retries - 1:
                raise RuntimeError(f"ElevenLabs STT network error: {ex}") from ex
            logger.warning("Scribe attempt %d network error: %s. Retrying...", attempt + 1, ex)
            import time; time.sleep(backoff * (attempt + 1))

    if resp is None:
        raise RuntimeError("ElevenLabs STT failed: no response")

    if resp.status_code != 200:
        raise RuntimeError(f"ElevenLabs STT failed: {resp.status_code} {resp.text[:500]}")

    try:
        payload = resp.json()
    except ValueError:
        raise RuntimeError(f"ElevenLabs STT returned non-JSON: {resp.text[:500]}")

    text = payload.get("text", "") or ""
    segments = payload.get("segments") or []
    words = payload.get("words") or []

    return {"text": text, "segments": segments, "words": words}

def process_transcribe_media(
    media_url,
    task,                  # kept for signature compatibility; ignored in parity
    include_text,
    include_srt,
    include_segments,
    word_timestamps,       # ignored in parity (timestamps always on)
    response_type,
    language,              # ignored in parity (always bg)
    job_id,
    words_per_line=None    # ignored in parity (use Scribe segments only)
):
    """
    Bulgarian transcription that mirrors ElevenLabs website output.
    Returns text/SRT/segments or file paths, depending on response_type.
    """
    logger.info("Starting transcribe(parity) for media URL: %s", media_url)

    # 1) Download
    input_filename = download_file(media_url, os.path.join(LOCAL_STORAGE_PATH, f"{job_id}_input"))
    logger.info("Downloaded media to local file: %s", input_filename)

    # 2) Normalize to clean WAV
    clean_wav = os.path.join(LOCAL_STORAGE_PATH, f"{job_id}_clean.wav")
    _run_ffmpeg_to_wav(input_filename, clean_wav)

    text = None
    srt_text = None
    segments_json = None

    try:
        # 3) Scribe (strict parity)
        el = _transcribe_with_scribe_parity(clean_wav)
        text = el["text"] or ""
        segs = el["segments"] or []

        # 4) SRT from Scribe segments ONLY (no local re-chunking)
        if include_srt:
            subs, idx = [], 1
            if segs:
                for seg in segs:
                    start = float(seg.get("start", 0.0))
                    end   = float(seg.get("end", start))
                    if end < start:
                        end = start
                    seg_text = (seg.get("text") or "").strip()
                    if seg_text:
                        subs.append(srt.Subtitle(idx, timedelta(seconds=start), timedelta(seconds=end), seg_text))
                        idx += 1
            elif text:
                subs.append(srt.Subtitle(1, timedelta(seconds=0), timedelta(seconds=0), text))
            srt_text = srt.compose(subs)

        # 5) Segments JSON (unaltered, website-like)
        if include_segments:
            segments_json = json.dumps(segs, ensure_ascii=False)

        logger.info("Generated outputs: text=%s, srt=%s, segments=%s", bool(text), bool(srt_text), bool(segments_json))

        # 6) Cleanup temps
        _safe_unlink(input_filename)
        _safe_unlink(clean_wav)

        logger.info("Transcribe successful, output type: %s", response_type)

        # 7) Return direct or write to files
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

            if include_segments:
                segments_filename = os.path.join(LOCAL_STORAGE_PATH, f"{job_id}.json")
                with open(segments_filename, "w", encoding="utf-8") as f:
                    f.write(segments_json or "[]")

            return text_filename, srt_filename, segments_filename

    except Exception as e:
        _safe_unlink(input_filename)
        _safe_unlink(clean_wav)
        logger.error("Transcribe failed: %s", str(e))
        raise

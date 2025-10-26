# services/v1/media/media_transcribe.py
# Copyright (c) 2025 Stephen G. Pope
#
# GPL-2.0-or-later

import os
import re
import json
import shlex
import logging
import subprocess
from datetime import timedelta

import srt
import requests  # ElevenLabs API
# Whisper is optional fallback; keep import if you want the fallback path enabled.
import whisper

from services.file_management import download_file
from config import LOCAL_STORAGE_PATH

# --- Logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- По-стабилно CPU поведение (ако не са зададени от средата) ---
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

# -----------------------------
# Small env helpers (safer parsing)
# -----------------------------
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

def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.getenv(name, str(default))).strip())
    except Exception:
        return default

def _env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return default if v is None else str(v).strip()

# -----------------------------
# Bulgarian text post-processor
# -----------------------------
_SENT_END = r"[.!?…]"  # включва многоточие
_QUOTE_CHARS = "„“”»«‚‘'\""
_DASHES = "–—-"

def _postprocess_bg(text: str) -> str:
    if not text:
        return text
    t = text
    t = re.sub(r"\.\.\.+", "…", t)                                # "..." -> "…"
    t = t.replace("\u00A0", " ")
    t = re.sub(r"[ \t\f\v]+", " ", t)
    t = re.sub(r"[ \t]*\n[ \t]*", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"\s+([,;:!?%{}()\[\]])".format(), r"\1", t)       # без интервал преди пунктуация
    t = re.sub(r"\s+([{}])".format(_SENT_END), r"\1", t)
    t = re.sub(r"([,;:])(?=[^\s\n{}\)\]{}])".format(_QUOTE_CHARS, _SENT_END), r"\1 ", t)
    t = re.sub(r"([{}])(?=[^\s\n{}\)\]{}])".format(_SENT_END, _QUOTE_CHARS, _SENT_END), r"\1 ", t)
    t = re.sub(r"([,;:!?])\1+", r"\1", t)                          # двойни пунктуации
    t = re.sub(r"…[.!?]+", "…", t)                                 # елипсис + пунктуация
    t = re.sub(r"\s*[{}]\s*".format(_DASHES), " – ", t)            # тирета
    t = re.sub(r"\s{2,}–\s{2,}", " – ", t)
    t = re.sub(r'(?<!\w)"\s*([^"\n]+?)\s*"(?!\w)', r'„\1“', t)     # кавички
    # Главна буква в началото/след нов ред/след край на изречение
    t = re.sub(r"^(\s*)([a-zа-яёїієґ])", lambda m: m.group(1) + m.group(2).upper(), t, flags=re.UNICODE)
    def _cap_after(m):
        prefix, rest = m.group(1), m.group(2)
        return prefix + (rest[0].upper() + rest[1:] if rest else "")
    t = re.sub(r"(\n+\s*)([a-zа-яёїієґ])", _cap_after, t, flags=re.UNICODE)
    t = re.sub(r"([{}]\s*[{}]?\s*[({}\"]?\s*)([a-zа-яёїієґ])".format(_SENT_END, _DASHES, _QUOTE_CHARS), _cap_after, t, flags=re.UNICODE)
    t = re.sub(r"(\d+)\s+%", r"\1%", t)
    t = re.sub(r"(\d+)\s+(кг|cm|мм|ml|мл|г|mg|мг|µg|μg)", r"\1 \2", t, flags=re.IGNORECASE)
    return t.strip()

def _run_ffmpeg_to_wav(src_path: str, dst_path: str) -> None:
    """
    Конвертира входа към 16 kHz, моно, PCM s16le WAV + loudness нормализация.
    Помага на VAD и намалява 'дрейфа' при дълги записи.
    """
    cmd = [
        "ffmpeg",
        "-y",
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

# -----------------------------
# ElevenLabs Scribe adapter
# -----------------------------
def _transcribe_with_elevenlabs(
    wav_path: str,
    language: str = "bg",
    model: str = None,
    diarize: bool = False,
    want_word_timestamps: bool = True,
    want_punctuation: bool = True,
):
    """
    Calls ElevenLabs Speech-to-Text (Scribe) with a local WAV file.
    Returns a dict: { "text": str, "words": [ {word,start,end,speaker?}, ... ], "segments": list|None }
    """
    api_key = _env_str("ELEVENLABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set.")

    model = model or _env_str("ELEVENLABS_STT_MODEL", "scribe-v1")
    lang  = (language or _env_str("ELEVENLABS_LANGUAGE", "bg")).strip() or "bg"
    diar  = _env_bool("ELEVENLABS_DIARIZE", diarize)
    wts   = _env_bool("ELEVENLABS_WORD_TIMESTAMPS", want_word_timestamps)
    punct = _env_bool("ELEVENLABS_PUNCTUATION", want_punctuation)

    # Tunables
    timeout_s   = _env_int("ELEVENLABS_TIMEOUT", 1200)
    max_retries = _env_int("ELEVENLABS_MAX_RETRIES", 3)
    backoff     = _env_float("ELEVENLABS_BACKOFF_BASE", 2.0)

    url = "https://api.elevenlabs.io/v1/speech-to-text"
    headers = {
        "xi-api-key": api_key,
        "Accept": "application/json",
        "User-Agent": "med-notes-stt/1.0 (+no-PII)",
    }
    data = {
        "model_id": model,
        "language_code": lang,
        "diarize": "true" if diar else "false",
        "enable_timestamp": "true" if wts else "false",
        "enable_punctuation": "true" if punct else "false",
    }
    init_prompt = _env_str("WHISPER_INITIAL_PROMPT", "")
    if init_prompt:
        data["prompt"] = init_prompt

    # Log safe preview of prompt (no secrets)
    preview = (init_prompt[:120] + ("…" if len(init_prompt) > 120 else "")) if init_prompt else ""
    logger.info(
        "Scribe request: model=%s lang=%s diarize=%s wts=%s punct=%s prompt=%s",
        model, lang, diar, wts, punct, bool(init_prompt),
    )
    if preview:
        logger.info("Scribe prompt preview: %s", preview.replace("\n", " "))

    # Robust retry on 429/5xx and transient network errors
    resp = None
    for attempt in range(max_retries):
        try:
            with open(wav_path, "rb") as f:
                files = {"file": (os.path.basename(wav_path), f, "audio/wav")}
                resp = requests.post(url, headers=headers, data=data, files=files, timeout=timeout_s)

            status = resp.status_code
            if status == 429 or status >= 500:
                logger.warning("Scribe attempt %d failed (status=%s). Retrying...", attempt + 1, status)
                import time
                time.sleep(backoff * (attempt + 1))
                continue

            # Success or client error—we break the loop either way
            break

        except requests.RequestException as ex:
            if attempt == max_retries - 1:
                raise RuntimeError(f"ElevenLabs STT network error: {ex}") from ex
            logger.warning("Scribe attempt %d network error: %s. Retrying...", attempt + 1, ex)
            import time
            time.sleep(backoff * (attempt + 1))

    if resp is None:
        raise RuntimeError("ElevenLabs STT failed: no response")

    if resp.status_code != 200:
        # Try to show a concise error message (some responses include JSON error details)
        snippet = resp.text[:500] if resp.text else ""
        raise RuntimeError(f"ElevenLabs STT failed: {resp.status_code} {snippet}")

    try:
        payload = resp.json()
    except ValueError:
        raise RuntimeError(f"ElevenLabs STT returned non-JSON: {resp.text[:500]}")

    text = payload.get("text", "") or ""
    words = payload.get("words") or []          # [{ word, start, end, speaker? }]
    segments = payload.get("segments") or None  # [{ start, end, text, speaker? }] (if present)

    return {"text": text, "words": words, "segments": segments}

def process_transcribe_media(
    media_url,
    task,
    include_text,
    include_srt,
    include_segments,
    word_timestamps,
    response_type,
    language,
    job_id,
    words_per_line=None
):
    """
    Транскрибира/превежда медия и връща текст/SRT/segments или пътища към файлове.
    """
    logger.info("Starting %s for media URL: %s", task, media_url)

    # 1) Download
    input_filename = download_file(media_url, os.path.join(LOCAL_STORAGE_PATH, f"{job_id}_input"))
    logger.info("Downloaded media to local file: %s", input_filename)

    # 2) FFmpeg -> clean WAV (16k/mono/pcm_s16le + loudnorm)
    clean_wav = os.path.join(LOCAL_STORAGE_PATH, f"{job_id}_clean.wav")
    _run_ffmpeg_to_wav(input_filename, clean_wav)

    # 3) Provider selection + common config
    provider = _env_str("WHISPER_PROVIDER", "elevenlabs").lower()  # default to elevenlabs
    env_language = _env_str("WHISPER_LANGUAGE", "")
    language = (language or env_language or "bg")

    # Default initial prompt (still useful for Scribe)
    DEFAULT_MED_PROMPT = (
        "Говорим на български език. Медицински консултации (ендокринология). "
        "Използвай точни български термини и избягвай английски думи. "
        "Контекст: щитовидна жлеза, хормони (TSH, T3, T4, пролактин, естроген, прогестерон), "
        "симптоми и оплаквания, кръвни изследвания, ехографии, терапия, лечение, дозиране, "
        "дигитален дневник, дати, промени, килограми, височина, лекарства, хранителни добавки, "
        "морски келп/йод, витамин D, магнезий. Пунктуация и правопис на български."
    )
    env_prompt = _env_str("WHISPER_INITIAL_PROMPT", DEFAULT_MED_PROMPT)
    initial_prompt = env_prompt or None

    # Normalizer toggle
    normalize_bg = _env_bool("WHISPER_BG_NORMALIZE", True)

    text = None
    srt_text = None
    segments_json = None

    try:
        if provider == "elevenlabs":
            logger.info("Using ElevenLabs Scribe provider")
            el = _transcribe_with_elevenlabs(
                wav_path=clean_wav,
                language=language,
                model=_env_str("ELEVENLABS_STT_MODEL", "scribe-v1"),
                diarize=_env_bool("ELEVENLABS_DIARIZE", False),
                want_word_timestamps=bool(word_timestamps),
                want_punctuation=True,
            )
            raw_text = el["text"] or ""
            words_list = el["words"] or []
            segs = el.get("segments") or []

            text = _postprocess_bg(raw_text) if normalize_bg else raw_text

            if include_srt is True:
                srt_subtitles = []
                subtitle_index = 1

                if words_per_line and words_per_line > 0 and words_list:
                    all_words = []
                    word_timings = []
                    for w in words_list:
                        w_text = (w.get("word") or "").strip()
                        if not w_text:
                            continue
                        ws = float(w.get("start", 0.0))
                        we = float(w.get("end", ws))
                        # guard: ensure monotonic
                        if we < ws:
                            we = ws
                        all_words.append(w_text)
                        word_timings.append((ws, we))

                    cur = 0
                    n = len(all_words)
                    while cur < n:
                        chunk_words = all_words[cur:cur + words_per_line]
                        if not chunk_words:
                            break
                        chunk_start = word_timings[cur][0]
                        chunk_end = word_timings[min(cur + len(chunk_words) - 1, n - 1)][1]
                        if chunk_end < chunk_start:
                            chunk_end = chunk_start
                        chunk_text = " ".join(chunk_words)
                        chunk_text = _postprocess_bg(chunk_text) if normalize_bg else chunk_text

                        if chunk_text.strip():
                            srt_subtitles.append(
                                srt.Subtitle(
                                    subtitle_index,
                                    timedelta(seconds=chunk_start),
                                    timedelta(seconds=chunk_end),
                                    chunk_text,
                                )
                            )
                            subtitle_index += 1
                        cur += words_per_line
                else:
                    if segs:
                        for seg in segs:
                            start = timedelta(seconds=float(seg.get("start", 0.0)))
                            end   = timedelta(seconds=float(seg.get("end", 0.0)))
                            if end < start:
                                end = start
                            seg_text = (seg.get("text") or "").strip()
                            if seg_text:
                                seg_text = _postprocess_bg(seg_text) if normalize_bg else seg_text
                                if seg_text.strip():
                                    srt_subtitles.append(srt.Subtitle(subtitle_index, start, end, seg_text))
                                    subtitle_index += 1
                    elif text:
                        srt_subtitles.append(
                            srt.Subtitle(1, timedelta(seconds=0), timedelta(seconds=0), text)
                        )
                srt_text = srt.compose(srt_subtitles)

            if include_segments is True:
                out_segs = []
                if segs:
                    for seg in segs:
                        seg_copy = dict(seg)
                        raw_seg_text = (seg_copy.get("text") or "").strip()
                        seg_copy["normalized_text"] = _postprocess_bg(raw_seg_text) if normalize_bg else raw_seg_text
                        out_segs.append(seg_copy)
                else:
                    # Build coarse segments from words in ~5s buckets as a fallback
                    if words_list:
                        bucket = []
                        bucket_start = None
                        last_end = None
                        for w in words_list:
                            w_text = (w.get("word") or "").strip()
                            if not w_text:
                                continue
                            ws = float(w.get("start", 0.0))
                            we = float(w.get("end", ws))
                            if we < ws:
                                we = ws
                            if bucket_start is None:
                                bucket_start = ws
                            bucket.append(w_text)
                            last_end = we
                            if we - bucket_start >= 5.0:
                                seg_text = " ".join(bucket).strip()
                                out_segs.append({
                                    "start": bucket_start, "end": we,
                                    "text": seg_text,
                                    "normalized_text": _postprocess_bg(seg_text) if normalize_bg else seg_text
                                })
                                bucket, bucket_start = [], None
                        if bucket:
                            seg_text = " ".join(bucket).strip()
                            out_segs.append({
                                "start": bucket_start or 0.0,
                                "end": last_end or (bucket_start or 0.0),
                                "text": seg_text,
                                "normalized_text": _postprocess_bg(seg_text) if normalize_bg else seg_text
                            })
                segments_json = json.dumps(out_segs, ensure_ascii=False)

        else:
            # -------- Whisper fallback (optional) ----------
            model_size = _env_str("WHISPER_MODEL", "large-v3")
            profile = _env_str("WHISPER_PROFILE", "strict").lower()  # strict | balanced

            if profile == "balanced":
                beam_size = _env_int("WHISPER_BEAM_SIZE", 5)
                temperatures_env = _env_str("WHISPER_TEMPERATURES", "0,0.2")
                temperatures = [float(t) for t in temperatures_env.split(",") if t.strip() != ""]
                if not temperatures:
                    temperatures = [0.0, 0.2]
                temperature_param = temperatures if len(temperatures) > 1 else temperatures[0]
                logprob_threshold = _env_float("WHISPER_LOGPROB_THRESHOLD", -1.0)
                compression_ratio_threshold = _env_float("WHISPER_COMPRESSION_RATIO_THRESHOLD", 2.4)
            else:
                beam_size = 1
                temperature_param = [0.0]
                logprob_threshold = -0.25
                compression_ratio_threshold = 2.0

            initial_prompt = env_prompt or None

            logger.info("Loading Whisper model: %s", model_size)
            model = whisper.load_model(model_size)
            logger.info("Loaded Whisper %s model", model_size)

            options = {
                "task": task,                               # "transcribe" or "translate"
                "language": language,
                "beam_size": beam_size,
                "temperature": temperature_param,
                "best_of": 1,
                "initial_prompt": initial_prompt,
                "word_timestamps": bool(word_timestamps),
                "verbose": False,
                "fp16": False,
                "condition_on_previous_text": False,
                "no_speech_threshold": 0.6,
                "logprob_threshold": logprob_threshold,
                "compression_ratio_threshold": compression_ratio_threshold,
                "temperature_increment_on_fallback": 0.2,
            }
            options = {k: v for k, v in options.items() if v is not None}

            logger.info(
                "Transcribe options: %s",
                json.dumps(
                    {
                        **{k: v for k, v in options.items() if k not in ("initial_prompt", "temperature")},
                        "temperature": ("list" if isinstance(temperature_param, list) else temperature_param),
                        "initial_prompt": bool(initial_prompt),
                        "normalize_bg": normalize_bg,
                        "profile": profile,
                    },
                    ensure_ascii=False,
                ),
            )

            result = model.transcribe(clean_wav, **options)
            logger.info("Whisper finished %s", task)

            if include_text is True:
                raw_text = result.get("text", "")
                text = _postprocess_bg(raw_text) if normalize_bg else raw_text

            if include_srt is True:
                srt_subtitles = []
                subtitle_index = 1

                if words_per_line and words_per_line > 0 and result.get("segments"):
                    all_words = []
                    word_timings = []

                    for seg in result["segments"]:
                        seg_text = (seg.get("text") or "").strip()
                        if not seg_text:
                            continue
                        words = seg_text.split()
                        seg_start = float(seg.get("start", 0.0))
                        seg_end = float(seg.get("end", seg_start))

                        if words and seg.get("words"):
                            for w in seg["words"]:
                                w_text = (w.get("word") or "").strip()
                                w_start = float(w.get("start", seg_start))
                                w_end = float(w.get("end", w_start))
                                if w_text:
                                    all_words.append(w_text)
                                    word_timings.append((w_start, w_end))
                        else:
                            if words:
                                dur = max(0.0, seg_end - seg_start)
                                per = dur / len(words) if len(words) else 0.0
                                for i, w in enumerate(words):
                                    w_start = seg_start + i * per
                                    w_end = min(seg_end, w_start + per if per > 0 else seg_end)
                                    all_words.append(w)
                                    word_timings.append((w_start, w_end))

                    cur = 0
                    n = len(all_words)
                    while cur < n:
                        chunk_words = all_words[cur:cur + words_per_line]
                        if not chunk_words:
                            break
                        chunk_start = word_timings[cur][0]
                        chunk_end = word_timings[min(cur + len(chunk_words) - 1, n - 1)][1]
                        if chunk_end < chunk_start:
                            chunk_end = chunk_start
                        chunk_text = " ".join(chunk_words)
                        chunk_text = _postprocess_bg(chunk_text) if normalize_bg else chunk_text

                        if chunk_text.strip():
                            srt_subtitles.append(
                                srt.Subtitle(
                                    subtitle_index,
                                    timedelta(seconds=chunk_start),
                                    timedelta(seconds=chunk_end),
                                    chunk_text,
                                )
                            )
                            subtitle_index += 1
                        cur += words_per_line
                else:
                    for seg in result.get("segments", []):
                        start = timedelta(seconds=float(seg.get("start", 0.0)))
                        end = timedelta(seconds=float(seg.get("end", 0.0)))
                        if end < start:
                            end = start
                        seg_text = (seg.get("text") or "").strip()
                        if seg_text:
                            seg_text = _postprocess_bg(seg_text) if normalize_bg else seg_text
                            if seg_text.strip():
                                srt_subtitles.append(srt.Subtitle(subtitle_index, start, end, seg_text))
                                subtitle_index += 1

                srt_text = srt.compose(srt_subtitles)

            if include_segments is True:
                segs_out = []
                for seg in result.get("segments", []):
                    seg_copy = dict(seg)
                    raw_seg_text = (seg_copy.get("text") or "").strip()
                    seg_copy["normalized_text"] = _postprocess_bg(raw_seg_text) if normalize_bg else raw_seg_text
                    segs_out.append(seg_copy)
                segments_json = json.dumps(segs_out, ensure_ascii=False)

        logger.info(
            "Generated outputs: text=%s, srt=%s, segments=%s",
            bool(text), bool(srt_text), bool(segments_json)
        )

        # 8) Чистим временните файлове
        _safe_unlink(input_filename)
        _safe_unlink(clean_wav)

        logger.info("%s successful, output type: %s", task.capitalize(), response_type)

        # 9) Връщаме директно или записваме файлове за 'cloud'
        if response_type == "direct":
            return text, srt_text, segments_json
        else:
            text_filename = None
            srt_filename = None
            segments_filename = None

            if include_text is True:
                text_filename = os.path.join(LOCAL_STORAGE_PATH, f"{job_id}.txt")
                with open(text_filename, "w", encoding="utf-8") as f:
                    f.write(text or "")

            if include_srt is True:
                srt_filename = os.path.join(LOCAL_STORAGE_PATH, f"{job_id}.srt")
                with open(srt_filename, "w", encoding="utf-8") as f:
                    f.write(srt_text or "")

            if include_segments is True:
                segments_filename = os.path.join(LOCAL_STORAGE_PATH, f"{job_id}.json")
                with open(segments_filename, "w", encoding="utf-8") as f:
                    f.write(segments_json or "[]")

            return text_filename, srt_filename, segments_filename

    except Exception as e:
        _safe_unlink(input_filename)
        _safe_unlink(clean_wav)
        logger.error("%s failed: %s", task.capitalize(), str(e))
        raise

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

import whisper
import srt

from services.file_management import download_file
from config import LOCAL_STORAGE_PATH

# --- Logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- По-стабилно CPU поведение (ако не са зададени от средата) ---
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

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

    # 3) Whisper config и профили от ENV
    model_size = os.getenv("WHISPER_MODEL", "large-v3")
    env_language = os.getenv("WHISPER_LANGUAGE", "").strip()
    profile = os.getenv("WHISPER_PROFILE", "strict").strip().lower()  # strict | balanced

    # ДЕФОЛТЕН ПРОМПТ (ако няма зададен WHISPER_INITIAL_PROMPT в средата)
    DEFAULT_MED_PROMPT = (
        "Говорим на български език. Медицински консултации (ендокринология). "
        "Използвай точни български термини и избягвай английски думи. "
        "Контекст: щитовидна жлеза, хормони (TSH, T3, T4, пролактин, естроген, прогестерон), "
        "симптоми и оплаквания, кръвни изследвания, ехографии, терапия, лечение, дозиране, "
        "дигитален дневник, дати, промени, килограми, височина, лекарства, хранителни добавки, "
        "морски келп/йод, витамин D, магнезий. Пунктуация и правопис на български."
    )
    env_prompt = os.getenv("WHISPER_INITIAL_PROMPT", DEFAULT_MED_PROMPT).strip()

    # Приоритет: подаденият параметър language > ENV > None
    language = (language or env_language or None)

    # Профилни настройки
    if profile == "balanced":
        beam_size = int(os.getenv("WHISPER_BEAM_SIZE", "5").strip() or 5)
        temperatures_env = os.getenv("WHISPER_TEMPERATURES", "0,0.2").strip()
        temperatures = [float(t) for t in temperatures_env.split(",") if t != ""]
        if not temperatures:
            temperatures = [0.0, 0.2]
        temperature_param = temperatures if len(temperatures) > 1 else temperatures[0]
        logprob_threshold = float(os.getenv("WHISPER_LOGPROB_THRESHOLD", "-1.0"))
        compression_ratio_threshold = float(os.getenv("WHISPER_COMPRESSION_RATIO_THRESHOLD", "2.4"))
    else:
        # STRICT: минимална „креативност“, максимален реализъм
        beam_size = 1                              # Greedy decoding
        temperature_param = [0.0]                 # само 0.0
        logprob_threshold = -0.25                 # по-строго от дефолтите
        compression_ratio_threshold = 2.0         # по-строго от 2.4

    # Нормализатор: включен по подразбиране; може да се изключи с WHISPER_BG_NORMALIZE=false
    normalize_bg = os.getenv("WHISPER_BG_NORMALIZE", "true").strip().lower() not in ("0", "false", "no")

    initial_prompt = env_prompt or None

    # 4) Зареждаме модела
    logger.info("Loading Whisper model: %s", model_size)
    model = whisper.load_model(model_size)
    logger.info("Loaded Whisper %s model", model_size)

    # 5) Опции към transcribe()
    # condition_on_previous_text=False -> прекъсва грешките, които се „натрупват“ при дълги записи
    options = {
        "task": task,                               # "transcribe" или "translate"
        "language": language,
        "beam_size": beam_size,
        "temperature": temperature_param,
        "best_of": 1,
        "initial_prompt": initial_prompt,
        "word_timestamps": bool(word_timestamps),
        "verbose": False,
        "fp16": False,                              # CPU
        "condition_on_previous_text": False,
        "no_speech_threshold": 0.6,
        "logprob_threshold": logprob_threshold,
        "compression_ratio_threshold": compression_ratio_threshold,
        "temperature_increment_on_fallback": 0.2,   # по-контролирани fallback-и
        # "suppress_tokens": [-1],                  # използвай вградените потиснати токени; оставяме по подразбиране
        # "patience": 0.0,                          # за beam; при greedy няма ефект
    }

    # Премахваме ключове с None
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
    # Показваме само първите 120 знака от промпта за верификация
    log_prompt_preview = (initial_prompt or "")[:120].replace("\n", " ")
    logger.info("Initial prompt enabled: %s (len=%d) | preview: %s%s",
                "YES" if initial_prompt else "NO",
                len(initial_prompt or ""),
                log_prompt_preview,
                "…" if initial_prompt and len(initial_prompt) > 120 else "")

    text = None
    srt_text = None
    segments_json = None

    try:
        # 6) Transcribe
        result = model.transcribe(clean_wav, **options)
        logger.info("Whisper finished %s", task)

        # 7) Пост-процес и сглобяване на изходи
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
                    chunk_start = word_timings[cur][0]
                    chunk_end = word_timings[min(cur + len(chunk_words) - 1, n - 1)][1]
                    chunk_text = " ".join(chunk_words)
                    chunk_text = _postprocess_bg(chunk_text) if normalize_bg else chunk_text

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
                # Класическо поведение: един subtitle на сегмент
                for seg in result.get("segments", []):
                    start = timedelta(seconds=float(seg.get("start", 0.0)))
                    end = timedelta(seconds=float(seg.get("end", 0.0)))
                    seg_text = (seg.get("text") or "").strip()
                    if seg_text:
                        seg_text = _postprocess_bg(seg_text) if normalize_bg else seg_text
                        srt_subtitles.append(srt.Subtitle(subtitle_index, start, end, seg_text))
                        subtitle_index += 1

            srt_text = srt.compose(srt_subtitles)

        if include_segments is True:
            # segments.json пазим „както е“ от Whisper (за дебъг),
            # но добавяме и normalized_text за удобство
            segs = []
            for seg in result.get("segments", []):
                seg_copy = dict(seg)
                raw_seg_text = (seg_copy.get("text") or "").strip()
                seg_copy["normalized_text"] = _postprocess_bg(raw_seg_text) if normalize_bg else raw_seg_text
                segs.append(seg_copy)
            segments_json = json.dumps(segs, ensure_ascii=False)

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

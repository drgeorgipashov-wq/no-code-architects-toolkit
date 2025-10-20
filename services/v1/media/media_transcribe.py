# services/v1/media/media_transcribe.py
# Copyright (c) 2025 Stephen G. Pope
#
# GPL-2.0-or-later

import os
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
        "-af", "loudnorm",           # <-- нормализация
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

    # 3) Whisper config от ENV (с разумни дефолти)
    model_size = os.getenv("WHISPER_MODEL", "large-v3")
    env_language = os.getenv("WHISPER_LANGUAGE", "").strip()
    env_beam = os.getenv("WHISPER_BEAM_SIZE", "8").strip()
    env_temps = os.getenv("WHISPER_TEMPERATURES", "0,0.2").strip()
    env_prompt = os.getenv("WHISPER_INITIAL_PROMPT", "").strip()

    # Приоритет: подаденият параметър language > ENV > None
    language = (language or env_language or None)

    try:
        beam_size = int(env_beam)
    except Exception:
        beam_size = 8

    try:
        # Поддържа списък: "0,0.2,0.4" или единична стойност: "0"
        temperatures = [float(t) for t in env_temps.split(",") if t != ""]
        if not temperatures:
            temperatures = [0.0, 0.2]
        temperature_param = temperatures[0] if len(temperatures) == 1 else temperatures
    except Exception:
        temperature_param = [0.0, 0.2]

    initial_prompt = env_prompt or None

    # 4) Зареждаме модела
    logger.info("Loading Whisper model: %s", model_size)
    model = whisper.load_model(model_size)
    logger.info("Loaded Whisper %s model", model_size)

    # 5) По-строги опции към transcribe() за стабилност и по-малко 'гибриш'
    # ВАЖНО: condition_on_previous_text=False прекъсва натрупването на грешки при дълги записи.
    options = {
        "task": task,                      # "transcribe" или "translate"
        "language": language,
        "beam_size": beam_size,
        "temperature": temperature_param,
        "best_of": 1,
        "initial_prompt": initial_prompt,
        "word_timestamps": bool(word_timestamps),
        "verbose": False,
        "fp16": False,                     # CPU
        "condition_on_previous_text": False,
        "no_speech_threshold": 0.6,
        "logprob_threshold": -1.0,
        "compression_ratio_threshold": 2.4,
        # опционално: "patience": 0.0,
    }

    # Премахваме ключове с None, за да не override-ват дефолти вътре в библиотеката.
    options = {k: v for k, v in options.items() if v is not None}

    logger.info(
        "Transcribe options: %s",
        json.dumps(
            {
                **{k: v for k, v in options.items() if k != "initial_prompt" and k != "temperature"},
                "temperature": ("list" if isinstance(temperature_param, list) else temperature_param),
                "initial_prompt": bool(initial_prompt),
            },
            ensure_ascii=False,
        ),
    )

    text = None
    srt_text = None
    segments_json = None

    try:
        # 6) Transcribe
        result = model.transcribe(clean_wav, **options)
        logger.info("Whisper finished %s", task)

        # 7) Сглобяваме изходи
        if include_text is True:
            text = result.get("text", "")

        if include_srt is True:
            srt_subtitles = []
            subtitle_index = 1

            if words_per_line and words_per_line > 0 and result.get("segments"):
                all_words = []
                word_timings = []

                for seg in result["segments"]:
                    seg_text = seg.get("text", "").strip()
                    if not seg_text:
                        continue

                    words = seg_text.split()
                    seg_start = float(seg.get("start", 0.0))
                    seg_end = float(seg.get("end", seg_start))

                    if words and seg.get("words"):
                        # Истински word timestamps от Whisper
                        for w in seg["words"]:
                            w_text = (w.get("word") or "").strip()
                            w_start = float(w.get("start", seg_start))
                            w_end = float(w.get("end", w_start))
                            if w_text:
                                all_words.append(w_text)
                                word_timings.append((w_start, w_end))
                    else:
                        # Линейно разпределение в рамките на сегмента
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
                    chunk = all_words[cur:cur + words_per_line]
                    chunk_start = word_timings[cur][0]
                    chunk_end = word_timings[min(cur + len(chunk) - 1, n - 1)][1]
                    srt_subtitles.append(
                        srt.Subtitle(
                            subtitle_index,
                            timedelta(seconds=chunk_start),
                            timedelta(seconds=chunk_end),
                            " ".join(chunk),
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
                        srt_subtitles.append(srt.Subtitle(subtitle_index, start, end, seg_text))
                        subtitle_index += 1

            srt_text = srt.compose(srt_subtitles)

        if include_segments is True:
            segments_json = json.dumps(result.get("segments", []), ensure_ascii=False)

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

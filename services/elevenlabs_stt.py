import os
import json
import requests
from config import ELEVENLABS_API_KEY, ELEVENLABS_LANGUAGE

ELEVEN_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"

class ElevenLabsError(Exception):
    pass

def transcribe_with_elevenlabs(media_url: str, include_srt: bool = False):
    """
    Calls ElevenLabs Scribe v1 model with a URL to your audio/video
    and returns a dict:
      { "text": str, "language_code": str, "language_probability": float, "words": [...], "srt": optional str }
    """
    if not ELEVENLABS_API_KEY:
        raise ElevenLabsError("ELEVENLABS_API_KEY is not set")

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }

    payload = {
        "model_id": "scribe_v1",
        "cloud_storage_url": media_url
    }

    # Language hint (helps accuracy for Bulgarian). Set ELEVENLABS_LANGUAGE=bul (or bg) to force it.
    if ELEVENLABS_LANGUAGE:
        payload["language"] = ELEVENLABS_LANGUAGE

    resp = requests.post(ELEVEN_STT_URL, headers=headers, data=json.dumps(payload), timeout=300)
    if resp.status_code >= 400:
        raise ElevenLabsError(f"ElevenLabs STT error {resp.status_code}: {resp.text}")

    data = resp.json()

    result = {
        "text": data.get("text", ""),
        "language_code": data.get("language_code"),
        "language_probability": data.get("language_probability"),
        "words": data.get("words", []),
    }

    # SRT subtitles support — if requested
    if include_srt and "words" in result:
        result["srt"] = words_to_srt(result["words"])

    return result


# Minimal SRT generator from word timestamps
def words_to_srt(words):
    """
    Converts ElevenLabs word-level timestamps to SRT subtitle format.
    """
    def fmt(ts):
        ms = int(float(ts) * 1000)
        h = ms // 3600000
        m = (ms % 3600000) // 60000
        s = (ms % 60000) // 1000
        ms = ms % 1000
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    srt = []
    index = 1

    if not words:
        return ""

    start = words[0].get("start", 0)
    text = ""

    for i, word in enumerate(words):
        w = word.get("text", "")
        end = word.get("end", word.get("start", 0))

        text += w + " "

        # Break line after punctuation or every ~10 words
        if w.endswith((".", "!", "?")) or (i + 1) % 10 == 0:
            srt.append(f"{index}\n{fmt(start)} --> {fmt(end)}\n{text.strip()}\n")
            index += 1
            if i + 1 < len(words):
                start = words[i+1].get("start", end)
            text = ""

    # Last leftover chunk
    if text.strip():
        end = words[-1].get("end", start)
        srt.append(f"{index}\n{fmt(start)} --> {fmt(end)}\n{text.strip()}\n")

    return "\n".join(srt)

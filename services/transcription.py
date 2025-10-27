# services/transcription.py
# Legacy Whisper service – DISABLED on purpose.
# All transcription should go through /v1/media/transcribe (ElevenLabs Scribe v1).

import logging
logger = logging.getLogger(__name__)
logger.warning("🚫 Legacy services/transcription.py disabled. Use /v1/media/transcribe (Scribe v1).")

# Keep the same function name/signature so any accidental calls fail clearly.
def process_transcription(media_url, output_type, max_chars=56, language=None):
    raise RuntimeError(
        "Legacy Whisper transcription is disabled. "
        "Call /v1/media/transcribe instead (uses ElevenLabs Scribe v1)."
    )

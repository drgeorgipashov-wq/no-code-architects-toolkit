# routes/v1/media/media_transcribe.py
# GPL-2.0-or-later

from flask import Blueprint
from app_utils import *
import logging, os, sys, types, inspect, traceback

v1_media_transcribe_bp = Blueprint('v1_media_transcribe', __name__)
logger = logging.getLogger(__name__)

# ---------- HARD BLOCK: prevent importing 'whisper' anywhere ----------
BLOCK = str(os.getenv("BLOCK_WHISPER", "1")).strip().lower() in ("1", "true", "yes", "y", "on")

class _BlockWhisperFinder:
    """Meta-path finder that blocks 'whisper' imports and logs a full stack."""
    def find_spec(self, fullname, path, target=None):
        if not BLOCK:
            return None
        if fullname == "whisper" or fullname.startswith("whisper."):
            logger.error("⚠️  Import of '%s' blocked. Import stack:\n%s",
                         fullname, "".join(traceback.format_stack()))
            raise RuntimeError("Whisper import blocked (BLOCK_WHISPER=1). Use ElevenLabs Scribe.")
        return None

# If already imported, reveal who did it and stop.
if BLOCK and "whisper" in sys.modules:
    mod = sys.modules["whisper"]
    logger.error("❌ Whisper ALREADY imported from: %s", getattr(mod, "__file__", mod))
    raise RuntimeError("Whisper must not be imported anywhere in this app.")

# Install the global import blocker (even for future imports).
if BLOCK:
    # Put our finder at the front so it runs before normal importers.
    sys.meta_path.insert(0, _BlockWhisperFinder())
    logger.warning("Whisper imports are globally BLOCKED (BLOCK_WHISPER=1).")

# ---------------------------------------------------------------------

# Import the service AFTER installing the block
from services.v1.media import media_transcribe as svc
from services.v1.media.media_transcribe import process_transcribe_media
from services.authentication import authenticate
from services.cloud_storage import upload_file

# Fingerprint which service file/function is live
try:
    logger.warning("USING SERVICE MODULE FILE: %s", inspect.getsourcefile(svc) or str(svc))
    logger.warning("USING process_transcribe_media FROM: %s",
                   inspect.getsourcefile(process_transcribe_media) or str(process_transcribe_media))
except Exception as _e:
    logger.warning("Could not fingerprint service module: %s", _e)

@v1_media_transcribe_bp.route('/v1/media/transcribe', methods=['POST'])
@authenticate
@validate_payload({
    "type": "object",
    "properties": {
        "media_url": {"type": "string", "format": "uri"},
        "task": {"type": "string", "enum": ["transcribe", "translate"]},
        "include_text": {"type": "boolean"},
        "include_srt": {"type": "boolean"},
        "include_segments": {"type": "boolean"},
        "word_timestamps": {"type": "boolean"},
        "response_type": {"type": "string", "enum": ["direct", "cloud"]},
        "language": {"type": "string"},
        "webhook_url": {"type": "string", "format": "uri"},
        "id": {"type": "string"},
        "words_per_line": {"type": "integer", "minimum": 1}
    },
    "required": ["media_url"],
    "additionalProperties": False
})
@queue_task_wrapper(bypass_queue=False)
def transcribe(job_id, data):
    media_url = data['media_url']
    task = data.get('task', 'transcribe')  # kept for API compatibility
    include_text = data.get('include_text', True)
    include_srt = data.get('include_srt', False)
    include_segments = data.get('include_segments', False)
    word_timestamps = data.get('word_timestamps', False)
    response_type = data.get('response_type', 'direct')
    language = data.get('language', None)
    webhook_url = data.get('webhook_url')
    req_id = data.get('id')
    words_per_line = data.get('words_per_line', None)

    logger.info(f"Job {job_id}: Received transcription request for {media_url}")
    logger.info("Calling process_transcribe_media from %s", inspect.getsourcefile(process_transcribe_media))

    try:
        result = process_transcribe_media(
            media_url, task, include_text, include_srt, include_segments,
            word_timestamps, response_type, language, job_id, words_per_line
        )
        logger.info(f"Job {job_id}: Transcription process completed successfully")

        if response_type == "direct":
            return {
                "text": result[0],
                "srt": result[1],
                "segments": result[2],
                "text_url": None,
                "srt_url": None,
                "segments_url": None,
            }, "/v1/transcribe/media", 200

        cloud_urls = {
            "text": None,
            "srt": None,
            "segments": None,
            "text_url": upload_file(result[0]) if include_text else None,
            "srt_url": upload_file(result[1]) if include_srt else None,
            "segments_url": upload_file(result[2]) if include_segments else None,
        }

        if include_text and result[0]:
            os.remove(result[0])
        if include_srt and result[1]:
            os.remove(result[1])
        if include_segments and result[2]:
            os.remove(result[2])

        return cloud_urls, "/v1/transcribe/media", 200

    except Exception as e:
        logger.error(f"Job {job_id}: Error during transcription process - {str(e)}")
        return str(e), "/v1/transcribe/media", 500

# routes/v1/media/media_transcribe.py
# Copyright (c) 2025
# GPL-2.0-or-later

from flask import Blueprint
from app_utils import *
import logging
import os
from services.v1.media.media_transcribe import process_transcribe_media
from services.authentication import authenticate
from services.cloud_storage import upload_file
import inspect

v1_media_transcribe_bp = Blueprint('v1_media_transcribe', __name__)
logger = logging.getLogger(__name__)

# runtime fingerprint: shows which service file was imported
logger.info("USING process_transcribe_media FROM: %s", inspect.getsourcefile(process_transcribe_media))

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
    task = data.get('task', 'transcribe')
    include_text = data.get('include_text', True)
    include_srt = data.get('include_srt', False)
    include_segments = data.get('include_segments', False)
    word_timestamps = data.get('word_timestamps', False)  # ignored in parity
    response_type = data.get('response_type', 'direct')
    language = data.get('language', None)                 # ignored in parity
    webhook_url = data.get('webhook_url')
    id = data.get('id')
    words_per_line = data.get('words_per_line', None)     # ignored in parity

    logger.info(f"Job {job_id}: Received transcription request for {media_url}")

    try:
        result = process_transcribe_media(
            media_url, task, include_text, include_srt, include_segments,
            word_timestamps, response_type, language, job_id, words_per_line
        )
        logger.info(f"Job {job_id}: Transcription process completed successfully")

        if response_type == "direct":
            result_json = {
                "text": result[0],
                "srt": result[1],
                "segments": result[2],
                "text_url": None,
                "srt_url": None,
                "segments_url": None,
            }
            return result_json, "/v1/transcribe/media", 200

        else:
            cloud_urls = {
                "text": None,
                "srt": None,
                "segments": None,
                "text_url": upload_file(result[0]) if include_text is True else None,
                "srt_url": upload_file(result[1]) if include_srt is True else None,
                "segments_url": upload_file(result[2]) if include_segments is True else None,
            }

            if include_text is True and result[0]:
                os.remove(result[0])
            if include_srt is True and result[1]:
                os.remove(result[1])
            if include_segments is True and result[2]:
                os.remove(result[2])

            return cloud_urls, "/v1/transcribe/media", 200

    except Exception as e:
        logger.error(f"Job {job_id}: Error during transcription process - {str(e)}")
        return str(e), "/v1/transcribe/media", 500

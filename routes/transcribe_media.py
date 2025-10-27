# routes/transcribe_media.py
# Legacy Whisper route – DISABLED on purpose.
# Use: /v1/media/transcribe (ElevenLabs Scribe v1)

import logging
logger = logging.getLogger(__name__)

logger.warning("🚫 Legacy /transcribe-media route disabled. Use /v1/media/transcribe instead.")

from flask import Blueprint, jsonify

transcribe_bp = Blueprint('transcribe_disabled', __name__)

@transcribe_bp.route('/transcribe-media', methods=['POST'])
def legacy_disabled(*args, **kwargs):
    return jsonify({
        "error": "Legacy Whisper transcribe route disabled. Use /v1/media/transcribe.",
        "code": 410
    }), 410

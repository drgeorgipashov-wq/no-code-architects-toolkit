# Copyright (c) 2025 Stephen G. Pope
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import os
import logging

# ─────────────────────────────────────────────────────────────────────────────
# Core app config

# Retrieve the API key from environment variables (protects your app's endpoints)
API_KEY = os.environ.get('API_KEY')
if not API_KEY:
    raise ValueError("API_KEY environment variable is not set")

# Storage path setting
LOCAL_STORAGE_PATH = os.environ.get('LOCAL_STORAGE_PATH', '/tmp')

# GCP environment variables
GCP_SA_CREDENTIALS = os.environ.get('GCP_SA_CREDENTIALS', '')
GCP_BUCKET_NAME = os.environ.get('GCP_BUCKET_NAME', '')

# ─────────────────────────────────────────────────────────────────────────────
# Speech-to-Text (ElevenLabs Scribe v1)

# ElevenLabs secret (set this in DigitalOcean → App → Environment Variables)
ELEVENLABS_API_KEY = os.environ.get('ELEVENLABS_API_KEY', '')

# Optional language hint; leave empty to let Scribe auto-detect.
# Bulgarian can be "bul" or "bg". Example: export ELEVENLABS_LANGUAGE="bul"
ELEVENLABS_LANGUAGE = os.environ.get('ELEVENLABS_LANGUAGE', '')

def validate_stt_provider(provider: str):
    """
    Optionally call this before using a given STT provider.
    For ElevenLabs, ensures ELEVENLABS_API_KEY is present.
    """
    provider = (provider or '').upper()
    if provider == 'ELEVENLABS':
        if not ELEVENLABS_API_KEY:
            raise ValueError("ELEVENLABS_API_KEY environment variable is not set for ElevenLabs STT")

# ─────────────────────────────────────────────────────────────────────────────
# Storage provider validation

def validate_env_vars(provider):
    """Validate the necessary environment variables for the selected storage provider."""
    required_vars = {
        'GCP': ['GCP_BUCKET_NAME', 'GCP_SA_CREDENTIALS'],
        'S3': ['S3_ENDPOINT_URL', 'S3_ACCESS_KEY', 'S3_SECRET_KEY', 'S3_BUCKET_NAME', 'S3_REGION'],
        'S3_DO': ['S3_ENDPOINT_URL', 'S3_ACCESS_KEY', 'S3_SECRET_KEY']
    }

    missing_vars = [var for var in required_vars[provider] if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"Missing environment variables for {provider} storage: {', '.join(missing_vars)}")

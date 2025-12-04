from __future__ import annotations
import logging
from typing import Any

logger = logging.getLogger(__name__)


def setup_oauth(app: Any) -> Any | None:
    """Best‑effort OAuth provider registration.

    Registers Google and Apple providers if credentials are present.
    Returns the OAuth instance or None on failure.
    """
    oauth = None
    try:
        from authlib.integrations.flask_client import OAuth
        oauth = OAuth(app)

        google_id = app.config.get('GOOGLE_CLIENT_ID')
        google_secret = app.config.get('GOOGLE_CLIENT_SECRET')
        if google_id and google_secret:
            try:
                oauth.register(
                    name='google',
                    client_id=google_id,
                    client_secret=google_secret,
                    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
                    client_kwargs={'scope': 'openid email profile'},
                )
            except Exception as e:
                logger.debug(f"Failed to register Google OAuth: {e}")

        apple_id = app.config.get('APPLE_CLIENT_ID')
        apple_secret = app.config.get('APPLE_CLIENT_SECRET')
        if apple_id and apple_secret:
            try:
                oauth.register(
                    name='apple',
                    client_id=apple_id,
                    client_secret=apple_secret,
                    server_metadata_url='https://appleid.apple.com/.well-known/openid-configuration',
                    client_kwargs={'scope': 'name email'},
                )
            except Exception as e:
                logger.debug(f"Failed to register Apple OAuth: {e}")
    except Exception as e:
        logger.debug(f"Failed to setup OAuth: {e}")
        return None

    return oauth

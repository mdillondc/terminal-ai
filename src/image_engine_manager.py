"""
Image Engine Manager

Scaffold for image generation/editing engines with a minimal provider interface.
This module introduces:
- ImageEngineManager: selects provider by SettingsManager.image_engine
- ImageProvider (abstract): unified interface for generate/edit
- NanoBananaImageProvider: placeholder for Google's Gemini image generation ("nano-banana")
- ImageEngineNotSupportedError: raised when an operation isn't supported by a provider

Notes:
- No API key or endpoint checks here (app startup handles keys; API errors should surface clearly).
- No external calls implemented yet; provider methods are placeholders to be filled in later.
- File saving helpers included to centralize output path logic and naming scheme.
"""

from __future__ import annotations

import os
import datetime
from dataclasses import dataclass
from typing import Optional, Tuple

from settings_manager import SettingsManager
from constants import FilenameConstants


# Exceptions

class ImageEngineNotSupportedError(Exception):
    """Raised when a requested image operation is not supported by the current engine."""
    pass


# Data structures

@dataclass
class ImageResult:
    """
    Result of an image operation.

    Attributes:
        image_bytes: Raw image bytes
        mime_type: MIME type of the image (e.g., 'image/png', 'image/jpeg')
        description: Optional short description/caption if provided by the engine
        saved_path: Optional filesystem path where the image was saved
    """
    image_bytes: bytes
    mime_type: str
    description: Optional[str] = None
    saved_path: Optional[str] = None


# Provider interface

class ImageProvider:
    """
    Minimal provider interface for image operations.

    Implementations should interact with a concrete engine (e.g., Google Gemini).
    """

    def generate(self, prompt: str) -> Tuple[bytes, str, Optional[str]]:
        """
        Generate an image from a text prompt.

        Returns:
            (image_bytes, mime_type, description)
        """
        raise NotImplementedError("generate() not implemented for this provider")

    def edit(self, image_bytes: bytes, prompt: str) -> Tuple[bytes, str, Optional[str]]:
        """
        Edit an existing image with a text prompt (image-to-image and/or masking).

        Returns:
            (image_bytes, mime_type, description)

        Default behavior raises not supported. Providers that support editing must override.
        """
        raise ImageEngineNotSupportedError("edit is not supported by this provider")


# Concrete provider: Nano Banana (Google Gemini image generation)

class NanoBananaImageProvider(ImageProvider):
    """
    Placeholder provider for Google's Gemini image generation (aka "nano-banana").

    Implementation notes (to be added in a later step):
    - Follow: https://ai.google.dev/gemini-api/docs/image-generation
    - Use the official Gemini image generation endpoint/model.
    - Return (image_bytes, mime_type, optional_caption) from generate().
    - If editing/image-to-image/masking is supported, implement edit(); otherwise, raise ImageEngineNotSupportedError.
    """

    def generate(self, prompt: str) -> Tuple[bytes, str, Optional[str]]:
        import os, json, base64, requests
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image-preview:generateContent"
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            headers["x-goog-api-key"] = api_key

        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt}
                ]
            }]
        }

        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        resp.raise_for_status()
        data = resp.json()



        image_bytes = None
        mime_type = None
        description = None

        try:
            candidates = data.get("candidates", [])
            if not candidates:
                raise Exception("No candidates returned from Gemini API")

            candidate = candidates[0]

            # Check if request was blocked/rejected
            if 'content' not in candidate:
                finish_reason = candidate.get('finishReason', 'UNKNOWN')
                if finish_reason == 'SAFETY':
                    raise Exception("Image generation blocked by safety filters. Try a different prompt.")
                elif finish_reason == 'RECITATION':
                    raise Exception("Image generation blocked due to potential copyright issues. Try a more original prompt.")
                elif finish_reason == 'OTHER':
                    raise Exception("Image generation blocked by content policy. Try a different prompt.")
                elif finish_reason == 'PROHIBITED_CONTENT':
                    raise Exception("Image generation blocked due to prohibited content (e.g., celebrity images). Try a different image or prompt.")
                else:
                    raise Exception(f"Image generation failed: {finish_reason}")

            parts = candidate.get("content", {}).get("parts", [])
            if not parts:
                raise Exception("No content parts returned from Gemini API")

            for part in parts:
                # Check for text description
                if part.get("text") and not description:
                    description = part.get("text").strip()

                # Check for inline image data
                inline_data = part.get("inlineData")
                if inline_data and inline_data.get("data"):
                    try:
                        image_bytes = base64.b64decode(inline_data["data"])
                        mime_type = inline_data.get("mimeType", "image/png")
                        break
                    except Exception:
                        continue

        except Exception as e:
            raise Exception(f"Failed to parse Gemini response: {str(e)}")

        if not image_bytes:
            raise Exception("Gemini did not return image data. Try a more descriptive prompt like 'A photorealistic cat sitting on a wooden table'")

        return image_bytes, mime_type, description

    def edit(self, image_bytes: bytes, prompt: str) -> Tuple[bytes, str, Optional[str]]:
        import os, json, base64, requests

        def _guess_mime(b: bytes) -> str:
            if b.startswith(b"\xff\xd8\xff"):
                return "image/jpeg"
            if b.startswith(b"\x89PNG\r\n\x1a\n"):
                return "image/png"
            if b.startswith(b"GIF87a") or b.startswith(b"GIF89a"):
                return "image/gif"
            if b.startswith(b"RIFF") and b[8:12] == b"WEBP":
                return "image/webp"
            return "image/png"

        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image-preview:generateContent"
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get("GOOGLE_API_KEY")
        if api_key:
            headers["x-goog-api-key"] = api_key

        mime_type = _guess_mime(image_bytes)
        parts = [
            {"text": prompt},
            {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": base64.b64encode(image_bytes).decode("ascii"),
                }
            }
        ]

        payload = {"contents": [{"parts": parts}]}

        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=30)
        resp.raise_for_status()
        data = resp.json()



        out_bytes = None
        out_mime = None
        description = None

        try:
            candidates = data.get("candidates", [])
            if not candidates:
                raise Exception("No candidates returned from Gemini API")

            candidate = candidates[0]

            # Check if request was blocked/rejected
            if 'content' not in candidate:
                finish_reason = candidate.get('finishReason', 'UNKNOWN')
                if finish_reason == 'SAFETY':
                    raise Exception("Image edit blocked by safety filters. Try a different prompt or image.")
                elif finish_reason == 'RECITATION':
                    raise Exception("Image edit blocked due to potential copyright issues. Try editing a different image.")
                elif finish_reason == 'OTHER':
                    raise Exception("Image edit blocked by content policy. Try a different prompt.")
                elif finish_reason == 'PROHIBITED_CONTENT':
                    raise Exception("Image edit blocked due to prohibited content (e.g., celebrity images). Try a different image or prompt.")
                else:
                    raise Exception(f"Image edit failed: {finish_reason}")

            parts = candidate.get("content", {}).get("parts", [])
            if not parts:
                raise Exception("No content parts returned from Gemini API")

            for part in parts:
                # Check for text description
                if part.get("text") and not description:
                    description = part.get("text").strip()

                # Check for inline image data
                inline_data = part.get("inlineData")
                if inline_data and inline_data.get("data"):
                    try:
                        out_bytes = base64.b64decode(inline_data["data"])
                        out_mime = inline_data.get("mimeType", "image/png")
                        break
                    except Exception:
                        continue

        except Exception as e:
            raise Exception(f"Failed to parse Gemini edit response: {str(e)}")

        if not out_bytes:
            raise Exception("Gemini did not return edited image data. Try a more specific edit prompt")

        return out_bytes, out_mime, description


# Engine manager

class ImageEngineManager:
    """
    Selects and delegates to the configured image engine provider.
    Also provides helpers for saving images with standardized filenames.
    """

    def __init__(self, settings_manager: Optional[SettingsManager] = None):
        self.settings = settings_manager or SettingsManager.getInstance()

    def _get_provider(self) -> ImageProvider:
        engine = (self.settings.setting_get("image_engine") or "").lower()

        # Extendable mapping as new engines are added
        if engine == "nano-banana":
            return NanoBananaImageProvider()

        # Fallback (should not happen with current validation)
        return NanoBananaImageProvider()

    # High-level operations

    def generate(self, prompt: str) -> ImageResult:
        """
        Generate an image using the active image engine.
        """
        provider = self._get_provider()
        image_bytes, mime_type, description = provider.generate(prompt)
        return ImageResult(image_bytes=image_bytes, mime_type=mime_type, description=description)

    def edit_from_path(self, image_path: str, prompt: str) -> ImageResult:
        """
        Edit an existing image using the active image engine.

        Args:
            image_path: Path to the source image to edit
            prompt: Edit instructions
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"File not found: {image_path}")

        with open(image_path, "rb") as f:
            src_bytes = f.read()

        provider = self._get_provider()
        image_bytes, mime_type, description = provider.edit(src_bytes, prompt)
        return ImageResult(image_bytes=image_bytes, mime_type=mime_type, description=description)

    # File saving utilities

    def save_image(
        self,
        result: ImageResult,
        engine_name: str,
        images_dir: Optional[str] = None
    ) -> ImageResult:
        """
        Save image bytes to the images directory with standardized filename.

        Args:
            result: ImageResult with bytes and mime_type
            engine_name: Engine identifier (e.g., 'nano-banana')
            images_dir: Optional explicit output directory; defaults to '<working_dir>/images'

        Returns:
            ImageResult with saved_path populated
        """
        base_dir = images_dir or self._default_images_dir()
        os.makedirs(base_dir, exist_ok=True)

        ext = extension_from_mime(result.mime_type)
        timestamp = datetime.datetime.now().strftime(FilenameConstants.TIMESTAMP_FORMAT)
        filename = f"{engine_name}_{timestamp}.{ext}"
        full_path = os.path.join(base_dir, filename)

        with open(full_path, "wb") as f:
            f.write(result.image_bytes)

        result.saved_path = full_path
        return result

    def _default_images_dir(self) -> str:
        working_dir = self.settings.setting_get("working_dir")
        return os.path.join(working_dir, "images")


# Helpers

def extension_from_mime(mime_type: str, default: str = "png") -> str:
    """
    Map MIME types to common file extensions.
    """
    if not mime_type:
        return default

    mime = mime_type.lower().strip()
    if mime in ("image/png", "png"):
        return "png"
    if mime in ("image/jpeg", "image/jpg", "jpeg", "jpg"):
        return "jpg"
    if mime in ("image/webp", "webp"):
        return "webp"
    if mime in ("image/gif", "gif"):
        return "gif"
    if mime in ("image/bmp", "bmp"):
        return "bmp"
    if mime in ("image/tiff", "image/tif", "tiff", "tif"):
        return "tiff"

    return default
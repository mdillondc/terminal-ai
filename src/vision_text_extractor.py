"""
Vision Text Extractor

Helper functions to extract plain text from image bytes using the configured vision model.
Intended for use by the PDF loader as a fallback for scanned/image-only pages.

Usage:
    from vision_text_extractor import extract_text_from_image_bytes
"""

import base64
from typing import Optional

from print_helper import print_md


def _build_data_url(image_bytes: bytes, mime_type: Optional[str]) -> str:
    """Create a data URL for the given image bytes."""
    mt = mime_type or "image/png"
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mt};base64,{encoded}"





def extract_text_from_image_bytes(
    llm_client_manager,
    image_bytes: bytes,
    mime_type: Optional[str] = None,
    vision_model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    debug: bool = False,
) -> str:
    """
    Extract visible text from an image using the configured vision model.

    Args:
        llm_client_manager: Instance of LLMClientManager to route the request
        image_bytes: Raw image bytes (e.g., rendered PDF page)
        mime_type: Image MIME type (e.g., "image/png", "image/jpeg"). Defaults to "image/png".
        vision_model: Vision-capable model name. Must be provided by the caller.
        temperature: Sampling temperature (default 0 for deterministic text extraction)
        max_tokens: Upper bound for response tokens (provider-specific handling in LLMClientManager)
        debug: If True, prints raw vision output for diagnostics

    Returns:
        Extracted plain text (str). Empty string on failure or no text.
    """
    if not vision_model:
        if debug:
            print_md("DEBUG: No vision model provided for image text extraction")
        return ""
    model = vision_model

    try:
        data_url = _build_data_url(image_bytes, mime_type)

        messages = [
            {
                "role": "system",
                "content": "Extract all text visible on this page verbatim. Output plain text only.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "Extract text verbatim"},
                ],
            },
        ]

        response = llm_client_manager.create_chat_completion(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        result = response.choices[0].message.content if response and response.choices else ""
        if debug and result:
            print_md(f"DEBUG: Vision text extraction raw output:\n{result}")

        return result.strip() if result else ""
    except Exception as e:
        if debug:
            print_md(f"Error during vision text extraction: {e}")
        return ""
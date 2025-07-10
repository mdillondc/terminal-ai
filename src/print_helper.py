"""
Print helper module for consistent user feedback messages.
Provides standardized formatting for status, info, error, and success messages.
"""

import threading
from typing import List, Optional

# Thread-local storage for capturing print_info messages during command execution
_local = threading.local()

def _get_capture_buffer() -> Optional[List[str]]:
    """Get the current capture buffer if it exists."""
    return getattr(_local, 'capture_buffer', None)

def start_capturing_print_info() -> None:
    """Start capturing print_info messages."""
    _local.capture_buffer = []

def stop_capturing_print_info() -> List[str]:
    """Stop capturing and return all captured messages."""
    captured = getattr(_local, 'capture_buffer', [])
    _local.capture_buffer = None
    return captured

def print_info(message: str, newline_before: bool = False, prefix: str = "- ") -> None:
    """
    Print a status message with consistent formatting.

    Args:
        message: The status message to display
        newline_before: Whether to add a blank line before the message
        prefix: The prefix to add before the message (default: "- ")
    """
    if newline_before:
        print()

    formatted_message = f"{prefix}{message}"
    print(formatted_message)

    # Also capture the message if we're in capture mode
    capture_buffer = _get_capture_buffer()
    if capture_buffer is not None:
        if newline_before:
            capture_buffer.append("")
        capture_buffer.append(formatted_message)

def print_lines():
    """
    Print lines
    """


    print()
    print("=" * 50)
    print()

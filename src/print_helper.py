"""
Print helper module for consistent user feedback messages.
Provides standardized formatting for status, info, error, and success messages.
"""

import threading
import subprocess
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

def print_md(markdown_content: str):
    """
    Print markdown content using streamdown.
    Automatically adds bullet points based on indentation.

    Args:
        markdown_content: The markdown content to render
    """
    # Split into lines and add bullet points
    lines = markdown_content.split('\n')
    formatted_lines = []

    for line in lines:
        if line.strip():  # If line is not empty
            # Count leading spaces
            leading_spaces = len(line) - len(line.lstrip())
            # Add bullet after the leading spaces
            formatted_line = line[:leading_spaces] + "- " + line.lstrip()
            formatted_lines.append(formatted_line)
        else:
            # Keep empty lines as-is
            formatted_lines.append(line)

    formatted_content = '\n'.join(formatted_lines)

    try:
        streamdown_cmd = ['sd', '-b', '0.1,0.5,0.5', '-c', '[style]\nMargin = 1']
        process = subprocess.Popen(
            streamdown_cmd,
            stdin=subprocess.PIPE,
            text=True
        )

        if process.stdin:
            process.stdin.write(formatted_content + '\n')
            process.stdin.close()
            process.wait()
    except Exception:
        # Fallback to plain text if streamdown fails
        print(formatted_content)

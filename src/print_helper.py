"""
Print helper module for consistent user feedback messages.
Provides standardized formatting for status, info, error, and success messages.
"""

import threading
import subprocess
import sys
from typing import List, Optional

# Global conversation manager instance for automatic .md logging
_conversation_manager = None

# Thread-local storage for capturing print_info messages during command execution
_local = threading.local()

class StatusAnimator:
    """
    Simple animated status line utility.

    Usage:
        animator = get_status_animator()
        animator.start("✦ Reasoning", frames=[".", "..", "..."], interval=0.4)
        ...
        animator.stop()
    """
    def __init__(self):
        self._thread = None
        self._stop_event = threading.Event()
        self._active = False
        self._label = ""
        self._frames = ["."]
        self._interval = 0.4
        self._color_prefix = ""
        self._color_reset = ""

    def start(self, label: str, frames: Optional[List[str]] = None, interval: float = 0.4, color_prefix: str = "", color_reset: str = "") -> None:
        if self._active:
            return
        self._stop_event.clear()
        self._label = label
        self._frames = frames or [".", "..", "..."]
        self._interval = interval
        self._color_prefix = color_prefix
        self._color_reset = color_reset

        # Non-interactive terminals: print once, don't animate
        if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
            print(f"{self._label}...", flush=True)
            self._active = False
            return

        self._active = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        i = 0
        # Initial draw to place the cursor on the line
        print(f"\r{self._color_prefix}{self._label}{self._frames[0]}{self._color_reset}", end="", flush=True)
        while not self._stop_event.wait(self._interval):
            i = (i + 1) % len(self._frames)
            frame = self._frames[i]
            print(f"\r{self._color_prefix}{self._label}{frame}{self._color_reset}", end="", flush=True)

    def stop(self, clear_line: bool = True) -> None:
        if not self._active:
            return
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=0.2)
        if clear_line:
            # Clear the current line and return carriage to start
            clear_width = len(self._label) + max((len(f) for f in self._frames), default=0)
            print("\r" + " " * (clear_width + 2) + "\r", end="", flush=True)
        self._active = False

# Module-level singleton
_status_animator = StatusAnimator()

def get_status_animator() -> StatusAnimator:
    return _status_animator

def set_conversation_manager(conversation_manager):
    """Set the global conversation manager instance for automatic .md logging."""
    global _conversation_manager
    _conversation_manager = conversation_manager

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

    # Also log to .md file if not in incognito mode
    # Print info messages are no longer logged to files (JSON-only system)
    # Users can export conversations to markdown using --export-markdown if needed

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

    CRITICAL INDENTATION RULE:
    Multiple separate print_md() calls break markdown indentation because each call
    is processed independently by streamdown. For proper bullet point hierarchy,
    combine related content into a single print_md() call.

    Args:
        markdown_content: The markdown content to render

    Examples:
        ❌ WRONG - Breaks indentation (separate calls):
            print_md("**Main Point**")
            print_md("    Detail 1")
            print_md("    Detail 2")

            Result: All items appear as separate top-level bullets

        ✅ CORRECT - Proper indentation (single call):
            content = "**Main Point**\n"
            content += "    Detail 1\n"
            content += "    Detail 2"
            print_md(content)

            Result: Details properly indented under main point

        ✅ CORRECT - Status with details:
            status_text = "RAG Status:\n"
            status_text += f"    Active: {active}\n"
            status_text += f"    Collection: {collection}"
            print_md(status_text)

        ✅ CORRECT - Instructions with steps:
            help_text = "Setup Instructions:\n"
            help_text += "    1. Go to settings\n"
            help_text += "    2. Enable feature\n"
            help_text += "    3. Restart application"
            print_md(help_text)

    Guidelines:
        - Use single print_md() call for hierarchical content
        - Use "    " (4 spaces) for indentation levels
        - Separate print_md() calls are fine for unrelated content
        - Build multi-line strings with \n for proper formatting
        - Consider user experience - don't delay important info unnecessarily
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
        from settings_manager import SettingsManager
        streamdown_cmd = SettingsManager.getInstance().markdown_settings
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



"""
Print helper module for consistent user feedback messages.
Provides standardized formatting for status, info, error, and success messages.
"""

def print_info(message: str, newline_before: bool = False) -> None:
    """
    Print a status message with consistent formatting.

    Args:
        message: The status message to display
        newline_before: Whether to add a blank line before the message
    """
    if newline_before:
        print()

    print(f"- {message}")

def print_lines():
    """
    Print lines
    """


    print()
    print("=" * 50)
    print()

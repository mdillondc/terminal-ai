"""
RAG Configuration Module

Central configuration for RAG system settings including supported file types.
This ensures DRY principle - all RAG-related file type definitions are in one place.
Uses MIME type detection to automatically support text-based files.
"""

import os
import mimetypes
from typing import Set, Dict, Any

import magic

# Binary file extensions that require special processing
BINARY_FILE_EXTENSIONS: Set[str] = {
    '.pdf',   # PDF documents
    '.docx',  # Microsoft Word documents
    '.doc',   # Legacy Microsoft Word documents
    '.xlsx',  # Microsoft Excel spreadsheets
    '.xls',   # Legacy Microsoft Excel spreadsheets
    '.rtf',   # Rich Text Format
}

# File type metadata for binary files
BINARY_FILE_TYPE_INFO: Dict[str, Dict[str, Any]] = {
    '.pdf': {
        'name': 'PDF',
        'description': 'Portable Document Format files',
        'processor': 'pdf'
    },
    '.docx': {
        'name': 'Word Document',
        'description': 'Microsoft Word documents',
        'processor': 'docx'
    },
    '.doc': {
        'name': 'Word Document (Legacy)',
        'description': 'Legacy Microsoft Word documents',
        'processor': 'doc'
    },
    '.xlsx': {
        'name': 'Excel Spreadsheet',
        'description': 'Microsoft Excel spreadsheets',
        'processor': 'xlsx'
    },
    '.xls': {
        'name': 'Excel Spreadsheet (Legacy)',
        'description': 'Legacy Microsoft Excel spreadsheets',
        'processor': 'xls'
    },
    '.rtf': {
        'name': 'Rich Text Format',
        'description': 'Rich Text Format documents',
        'processor': 'rtf'
    }
}

def is_text_file_by_mime(file_path: str) -> bool:
    """
    Check if a file is a text file using MIME type detection.

    Args:
        file_path: Path to the file to check

    Returns:
        True if the file is detected as text, False otherwise
    """
    try:
        # Use python-magic for accurate MIME detection
        mime_type = magic.from_file(file_path, mime=True)
        return mime_type.startswith('text/') or mime_type in [
            'application/json',
            'application/xml',
            'application/javascript',
            'application/x-yaml',
            'application/x-sh'
        ]
    except Exception:
        # Fallback to built-in mimetypes
        mime_type, _ = mimetypes.guess_type(file_path)
        if mime_type:
            return mime_type.startswith('text/') or mime_type in [
                'application/json',
                'application/xml',
                'application/javascript',
                'application/x-yaml',
                'application/x-sh'
            ]
        return False

def get_supported_extensions() -> Set[str]:
    """
    Get the set of binary file extensions that require special processing.
    Note: Text files are detected dynamically via MIME types.

    Returns:
        Set of binary file extensions (including the dot, e.g., '.pdf')
    """
    return BINARY_FILE_EXTENSIONS.copy()

def is_supported_file(file_path: str) -> bool:
    """
    Check if a file is supported by the RAG system.
    Supports binary files (PDF, DOCX, etc.) and text files (detected via MIME type).

    Args:
        file_path: Path to the file to check

    Returns:
        True if the file type is supported, False otherwise
    """
    file_ext = os.path.splitext(file_path)[1].lower()

    # Check if it's a known binary file type
    if file_ext in BINARY_FILE_EXTENSIONS:
        return True

    # Check if it's a text file using MIME detection
    if os.path.exists(file_path):
        return is_text_file_by_mime(file_path)

    return False

def get_file_type_info(file_path: str) -> Dict[str, Any]:
    """
    Get metadata information for a file type.

    Args:
        file_path: Path to the file (used for MIME detection if needed)

    Returns:
        Dictionary with file type information
    """
    extension = os.path.splitext(file_path)[1].lower()

    # Check binary file types first
    if extension in BINARY_FILE_TYPE_INFO:
        return BINARY_FILE_TYPE_INFO[extension].copy()

    # For all other supported files (text files), return generic text info
    if is_supported_file(file_path):
        return {
            'name': 'Text file',
            'description': 'Text file',
            'processor': 'text'
        }

    return {}

def get_supported_extensions_display() -> str:
    """
    Get a human-readable string of supported file types for help messages.

    Returns:
        Description of supported file types
    """
    binary_exts = ", ".join(sorted(BINARY_FILE_EXTENSIONS))
    return f"Binary files: {binary_exts}; Text files: auto-detected via MIME type"
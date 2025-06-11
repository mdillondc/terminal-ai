"""
RAG Configuration Module

Central configuration for RAG system settings including supported file types.
This ensures DRY principle - all RAG-related file type definitions are in one place.
"""

from typing import Set, Dict, Any

# Supported file extensions for RAG processing
# To add support for a new file type, add it here and implement processing in DocumentProcessor
SUPPORTED_FILE_EXTENSIONS: Set[str] = {
    '.txt',   # Plain text files
    '.md',    # Markdown files  
    '.pdf',   # PDF documents
    # To add more file types, simply add them here:
    # '.docx',  # Microsoft Word documents
    # '.rtf',   # Rich Text Format
    # '.html',  # HTML documents
}

# File type metadata for display purposes
FILE_TYPE_INFO: Dict[str, Dict[str, Any]] = {
    '.txt': {
        'name': 'Text',
        'description': 'Plain text documents',
        'processor': 'text'
    },
    '.md': {
        'name': 'Markdown',
        'description': 'Markdown formatted documents',
        'processor': 'text'
    },
    '.pdf': {
        'name': 'PDF',
        'description': 'Portable Document Format files',
        'processor': 'pdf'
    }
    # When adding new file types above, also add their metadata here:
    # '.docx': {
    #     'name': 'Word Document',
    #     'description': 'Microsoft Word documents',
    #     'processor': 'docx'
    # }
}

def get_supported_extensions() -> Set[str]:
    """
    Get the set of supported file extensions.
    
    Returns:
        Set of file extensions (including the dot, e.g., '.txt')
    """
    return SUPPORTED_FILE_EXTENSIONS.copy()

def is_supported_file(filename: str) -> bool:
    """
    Check if a file is supported by the RAG system based on its extension.
    
    Args:
        filename: Name or path of the file to check
        
    Returns:
        True if the file type is supported, False otherwise
    """
    import os
    file_ext = os.path.splitext(filename)[1].lower()
    return file_ext in SUPPORTED_FILE_EXTENSIONS

def get_file_type_info(extension: str) -> Dict[str, Any]:
    """
    Get metadata information for a file type.
    
    Args:
        extension: File extension (e.g., '.pdf')
        
    Returns:
        Dictionary with file type information, or empty dict if not supported
    """
    return FILE_TYPE_INFO.get(extension.lower(), {})

def get_supported_extensions_display() -> str:
    """
    Get a human-readable string of supported file types for help messages.
    
    Returns:
        Comma-separated string of supported extensions (e.g., ".txt, .md, .pdf")
    """
    return ", ".join(sorted(SUPPORTED_FILE_EXTENSIONS))
"""
Search utility functions

This module provides shared utilities for search result processing,
including full content extraction from search result URLs.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any
from web_content_extractor import WebContentExtractor
from print_helper import print_md


def extract_full_content_from_search_results(raw_results: Dict[str, Any], settings_manager, llm_client_manager) -> Dict[str, Any]:
    """
    Extract full page content from search result URLs using WebContentExtractor.

    This function replaces the short content snippets in search results with
    full page content extracted using the WebContentExtractor with all its
    bypass capabilities (paywall bypass, bot detection avoidance, etc.).

    Args:
        raw_results: Raw search results dictionary containing 'results' array
        settings_manager: Settings manager instance
        llm_client_manager: LLM client manager for WebContentExtractor

    Returns:
        Modified raw_results with full content extracted where possible
    """
    if not raw_results or 'results' not in raw_results:
        return raw_results

    results = raw_results.get('results', [])
    if not results:
        return raw_results

    # Extract URLs for processing
    urls_to_extract = []
    for i, result in enumerate(results):
        url = result.get('url', '')
        if url and isinstance(url, str):
            urls_to_extract.append((i, url))

    if not urls_to_extract:
        return raw_results

    # Show simple extraction start message
    url_list = ""
    for i, (index, url) in enumerate(urls_to_extract):
        title = results[index].get('title', 'Unknown Source')
        url_list += f"    {title}\n"

    print_md(f"Extracting full content from all URLs...\n{url_list.rstrip()}")

    # Initialize WebContentExtractor
    extractor = WebContentExtractor(llm_client_manager)

    # Track extraction results
    successful_extractions = 0
    failed_extractions = 0
    truncated_count = 0
    total_chars_before = 0
    total_chars_after = 0
    extraction_results = {}

    # Process URLs in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=settings_manager.concurrent_workers) as executor:
        # Submit all extraction tasks
        future_to_index = {}
        for index, url in urls_to_extract:
            future = executor.submit(extractor.extract_content, url, verbose=False)
            future_to_index[future] = (index, url)

        # Process completed extractions
        for future in as_completed(future_to_index):
            index, url = future_to_index[future]

            try:
                extraction_result = future.result()

                if extraction_result.get('content') and not extraction_result.get('error'):
                    # Successful extraction - replace the content with truncation
                    original_content = results[index].get('content', '')
                    extracted_content = extraction_result['content']

                    # Apply truncation if configured (word-based)
                    truncate_length = getattr(settings_manager, 'searxng_extract_full_content_truncate', 1000)
                    words = extracted_content.split()
                    original_word_count = len(words)
                    total_chars_before += len(extracted_content)

                    if original_word_count > truncate_length:
                        extracted_content = ' '.join(words[:truncate_length]) + "..."
                        truncated_count += 1

                    total_chars_after += len(extracted_content)
                    results[index]['content'] = extracted_content

                    # Optionally update title if extraction provided a better one
                    if extraction_result.get('title') and not results[index].get('title'):
                        results[index]['title'] = extraction_result['title']

                    successful_extractions += 1
                    extraction_results[index] = {'status': 'success', 'words': len(extracted_content.split()), 'truncated': original_word_count > truncate_length}

                else:
                    # Extraction failed - keep original snippet
                    failed_extractions += 1
                    extraction_results[index] = {'status': 'failed'}

            except Exception as e:
                # Extraction failed - keep original snippet
                failed_extractions += 1
                extraction_results[index] = {'status': 'failed'}
                continue

    # Build simple results summary
    total_attempts = len(urls_to_extract)
    summary_text = "Extracted content from all URLs\n"

    # Add per-URL results
    for i, (index, url) in enumerate(urls_to_extract):
        title = results[index].get('title', 'Unknown Source')
        result = extraction_results.get(index, {'status': 'failed'})

        if result['status'] == 'success':
            word_info = f"({result['words']} words"
            if result.get('truncated'):
                word_info += ", truncated"
            word_info += ")"
            summary_text += f"    {title} {word_info}\n"
        else:
            summary_text += f"    {title} (extraction failed)\n"

    print_md(summary_text.rstrip())

    return raw_results
#!/usr/bin/env python3
"""
Test script to demonstrate the improved transparency of the web content extractor.
Shows how the system now provides detailed information about access restrictions
and bypass attempts.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from web_content_extractor import WebContentExtractor


def test_access_blocking_detection():
    """Test the improved access blocking detection."""
    print("=" * 60)
    print("TESTING ACCESS BLOCKING DETECTION")
    print("=" * 60)

    extractor = WebContentExtractor()

    test_cases = [
        ("Paywall content", "Please subscribe to continue reading this premium article"),
        ("Login wall", "Please log in to continue reading this content"),
        ("Bot detection", "Access denied - unusual traffic detected from your IP"),
        ("Geographic restriction", "This content is not available in your region"),
        ("Rate limiting", "Too many requests - please try again later"),
        ("Normal content", "This is a regular article with plenty of content to read")
    ]

    for description, content in test_cases:
        result = extractor._is_access_blocked(content)
        print(f"✓ {description}: {result or 'No restriction detected'}")

    print()


def test_method_transparency():
    """Test a real URL to see the transparency in action."""
    print("=" * 60)
    print("TESTING BYPASS METHOD TRANSPARENCY")
    print("=" * 60)

    extractor = WebContentExtractor()

    # Test with a URL that's likely to be blocked or restricted
    # Using a well-known news site that often has paywalls
    test_url = "https://www.nytimes.com/2024/01/01/technology/ai-future.html"

    print(f"Testing URL: {test_url}")
    print("Note: This will show detailed information about what the system is trying...")
    print()

    # This will demonstrate the transparency in real-time
    result = extractor.extract_content(test_url)

    print("\n" + "=" * 40)
    print("FINAL RESULT:")
    print("=" * 40)
    print(f"Title: {result.get('title', 'N/A')}")
    print(f"Content length: {len(result.get('content', '')) if result.get('content') else 0} characters")
    print(f"Error: {result.get('error', 'None')}")
    print(f"Warning: {result.get('warning', 'None')}")

    if result.get('content'):
        preview = result['content'][:200] + "..." if len(result['content']) > 200 else result['content']
        print(f"Content preview: {preview}")


def test_bypass_methods_individually():
    """Test individual bypass methods for transparency."""
    print("=" * 60)
    print("TESTING INDIVIDUAL BYPASS METHODS")
    print("=" * 60)

    extractor = WebContentExtractor()
    test_url = "https://example.com/blocked-article"

    print("Testing each bypass method individually to show transparency...")
    print()

    # Test Archive.org method
    print("1. Testing Archive.org method:")
    try:
        archive_result = extractor._try_archive_org(test_url)
        print(f"   Result: {archive_result.get('error', 'Success')}")
    except Exception as e:
        print(f"   Error: {e}")

    print()

    # Test bot user agent method
    print("2. Testing bot user agent method:")
    try:
        bot_result = extractor._try_bot_user_agent(test_url)
        print(f"   Result: {bot_result.get('error', 'Success')}")
    except Exception as e:
        print(f"   Error: {e}")

    print()

    # Test print version method
    print("3. Testing print version method:")
    try:
        print_result = extractor._try_print_version(test_url)
        print(f"   Result: {print_result.get('error', 'Success')}")
    except Exception as e:
        print(f"   Error: {e}")

    print()

    # Test AMP version method
    print("4. Testing AMP version method:")
    try:
        amp_result = extractor._try_amp_version(test_url)
        print(f"   Result: {amp_result.get('error', 'Success')}")
    except Exception as e:
        print(f"   Error: {e}")


def main():
    """Run all transparency tests."""
    print("WEB CONTENT EXTRACTOR TRANSPARENCY TEST")
    print("This demonstrates the improved transparency and generic terminology")
    print("for handling various types of access restrictions.")
    print()

    # Test 1: Access blocking detection
    test_access_blocking_detection()

    # Test 2: Individual bypass methods (doesn't require internet)
    test_bypass_methods_individually()

    print("\n" + "=" * 60)
    print("TRANSPARENCY IMPROVEMENTS SUMMARY")
    print("=" * 60)
    print("✓ Generic terminology: 'access restriction' instead of just 'paywall'")
    print("✓ Specific block type detection: paywall, login, bot detection, etc.")
    print("✓ Detailed bypass method reporting: shows what's being tried")
    print("✓ Success/failure transparency: reports why methods fail")
    print("✓ Content quality reporting: shows word counts and content quality")
    print("✓ Method-specific details: timestamps for Archive.org, user agents, etc.")
    print()
    print("To see full transparency in action, uncomment the test_method_transparency() call below")
    print("and run with internet connection to see real-time bypass attempts.")

    # Uncomment the line below to test with a real URL (requires internet)
    # test_method_transparency()


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Test script for o1 and o3 model detection in LLMClientManager
"""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from llm_client_manager import LLMClientManager
from openai import OpenAI

def test_o1_model_detection():
    """Test o1 model detection"""
    # Create a dummy OpenAI client for testing
    client = OpenAI(api_key="test")
    manager = LLMClientManager(client)

    # Test cases for o1 models (should return True)
    o1_models = [
        'o1',
        'O1',
        'o1-preview',
        'o1-mini',
        'o1-pro',
        'o1-2024-12-17',
        'o1-mini-2024-09-12',
        'o1-pro-2024-12-17'
    ]

    print("Testing o1 model detection:")
    for model in o1_models:
        result = manager._is_o1_model(model)
        status = "✓" if result else "✗"
        print(f"  {status} {model}: {result}")
        assert result, f"Failed to detect o1 model: {model}"

    # Test cases for non-o1 models (should return False)
    non_o1_models = [
        'gpt-4',
        'gpt-4o',
        'gpt-3.5-turbo',
        'o3',
        'o3-mini',
        'claude-3',
        'llama-2'
    ]

    print("\nTesting non-o1 models (should be False):")
    for model in non_o1_models:
        result = manager._is_o1_model(model)
        status = "✓" if not result else "✗"
        print(f"  {status} {model}: {result}")
        assert not result, f"Incorrectly detected non-o1 model as o1: {model}"

def test_o3_model_detection():
    """Test o3 model detection"""
    # Create a dummy OpenAI client for testing
    client = OpenAI(api_key="test")
    manager = LLMClientManager(client)

    # Test cases for o3 models (should return True)
    o3_models = [
        'o3',
        'O3',
        'o3-mini',
        'o3-pro',
        'o3-2025-04-16',
        'o3-mini-2025-01-31',
        'o3-pro-2025-06-10'
    ]

    print("\nTesting o3 model detection:")
    for model in o3_models:
        result = manager._is_o3_model(model)
        status = "✓" if result else "✗"
        print(f"  {status} {model}: {result}")
        assert result, f"Failed to detect o3 model: {model}"

    # Test cases for non-o3 models (should return False)
    non_o3_models = [
        'gpt-4',
        'gpt-4o',
        'gpt-3.5-turbo',
        'o1',
        'o1-mini',
        'claude-3',
        'llama-2'
    ]

    print("\nTesting non-o3 models (should be False):")
    for model in non_o3_models:
        result = manager._is_o3_model(model)
        status = "✓" if not result else "✗"
        print(f"  {status} {model}: {result}")
        assert not result, f"Incorrectly detected non-o3 model as o3: {model}"

def main():
    """Run all tests"""
    print("Testing o1 and o3 model detection...")
    print("=" * 50)

    try:
        test_o1_model_detection()
        test_o3_model_detection()
        print("\n" + "=" * 50)
        print("✓ All tests passed!")

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
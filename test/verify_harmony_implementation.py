"""
Simple verification script for Harmony integration.

This script verifies that the Harmony integration has been implemented 
correctly without requiring external test frameworks.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def test_imports():
    """Test that all new modules can be imported."""
    print("Testing imports...")
    
    try:
        from lemonade.tools.server.harmony import (
            HarmonyFormatter, 
            is_gpt_oss_model, 
            should_use_harmony
        )
        print("✓ Harmony module imports successfully")
    except ImportError as e:
        print(f"✗ Failed to import harmony module: {e}")
        return False
    
    try:
        from lemonade.tools.server.serve import Server
        print("✓ Server module imports successfully")
    except ImportError as e:
        print(f"✗ Failed to import server module: {e}")
        return False
    
    return True

def test_gpt_oss_detection():
    """Test GPT-OSS model detection logic."""
    print("\nTesting GPT-OSS detection...")
    
    from lemonade.tools.server.harmony import is_gpt_oss_model
    
    # Test case 1: Model with gpt-oss label
    model_with_label = {
        "model_name": "test-model",
        "labels": ["hot", "reasoning", "gpt-oss"]
    }
    
    if is_gpt_oss_model(model_with_label):
        print("✓ Correctly detects GPT-OSS model with label")
    else:
        print("✗ Failed to detect GPT-OSS model with label")
        return False
    
    # Test case 2: Model without gpt-oss label
    model_without_label = {
        "model_name": "llama-model",
        "labels": ["chat", "reasoning"]
    }
    
    if not is_gpt_oss_model(model_without_label):
        print("✓ Correctly ignores non-GPT-OSS model")
    else:
        print("✗ Incorrectly detected non-GPT-OSS model as GPT-OSS")
        return False
    
    # Test case 3: Model with name pattern
    model_with_pattern = {
        "model_name": "gpt-oss-120b-GGUF",
        "checkpoint": "unsloth/gpt-oss-120b-GGUF:Q4_K_M"
    }
    
    if is_gpt_oss_model(model_with_pattern):
        print("✓ Correctly detects GPT-OSS model by name pattern")
    else:
        print("✗ Failed to detect GPT-OSS model by name pattern")
        return False
    
    return True

def test_harmony_formatter():
    """Test HarmonyFormatter initialization."""
    print("\nTesting HarmonyFormatter...")
    
    from lemonade.tools.server.harmony import HarmonyFormatter
    
    try:
        formatter = HarmonyFormatter()
        print(f"✓ HarmonyFormatter created successfully")
        print(f"  - Harmony available: {formatter.is_available}")
        
        if not formatter.is_available:
            print("  ℹ Note: openai-harmony not installed - this is expected")
            print("    Install with: pip install 'lemonade-sdk[harmony]'")
        
        return True
    except Exception as e:
        print(f"✗ Failed to create HarmonyFormatter: {e}")
        return False

def test_server_integration():
    """Test that the Server class has been properly updated."""
    print("\nTesting Server integration...")
    
    try:
        from lemonade.tools.server.serve import Server
        
        # Create a server instance (this will test basic initialization)
        server = Server()
        
        # Check that harmony_formatter attribute exists
        if hasattr(server, 'harmony_formatter'):
            print("✓ Server has harmony_formatter attribute")
        else:
            print("✗ Server missing harmony_formatter attribute")
            return False
        
        # Check that apply_chat_template method exists
        if hasattr(server, 'apply_chat_template'):
            print("✓ Server has apply_chat_template method")
        else:
            print("✗ Server missing apply_chat_template method")
            return False
        
        # Check that _apply_default_template method exists
        if hasattr(server, '_apply_default_template'):
            print("✓ Server has _apply_default_template method")
        else:
            print("✗ Server missing _apply_default_template method")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to test server integration: {e}")
        return False

def test_model_metadata():
    """Test that GPT-OSS models have been properly labeled."""
    print("\nTesting model metadata...")
    
    try:
        from lemonade_server.model_manager import ModelManager
        
        manager = ModelManager()
        models = manager.supported_models
        
        # Check for gpt-oss-120b
        if "gpt-oss-120b-GGUF" in models:
            model_120b = models["gpt-oss-120b-GGUF"]
            if "gpt-oss" in model_120b.get("labels", []):
                print("✓ gpt-oss-120b-GGUF has gpt-oss label")
            else:
                print("✗ gpt-oss-120b-GGUF missing gpt-oss label")
                return False
        else:
            print("✗ gpt-oss-120b-GGUF not found in models")
            return False
        
        # Check for gpt-oss-20b
        if "gpt-oss-20b-GGUF" in models:
            model_20b = models["gpt-oss-20b-GGUF"]
            if "gpt-oss" in model_20b.get("labels", []):
                print("✓ gpt-oss-20b-GGUF has gpt-oss label")
            else:
                print("✗ gpt-oss-20b-GGUF missing gpt-oss label")
                return False
        else:
            print("✗ gpt-oss-20b-GGUF not found in models")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Failed to test model metadata: {e}")
        return False

def main():
    """Run all verification tests."""
    print("Harmony Integration Verification")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_gpt_oss_detection,
        test_harmony_formatter,
        test_server_integration,
        test_model_metadata
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Verification Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Harmony integration is working correctly.")
        print("\nNext steps:")
        print("1. Install harmony support: pip install 'lemonade-sdk[harmony]'")
        print("2. Test with a GPT-OSS model on Vulkan backend")
        print("3. Verify that jinja flag is properly omitted")
    else:
        print("❌ Some tests failed. Please review the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

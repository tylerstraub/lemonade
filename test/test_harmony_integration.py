"""
Test suite for Harmony integration with GPT-OSS models.

This module tests the integration of OpenAI's Harmony response format
into the Lemonade server for improved GPT-OSS model compatibility.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lemonade.tools.server.harmony import (
    HarmonyFormatter, 
    is_gpt_oss_model, 
    should_use_harmony
)


class TestHarmonyFormatter(unittest.TestCase):
    """Test the HarmonyFormatter class functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.formatter = HarmonyFormatter()
    
    def test_harmony_availability_check(self):
        """Test that harmony availability is checked correctly."""
        # The availability depends on whether openai-harmony is installed
        # So we just test that the property exists and returns a boolean
        self.assertIsInstance(self.formatter.is_available, bool)
    
    @patch('lemonade.tools.server.harmony.HarmonyFormatter._check_harmony_availability')
    def test_format_messages_without_harmony(self, mock_check):
        """Test that format_messages raises ImportError when harmony is unavailable."""
        mock_check.return_value = False
        formatter = HarmonyFormatter()
        
        messages = [{"role": "user", "content": "Hello"}]
        
        with self.assertRaises(ImportError):
            formatter.format_messages(messages)
    
    def test_format_messages_invalid_input(self):
        """Test that format_messages handles invalid input correctly."""
        if not self.formatter.is_available:
            self.skipTest("openai-harmony not available")
        
        # Test with invalid message format
        invalid_messages = [{"content": "Hello"}]  # Missing role
        
        with self.assertRaises(ValueError):
            self.formatter.format_messages(invalid_messages)
    
    @patch('lemonade.tools.server.harmony.logging')
    def test_format_messages_unknown_role(self, mock_logging):
        """Test that unknown roles are handled gracefully."""
        if not self.formatter.is_available:
            self.skipTest("openai-harmony not available")
        
        messages = [{"role": "unknown_role", "content": "Hello"}]
        
        try:
            result = self.formatter.format_messages(messages)
            # Should not raise an exception, just log a warning
            mock_logging.warning.assert_called()
            self.assertIsNotNone(result)
        except ImportError:
            # Expected if openai-harmony is not installed
            pass


class TestGPTOSSDetection(unittest.TestCase):
    """Test GPT-OSS model detection functions."""
    
    def test_is_gpt_oss_model_with_label(self):
        """Test detection when gpt-oss label is present."""
        model_info = {
            "model_name": "test-model",
            "labels": ["hot", "reasoning", "gpt-oss"]
        }
        
        self.assertTrue(is_gpt_oss_model(model_info))
    
    def test_is_gpt_oss_model_without_label(self):
        """Test detection when gpt-oss label is missing."""
        model_info = {
            "model_name": "test-model",
            "labels": ["hot", "reasoning"]
        }
        
        self.assertFalse(is_gpt_oss_model(model_info))
    
    def test_is_gpt_oss_model_with_name_pattern(self):
        """Test detection using model name patterns."""
        model_info = {
            "model_name": "gpt-oss-120b-GGUF",
            "checkpoint": "unsloth/gpt-oss-120b-GGUF:Q4_K_M"
        }
        
        self.assertTrue(is_gpt_oss_model(model_info))
    
    def test_is_gpt_oss_model_invalid_input(self):
        """Test detection with invalid input."""
        self.assertFalse(is_gpt_oss_model(None))
        self.assertFalse(is_gpt_oss_model("invalid"))
        self.assertFalse(is_gpt_oss_model({}))
    
    def test_should_use_harmony_gpt_oss_vulkan(self):
        """Test that Harmony is used for gpt-oss-120b on Vulkan."""
        model_info = {
            "model_name": "gpt-oss-120b-GGUF",
            "labels": ["gpt-oss"]
        }
        
        # Mock harmony formatter as available
        harmony_formatter = Mock()
        harmony_formatter.is_available = True
        
        result = should_use_harmony(model_info, "vulkan", harmony_formatter)
        self.assertTrue(result)
    
    def test_should_use_harmony_non_gpt_oss(self):
        """Test that Harmony is not used for non-GPT-OSS models."""
        model_info = {
            "model_name": "llama-model",
            "labels": ["chat"]
        }
        
        harmony_formatter = Mock()
        harmony_formatter.is_available = True
        
        result = should_use_harmony(model_info, "vulkan", harmony_formatter)
        self.assertFalse(result)
    
    def test_should_use_harmony_unavailable(self):
        """Test that Harmony is not used when unavailable."""
        model_info = {
            "model_name": "gpt-oss-120b-GGUF",
            "labels": ["gpt-oss"]
        }
        
        harmony_formatter = Mock()
        harmony_formatter.is_available = False
        
        result = should_use_harmony(model_info, "vulkan", harmony_formatter)
        self.assertFalse(result)


class TestHarmonyIntegration(unittest.TestCase):
    """Test the integration of Harmony formatting in the server."""
    
    @patch('lemonade.tools.server.serve.should_use_harmony')
    @patch('lemonade.tools.server.serve.HarmonyFormatter')
    def test_server_apply_chat_template_with_harmony(self, mock_formatter_class, mock_should_use):
        """Test that the server uses Harmony when appropriate."""
        from lemonade.tools.server.serve import Server
        
        # Mock the formatter
        mock_formatter = Mock()
        mock_formatter.format_messages.return_value = [1, 2, 3]  # Mock token IDs
        mock_formatter_class.return_value = mock_formatter
        
        # Mock should_use_harmony to return True
        mock_should_use.return_value = True
        
        # Create a mock server
        server = Server()
        server.llm_loaded = Mock()
        server.llm_loaded.__dict__ = {"model_name": "gpt-oss-120b"}
        server.llamacpp_backend = "vulkan"
        
        # Mock tokenizer
        server.tokenizer = Mock()
        server.tokenizer.decode.return_value = "formatted text"
        server.tokenizer.chat_template = None
        
        messages = [{"role": "user", "content": "Hello"}]
        
        result = server.apply_chat_template(messages)
        
        # Should use Harmony formatting
        mock_formatter.format_messages.assert_called_once_with(messages, None)
        server.tokenizer.decode.assert_called_once_with([1, 2, 3])
        self.assertEqual(result, "formatted text")


if __name__ == '__main__':
    unittest.main()

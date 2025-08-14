"""
Test suite for Harmony streaming parser functionality.

This module tests the new real-time streaming parser that enables proper
streaming output for GPT-OSS models using Harmony format while preserving
the structured response parsing.
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from lemonade.tools.server.harmony import HarmonyStreamingParser, HarmonyChannel


class TestHarmonyStreamingParser(unittest.TestCase):
    """Test the streaming Harmony response parser."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = HarmonyStreamingParser(expect_gpt_oss_format=False)
    
    def test_reset_functionality(self):
        """Test that parser resets correctly."""
        # Modify parser state
        self.parser.current_channel = HarmonyChannel.ANALYSIS
        self.parser.buffer = "test content"
        self.parser.has_harmony_structure = True
        
        # Reset and verify
        self.parser.reset()
        self.assertEqual(self.parser.current_channel, HarmonyChannel.UNKNOWN)
        self.assertEqual(self.parser.buffer, "")
        self.assertFalse(self.parser.has_harmony_structure)
    
    def test_raw_content_passthrough(self):
        """Test that non-Harmony content passes through unchanged."""
        chunk_text = "This is just normal text without Harmony structure."
        result = self.parser.parse_chunk(chunk_text)
        
        self.assertEqual(result["type"], "raw")
        self.assertEqual(result["content"], chunk_text)
        self.assertFalse(self.parser.has_harmony_structure)
    
    def test_harmony_channel_detection(self):
        """Test detection and switching between Harmony channels."""
        # Test analysis channel detection
        chunk = "<|channel|>analysis<|message|>This is reasoning content"
        result = self.parser.parse_chunk(chunk)
        
        self.assertTrue(self.parser.has_harmony_structure)
        self.assertEqual(self.parser.current_channel, HarmonyChannel.ANALYSIS)
        self.assertEqual(result["type"], "reasoning")
        self.assertEqual(result["content"], "<think>This is reasoning content")
    
    def test_commentary_channel_handling(self):
        """Test handling of commentary channel (treated as raw)."""
        chunk = "<|channel|>commentary<|message|>Tool output here"
        result = self.parser.parse_chunk(chunk)
        
        self.assertEqual(self.parser.current_channel, HarmonyChannel.COMMENTARY)
        self.assertEqual(result["type"], "raw")
        self.assertEqual(result["content"], "Tool output here")
    
    def test_partial_channel_marker_buffering(self):
        """Test that partial channel markers are properly buffered."""
        # Send partial channel marker
        result1 = self.parser.parse_chunk("<|chann")
        self.assertEqual(result1["content"], "")
        
        # Complete the marker
        result2 = self.parser.parse_chunk("el|>analysis<|message|>Content")
        self.assertEqual(result2["type"], "reasoning")
        self.assertEqual(result2["content"], "<think>Content")


class TestHarmonyStreamingIntegration(unittest.TestCase):
    """Test integration of streaming parser with the actual Harmony response format."""
    
    def test_real_harmony_response_simulation(self):
        """Simulate processing the actual muffin man response format."""
        parser = HarmonyStreamingParser(expect_gpt_oss_format=True)
        
        # Simulate the actual response as one chunk (as it might appear in streaming)
        # Based on the user's example, the format is:
        # 1. Reasoning content (should be wrapped in <think>)  
        # 2. <|start|>assistant<|channel|>final<|message|>
        # 3. Final response content (should NOT be in <think>)
        
        reasoning_part = "The user asks \"do you know the muffin man?\" That's a simple question. According to policy, it's a general knowledge question. We can answer. The muffin man is a nursery rhyme character. Provide a brief answer. No disallowed content. So respond.\n\n"
        
        final_part = "<|start|>assistant<|channel|>final<|message|>Yes! The \"Muffin Man\" is a character from a traditional English nursery rhyme that dates back to the 18th century. In the rhyme, the Muffin Man lives \"on Drury Lane,\" and the song is often sung as a call‑and‑response chant:"
        
        # Test as separate chunks to simulate streaming
        chunks = [
            reasoning_part,
            final_part
        ]
        
        results = []
        for chunk in chunks:
            result = parser.parse_chunk(chunk)
            if result["content"]:  # Only collect non-empty results
                results.append(result)
                print(f"Chunk: '{chunk[:50]}...' -> Type: {result['type']}, Content: '{result['content'][:50]}...'")
            
            # Check for pending outputs (simulating the streaming behavior)
            while parser.has_pending_outputs():
                pending = parser.parse_chunk("")
                if pending["content"]:
                    results.append(pending)
                    print(f"Pending: -> Type: {pending['type']}, Content: '{pending['content'][:50]}...'")
        
        # Finalize
        final = parser.finalize()
        if final["content"]:
            results.append(final)
            print(f"Final: Type: {final['type']}, Content: '{final['content'][:50]}...'")
        
        # The reasoning should be properly detected and marked as analysis
        reasoning_results = [r for r in results if r["type"] == "reasoning"]
        final_results = [r for r in results if r["type"] == "final"]
        
        print(f"Reasoning results: {len(reasoning_results)}")
        print(f"Final results: {len(final_results)}")
        
        # Should have both reasoning and final content
        self.assertTrue(len(reasoning_results) > 0, "Should detect reasoning content")
        self.assertTrue(len(final_results) > 0, "Should detect final content")
        
        # Verify final complete result has proper structure
        complete = parser.get_final_result()
        print(f"Complete result: {complete['content'][:100]}...")
        
        # Should have think tags around reasoning and final content separate
        self.assertIn("<think>", complete["content"])
        self.assertIn("</think>", complete["content"])
        self.assertIn("nursery rhyme", complete["content"])
        
        # Template decorators should be filtered out
        self.assertNotIn("<|start|>assistant", complete["content"])
        self.assertNotIn("<|channel|>final<|message|>", complete["content"])

    def test_chunked_streaming_fix(self):
        """Test the fix for template markers split across multiple chunks."""
        parser = HarmonyStreamingParser(expect_gpt_oss_format=True)
        
        # Simulate the exact chunking pattern that was causing issues
        chunks = [
            "Reasoning content here.",
            ".<|start|>ass",
            "istant<|chan", 
            "nel|>final<|mes",
            "sage|>Final response content"
        ]
        
        outputs = []
        for chunk in chunks:
            result = parser.parse_chunk(chunk)
            if result["content"]:
                outputs.append(result)
            
            # Check for pending outputs
            while parser.has_pending_outputs():
                pending = parser.parse_chunk("")
                if pending["content"]:
                    outputs.append(pending)
        
        # Should have both reasoning and final outputs
        reasoning_outputs = [o for o in outputs if o["type"] == "reasoning"]
        final_outputs = [o for o in outputs if o["type"] == "final"]
        
        self.assertTrue(len(reasoning_outputs) > 0, "Should have reasoning outputs")
        self.assertTrue(len(final_outputs) > 0, "Should have final outputs")
        
        # Verify structure
        full_content = "".join(o["content"] for o in outputs)
        self.assertIn("<think>", full_content)
        self.assertIn("</think>", full_content)
        self.assertIn("Final response content", full_content)
        
        # No template leakage
        self.assertNotIn("<|start|>", full_content)
        self.assertNotIn("<|channel|>", full_content)


if __name__ == "__main__":
    unittest.main()
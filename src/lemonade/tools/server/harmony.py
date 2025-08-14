"""
Harmony template formatter for GPT-OSS models.

This module provides integration with OpenAI's Harmony response format,
which is specifically designed for GPT-OSS models to handle structured
conversations, reasoning, and tool calls.
"""

import logging
import re
from typing import List, Dict, Any, Optional
from enum import Enum


class HarmonyChannel(Enum):
    """Harmony response channels."""
    UNKNOWN = "unknown"
    ANALYSIS = "analysis"  # Reasoning content
    FINAL = "final"        # User-facing response
    COMMENTARY = "commentary"  # Tool outputs


class HarmonyStreamingParser:
    """
    Real-time streaming parser for Harmony responses that preserves streaming behavior.
    
    This parser can incrementally process Harmony format tokens as they arrive,
    allowing for true streaming output while properly handling the structured
    Harmony format with analysis and final channels.
    """
    
    def __init__(self, expect_gpt_oss_format=True):
        """
        Initialize the parser.
        
        Args:
            expect_gpt_oss_format: If True, treats initial content as reasoning even without markers.
                                 If False, requires explicit channel markers.
        """
        self.expect_gpt_oss_format = expect_gpt_oss_format
        self.reset()
    
    def reset(self):
        """Reset parser state for a new response."""
        self.current_channel = HarmonyChannel.UNKNOWN
        self.buffer = ""
        self.content_started = False
        self.reasoning_content = ""
        self.final_content = ""
        self.has_harmony_structure = False
        self._raw_content = ""
        self._pending_outputs = []  # Queue for multiple outputs from single chunk
        
    def parse_chunk(self, chunk_text: str) -> Dict[str, str]:
        """
        Parse a chunk of text and return any content ready for streaming.
        
        Args:
            chunk_text: Raw text chunk from the model
            
        Returns:
            Dictionary with 'type' and 'content' keys:
            - type: 'reasoning', 'final', or 'raw' 
            - content: Text ready to stream to user
        """
        # Check if we have pending outputs from previous processing
        if self._pending_outputs:
            return self._pending_outputs.pop(0)
            
        if not chunk_text:
            return {"type": "raw", "content": ""}
            
        self.buffer += chunk_text
        
        # Look for channel transitions in the buffer
        channel_match = re.search(r'<\|channel\|>(\w+)<\|message\|>', self.buffer)
        start_match = re.search(r'<\|start\|>assistant', self.buffer)
        
        # Check if we're in the middle of a potential marker (more comprehensive)
        has_partial_marker = (
            '<|' in self.buffer and 
            not channel_match and 
            not start_match and
            (self.buffer.endswith('<|') or 
             self.buffer.endswith('<|s') or 
             self.buffer.endswith('<|st') or 
             self.buffer.endswith('<|sta') or 
             self.buffer.endswith('<|star') or 
             self.buffer.endswith('<|start') or 
             self.buffer.endswith('<|start|') or 
             self.buffer.endswith('<|start|>') or 
             self.buffer.endswith('<|start|>a') or 
             self.buffer.endswith('<|start|>as') or 
             self.buffer.endswith('<|start|>ass') or 
             self.buffer.endswith('<|start|>assi') or 
             self.buffer.endswith('<|start|>assis') or 
             self.buffer.endswith('<|start|>assist') or 
             self.buffer.endswith('<|start|>assista') or 
             self.buffer.endswith('<|start|>assistan') or 
             self.buffer.endswith('<|start|>assistant') or 
             self.buffer.endswith('<|start|>assistant<') or 
             self.buffer.endswith('<|start|>assistant<|') or 
             self.buffer.endswith('<|start|>assistant<|c') or 
             self.buffer.endswith('<|start|>assistant<|ch') or 
             self.buffer.endswith('<|start|>assistant<|cha') or 
             self.buffer.endswith('<|start|>assistant<|chan') or 
             self.buffer.endswith('<|start|>assistant<|chann') or 
             self.buffer.endswith('<|start|>assistant<|channe') or 
             self.buffer.endswith('<|start|>assistant<|channel') or 
             self.buffer.endswith('<|start|>assistant<|channel|') or 
             self.buffer.endswith('<|start|>assistant<|channel|>') or
             bool(re.search(r'<\|(?:start|channel|message)', self.buffer)))
        )
        
        if start_match and not self.has_harmony_structure:
            # Beginning of Harmony structured response
            self.has_harmony_structure = True
            
            # Everything before <|start|>assistant is reasoning content
            before_start = self.buffer[:start_match.start()]
            
            # Process reasoning content if any
            reasoning_output = {"type": "raw", "content": ""}
            if before_start.strip():
                # Filter and process reasoning content
                filtered_before = self._filter_template_decorators(before_start)
                if filtered_before.strip():
                    # This is reasoning content that should be wrapped in <think>
                    if not self.content_started:
                        self.content_started = True
                        self.reasoning_content += filtered_before
                        reasoning_output = {"type": "reasoning", "content": f"<think>{filtered_before}"}
                    else:
                        self.reasoning_content += filtered_before
                        reasoning_output = {"type": "reasoning", "content": filtered_before}
            
            # Look for immediate channel marker after start
            after_start = self.buffer[start_match.end():]
            immediate_channel = re.search(r'<\|channel\|>(\w+)<\|message\|>', after_start)
            
            if immediate_channel:
                # Direct transition to specific channel
                channel_name = immediate_channel.group(1)
                if channel_name == "final":
                    self.current_channel = HarmonyChannel.FINAL
                    # Close reasoning if we have it
                    if self.reasoning_content and reasoning_output["content"]:
                        reasoning_output["content"] += "</think>"
                    elif self.reasoning_content and not reasoning_output["content"]:
                        reasoning_output = {"type": "reasoning", "content": "</think>"}
                elif channel_name == "analysis":
                    self.current_channel = HarmonyChannel.ANALYSIS
                else:
                    self.current_channel = HarmonyChannel.UNKNOWN
                
                # Process content after the channel marker
                after_channel = after_start[immediate_channel.end():]
                self.buffer = ""
                self.content_started = False
                
                if after_channel:
                    filtered_after = self._filter_template_decorators(after_channel)
                    if filtered_after.strip():
                        final_output = self._process_channel_content(filtered_after)
                        # Queue the final output to be returned next
                        if final_output["content"]:
                            self._pending_outputs.append(final_output)
                        
                        # Return the transition closure if we have reasoning
                        if reasoning_output["content"]:
                            return reasoning_output
                        else:
                            # No reasoning closure needed, return final directly
                            return final_output if final_output["content"] else {"type": "raw", "content": ""}
                        
                return reasoning_output if reasoning_output["content"] else {"type": "raw", "content": ""}
            else:
                # No immediate channel found, but keep looking for partial channel markers
                # Check if we have partial channel marker that spans chunks
                if '<|' in after_start:
                    # Keep the buffer content after start match for next chunk
                    self.buffer = after_start
                    self.current_channel = HarmonyChannel.ANALYSIS
                    self.content_started = False
                    return reasoning_output if reasoning_output["content"] else {"type": "raw", "content": ""}
                else:
                    # No channel markers, continue in reasoning mode
                    self.current_channel = HarmonyChannel.ANALYSIS
                    self.buffer = ""
                    return reasoning_output if reasoning_output["content"] else {"type": "raw", "content": ""}
            
        elif channel_match:
            self.has_harmony_structure = True
            channel_name = channel_match.group(1)
            
            # Process content before channel marker (in current channel)
            before_channel = self.buffer[:channel_match.start()]
            output = {"type": "raw", "content": ""}
            
            if before_channel and self.current_channel != HarmonyChannel.UNKNOWN:
                filtered_before = self._filter_template_decorators(before_channel)
                if filtered_before.strip():
                    output = self._process_channel_content(filtered_before)
            
            # Update current channel
            old_channel = self.current_channel
            if channel_name == "analysis":
                self.current_channel = HarmonyChannel.ANALYSIS
            elif channel_name == "final":
                self.current_channel = HarmonyChannel.FINAL
            elif channel_name == "commentary":
                self.current_channel = HarmonyChannel.COMMENTARY
            else:
                self.current_channel = HarmonyChannel.UNKNOWN
                
            # Handle transition from analysis to final
            if (old_channel == HarmonyChannel.ANALYSIS and 
                self.current_channel == HarmonyChannel.FINAL and 
                self.reasoning_content):
                # Close reasoning and prepare for final content
                if output["content"]:
                    output["content"] += "</think>"
                else:
                    output = {"type": "reasoning", "content": "</think>"}
            
            # Keep everything after the channel marker for next processing
            after_channel = self.buffer[channel_match.end():]
            self.buffer = ""
            self.content_started = False
            
            # If there's content after the channel marker, process it immediately
            if after_channel:
                filtered_after = self._filter_template_decorators(after_channel)
                if filtered_after.strip():
                    next_output = self._process_channel_content(filtered_after)
                    if next_output["content"]:
                        # Queue the next output for later retrieval
                        self._pending_outputs.append(next_output)
                        
                        # Return transition output if we have it
                        if output["content"]:
                            return output
                        else:
                            # No transition needed, return the content directly
                            return next_output
            
            return output
            
        else:
            # No channel marker found, but check for partial channel markers first
            if self.has_harmony_structure and '<|' in self.buffer and not channel_match:
                # We might have a partial channel marker, keep buffering
                return {"type": "raw", "content": ""}
            elif self.current_channel != HarmonyChannel.UNKNOWN:
                # We're in a known channel, process the filtered content
                filtered_content = self._filter_template_decorators(chunk_text)
                if filtered_content.strip():
                    output = self._process_channel_content(filtered_content)
                    self.buffer = ""  # Clear buffer after processing
                    return output
                else:
                    # Content was filtered out, continue buffering
                    return {"type": "raw", "content": ""}
            else:
                # No Harmony structure detected yet
                if not self.has_harmony_structure:
                    # Look for potential partial markers - be more aggressive about buffering
                    if has_partial_marker or ('<|' in self.buffer and not channel_match and not start_match):
                        # Might be a partial marker, keep buffering
                        return {"type": "raw", "content": ""}
                    else:
                        # Only treat as reasoning for GPT-OSS format, otherwise pass through
                        # Process content that doesn't have markers
                        if self.expect_gpt_oss_format:
                            # For GPT-OSS models, content before any structure markers is reasoning
                            # Filter the entire buffer before processing
                            filtered_buffer = self._filter_template_decorators(self.buffer)
                            if filtered_buffer.strip():
                                if not self.content_started:
                                    self.content_started = True
                                    self.reasoning_content += filtered_buffer
                                    self.buffer = ""
                                    return {"type": "reasoning", "content": f"<think>{filtered_buffer}"}
                                else:
                                    self.reasoning_content += filtered_buffer
                                    self.buffer = ""
                                    return {"type": "reasoning", "content": filtered_buffer}
                            else:
                                # Filtered content is empty, continue buffering
                                return {"type": "raw", "content": ""}
                        else:
                            # Not GPT-OSS format, pass through as raw
                            filtered_content = self._filter_template_decorators(chunk_text)
                            if filtered_content.strip():
                                self._raw_content += filtered_content
                                self.buffer = ""
                                return {"type": "raw", "content": filtered_content}
                            else:
                                return {"type": "raw", "content": ""}
                else:
                    # Had structure before but now in unknown state
                    filtered_content = self._filter_template_decorators(chunk_text)
                    return {"type": "raw", "content": filtered_content}
    
    def _filter_template_decorators(self, content: str) -> str:
        """Remove Harmony template decorators that shouldn't appear in output."""
        # Remove various template markers
        filtered = content
        
        # Remove start/end markers
        filtered = re.sub(r'<\|start\|>assistant', '', filtered)
        filtered = re.sub(r'<\|end\|>', '', filtered)
        
        # Remove channel markers 
        filtered = re.sub(r'<\|channel\|>\w+<\|message\|>', '', filtered)
        
        # Remove other system tokens
        filtered = re.sub(r'<\|system\|>', '', filtered)
        filtered = re.sub(r'<\|user\|>', '', filtered)
        filtered = re.sub(r'<\|assistant\|>', '', filtered)
        
        return filtered
    
    def has_pending_outputs(self) -> bool:
        """Check if there are pending outputs to be retrieved."""
        return len(self._pending_outputs) > 0
    
    def _process_channel_content(self, content: str) -> Dict[str, str]:
        """Process content within a specific channel."""
        if not content:
            return {"type": "raw", "content": ""}
            
        if self.current_channel == HarmonyChannel.ANALYSIS:
            # Stream reasoning content as <think> format
            self.reasoning_content += content
            if not self.content_started:
                # First content in reasoning channel, add opening tag
                self.content_started = True
                return {"type": "reasoning", "content": f"<think>{content}"}
            else:
                return {"type": "reasoning", "content": content}
                
        elif self.current_channel == HarmonyChannel.FINAL:
            # Stream final content directly (not wrapped in think tags)
            self.final_content += content
            if not self.content_started:
                # First content in final channel
                self.content_started = True
                return {"type": "final", "content": content}
            else:
                return {"type": "final", "content": content}
                
        else:
            # Unknown channel or commentary, treat as raw
            return {"type": "raw", "content": content}
    
    def finalize(self) -> Dict[str, str]:
        """
        Finalize parsing and return any remaining content.
        Call this when the stream ends.
        """
        if self.buffer:
            output = self._process_channel_content(self.buffer)
            self.buffer = ""
        else:
            output = {"type": "raw", "content": ""}
            
        # Ensure reasoning tags are properly closed
        if self.reasoning_content and not self.final_content:
            # Only add closing tag if we haven't already closed it
            if output["type"] == "reasoning":
                output["content"] += "</think>"
            elif not output["content"]:
                output = {"type": "reasoning", "content": "</think>"}
                
        return output
    
    def get_final_result(self) -> Dict[str, str]:
        """Get the final parsed result for non-streaming use."""
        if not self.has_harmony_structure:
            # For unstructured content, return whatever was processed
            all_content = self.reasoning_content + self.final_content
            if not all_content and hasattr(self, '_raw_content'):
                return {"content": self._raw_content}
            return {"content": all_content}
            
        if self.reasoning_content and self.final_content:
            return {"content": f"<think>{self.reasoning_content}</think>{self.final_content}"}
        elif self.reasoning_content:
            return {"content": f"<think>{self.reasoning_content}</think>"}
        elif self.final_content:
            return {"content": self.final_content}
        else:
            return {"content": ""}


class HarmonyFormatter:
    """
    Formatter for OpenAI's Harmony response format used by GPT-OSS models.
    
    Harmony is a structured conversation protocol that enables:
    - Multiple role types (system, developer, user, assistant)
    - Channel-based outputs (analysis, commentary, final)
    - Built-in tool calling and reasoning support
    
    This class provides a clean interface to format messages using Harmony
    when the openai-harmony library is available, with graceful fallback
    when it's not installed.
    """
    
    def __init__(self):
        self._harmony_available = self._check_harmony_availability()
        
    def _check_harmony_availability(self) -> bool:
        """Check if openai-harmony is available."""
        try:
            import openai_harmony  # noqa: F401
            return True
        except ImportError:
            logging.debug("openai-harmony library not available")
            return False
    
    @property
    def is_available(self) -> bool:
        """Return True if Harmony formatting is available."""
        return self._harmony_available
    
    def format_messages(
        self, 
        messages: List[Dict[str, Any]], 
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """
        Format messages using OpenAI's Harmony format.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            tools: Optional list of tool definitions (for future tool calling support)
            
        Returns:
            Formatted string ready for GPT-OSS model input
            
        Raises:
            ImportError: If openai-harmony is not available
            ValueError: If message format is invalid
        """
        if not self._harmony_available:
            raise ImportError(
                "openai-harmony library is required for Harmony formatting. "
                "Install it with: pip install 'lemonade-sdk[harmony]'"
            )
        
        try:
            from openai_harmony import (
                HarmonyEncodingName,
                load_harmony_encoding,
                Conversation,
                Message,
                Role,
                SystemContent,
                DeveloperContent
            )
            
            # Load the Harmony encoding for GPT-OSS
            encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            
            # Convert messages to Harmony format
            harmony_messages = []
            
            for msg in messages:
                if not isinstance(msg, dict) or 'role' not in msg:
                    raise ValueError(f"Invalid message format: {msg}")
                
                role = msg['role']
                content = msg.get('content', '')
                
                # Map OpenAI chat roles to Harmony roles
                if role == 'system':
                    harmony_role = Role.SYSTEM
                    harmony_content = SystemContent.new()
                elif role == 'developer':
                    harmony_role = Role.DEVELOPER
                    harmony_content = DeveloperContent.new().with_instructions(content)
                elif role == 'user':
                    harmony_role = Role.USER
                    harmony_content = content
                elif role == 'assistant':
                    harmony_role = Role.ASSISTANT
                    harmony_content = content
                else:
                    # Default unknown roles to user
                    logging.warning(f"Unknown role '{role}', treating as user")
                    harmony_role = Role.USER
                    harmony_content = content
                
                harmony_messages.append(
                    Message.from_role_and_content(harmony_role, harmony_content)
                )
            
            # Create conversation from messages
            conversation = Conversation.from_messages(harmony_messages)
            
            # Render conversation for completion
            prefill_ids = encoding.render_conversation_for_completion(
                conversation, Role.ASSISTANT
            )
            
            # Return the token IDs - the caller will handle conversion to text
            # This allows the server to use its own tokenizer for decoding
            return prefill_ids
            
        except Exception as e:
            logging.error(f"Error formatting messages with Harmony: {e}")
            raise
    
    def get_stop_tokens(self) -> List[int]:
        """
        Get the appropriate stop token IDs for Harmony-formatted conversations.
        
        Returns:
            List of token IDs that should terminate generation
            
        Raises:
            ImportError: If openai-harmony is not available
        """
        if not self._harmony_available:
            raise ImportError("openai-harmony library is required")
        
        try:
            from openai_harmony import HarmonyEncodingName, load_harmony_encoding
            
            encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            return encoding.stop_tokens_for_assistant_action()
            
        except Exception as e:
            logging.error(f"Error getting Harmony stop tokens: {e}")
            raise


def is_gpt_oss_model(model_info: Dict[str, Any]) -> bool:
    """
    Determine if a model is a GPT-OSS model based on its metadata.
    
    Args:
        model_info: Model information dictionary from server_models.json
        
    Returns:
        True if the model is a GPT-OSS model that should use Harmony formatting
    """
    if not isinstance(model_info, dict):
        return False
    
    # Check for gpt-oss label
    labels = model_info.get('labels', [])
    if 'gpt-oss' in labels:
        return True
    
    # Fallback: check model name for gpt-oss pattern
    model_name = model_info.get('model_name', '')
    checkpoint = model_info.get('checkpoint', '')
    
    gpt_oss_patterns = ['gpt-oss', 'gpt_oss']
    
    for pattern in gpt_oss_patterns:
        if pattern in model_name.lower() or pattern in checkpoint.lower():
            return True
    
    return False


def parse_harmony_response(response_text: str, use_official_parser: bool = True) -> Dict[str, str]:
    """
    Parse Harmony response format and convert to Lemonade's thinking model format.
    
    This function implements both the official OpenAI approach (preferred) and a 
    fallback text-based approach for compatibility.
    
    Args:
        response_text: Raw response text in Harmony format
        use_official_parser: Whether to use the official openai-harmony parser
        
    Returns:
        Dictionary with 'content' and optionally 'reasoning' keys
    """
    
    # Try the official OpenAI Harmony parsing approach first
    if use_official_parser:
        try:
            from openai_harmony import (
                HarmonyEncodingName,
                load_harmony_encoding,
                Role
            )
            
            # Load the Harmony encoding
            encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
            
            # Note: The official approach expects token IDs, but we have text
            # For now, we'll use the text approach but log that we should improve this
            logging.debug("Using official Harmony library for parsing (text mode)")
            
            # Try to tokenize the response text back to IDs for proper parsing
            # This is a limitation - ideally we'd have the original token IDs
            # For now, fall through to text-based parsing
            
        except ImportError:
            logging.debug("Official openai-harmony library not available, using text parsing")
        except Exception as e:
            logging.debug(f"Official Harmony parsing failed: {e}, falling back to text parsing")
    
    # Text-based parsing (fallback or when official parsing unavailable)
    try:
        import re
        
        logging.debug("Using text-based Harmony parsing")
        
        # Extract analysis channel (reasoning) - improved regex patterns
        analysis_patterns = [
            r'<\|channel\|>analysis<\|message\|>(.*?)(?=<\|start\|>assistant|<\|channel\|>final|<\|channel\|>commentary|$)',
            r'<\|channel\|>analysis<\|message\|>(.*?)(?=<\|)',
        ]
        
        reasoning_content = ""
        for pattern in analysis_patterns:
            analysis_match = re.search(pattern, response_text, re.DOTALL)
            if analysis_match:
                reasoning_content = analysis_match.group(1).strip()
                break
        
        # Extract final channel (user-facing response) - improved regex patterns
        final_patterns = [
            r'<\|channel\|>final<\|message\|>(.*?)(?=<\|channel\|>|$)',
            r'<\|start\|>assistant<\|channel\|>final<\|message\|>(.*?)(?=<\|channel\|>|$)',
        ]
        
        final_content = ""
        for pattern in final_patterns:
            final_match = re.search(pattern, response_text, re.DOTALL)
            if final_match:
                final_content = final_match.group(1).strip()
                break
        
        # If no structured content found, return the original text
        if not reasoning_content and not final_content:
            logging.debug("No Harmony structure detected, returning original text")
            return {"content": response_text}
        
        # Build Lemonade thinking format
        result = {"content": final_content}
        
        if reasoning_content:
            # Convert to Lemonade's <think> format
            if final_content:
                result["content"] = f"<think>{reasoning_content}</think>{final_content}"
            else:
                result["content"] = f"<think>{reasoning_content}</think>"
            
            logging.info(f"🎯 Parsed Harmony response: reasoning={len(reasoning_content)} chars, final={len(final_content)} chars")
        
        return result
        
    except Exception as e:
        logging.error(f"Error parsing Harmony response: {e}")
        return {"content": response_text}


def should_use_harmony(
    model_info: Dict[str, Any], 
    backend: str, 
    harmony_formatter: HarmonyFormatter
) -> bool:
    """
    Determine if Harmony formatting should be used for a given model and backend.
    
    This function encapsulates the logic for when to use Harmony instead of
    the standard jinja templating.
    
    Args:
        model_info: Model information dictionary
        backend: Backend being used (e.g., 'vulkan', 'rocm')
        harmony_formatter: HarmonyFormatter instance
        
    Returns:
        True if Harmony should be used instead of jinja
    """
    # Must be a GPT-OSS model
    if not is_gpt_oss_model(model_info):
        return False
    
    # Harmony must be available
    if not harmony_formatter.is_available:
        logging.warning(
            "GPT-OSS model detected but openai-harmony not available. "
            "Install with: pip install 'lemonade-sdk[harmony]'"
        )
        return False
    
    # For now, use Harmony specifically for the Vulkan backend issue
    # This can be expanded later for other scenarios
    if backend == 'vulkan':
        model_name = model_info.get('model_name', '')
        if 'gpt-oss-120b' in model_name.lower():
            return True
    
    # Future: Could expand to use Harmony for all GPT-OSS models
    # regardless of backend for optimal performance
    
    return False

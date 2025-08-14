"""
Harmony template formatter for GPT-OSS models.

This module provides integration with OpenAI's Harmony response format,
which is specifically designed for GPT-OSS models to handle structured
conversations, reasoning, and tool calls.
"""

import logging
from typing import List, Dict, Any, Optional


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

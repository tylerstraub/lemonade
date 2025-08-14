# Harmony Support for GPT-OSS Models

Lemonade Server now includes support for OpenAI's Harmony response format, which provides improved compatibility and functionality for GPT-OSS models.

## Overview

**Harmony** is OpenAI's structured response format specifically designed for GPT-OSS models. It enables:

- **Structured Conversations**: Multiple role types (system, developer, user, assistant)
- **Channel-based Outputs**: Analysis (reasoning), commentary (tool outputs), final (user responses)
- **Built-in Tool Support**: Native support for function calling and reasoning
- **Better Performance**: Optimized format that GPT-OSS models were trained to use

## What This Solves

Previously, Lemonade had a workaround for the gpt-oss-120b model on the Vulkan backend, where the `--jinja` flag had to be disabled due to a llama.cpp compatibility issue. This resulted in:

- Loss of tool functionality
- Hacky add-then-remove flag logic
- Poor user experience

With Harmony support, Lemonade now:

- ✅ Uses the native format GPT-OSS models were trained for
- ✅ Maintains tool functionality through Harmony's built-in support
- ✅ Provides cleaner, more maintainable code
- ✅ Offers better performance for GPT-OSS models

## Installation

To enable Harmony support, install the optional dependency:

```bash
pip install 'lemonade-sdk[harmony]'
```

Or install directly:

```bash
pip install openai-harmony
```

## How It Works

### Automatic Detection

Lemonade automatically detects GPT-OSS models based on:

1. **Model Labels**: Models with the `gpt-oss` label in `server_models.json`
2. **Name Patterns**: Models with `gpt-oss` or `gpt_oss` in their name or checkpoint

### When Harmony is Used

Harmony formatting is automatically used when:

- The model is detected as a GPT-OSS model
- The `openai-harmony` library is installed
- Specific backend conditions are met (currently Vulkan backend for gpt-oss-120b)

### Integration Points

1. **Message Processing**: `apply_chat_template()` method checks for GPT-OSS models
2. **Flag Management**: `_launch_llama_subprocess()` conditionally omits `--jinja` flag
3. **Template Formatting**: Harmony formatter converts OpenAI chat format to Harmony format

## Supported Models

Currently configured GPT-OSS models:

- `gpt-oss-120b-GGUF`: 120B parameter model with reasoning capabilities
- `gpt-oss-20b-GGUF`: 20B parameter model with reasoning capabilities

## Technical Details

### Architecture

```
Chat Request → GPT-OSS Detection → Harmony Formatter → Model Processing
                     ↓
               [Standard Models] → Traditional Jinja Templates
```

### Code Structure

- `src/lemonade/tools/server/harmony.py`: Harmony formatting logic
- `src/lemonade/tools/server/serve.py`: Server integration
- `src/lemonade/tools/server/llamacpp.py`: Flag management
- `src/lemonade_server/server_models.json`: Model metadata

### Message Flow

1. **Chat Completion Request** received
2. **Model Detection** checks if GPT-OSS model
3. **Harmony Check** verifies if Harmony should be used
4. **Format Selection**:
   - GPT-OSS + Harmony available → Use Harmony formatter
   - Otherwise → Use standard jinja templates
5. **Response Generation** with appropriate formatting

## Configuration

### Model Labels

Models are identified as GPT-OSS through labels in `server_models.json`:

```json
{
  "gpt-oss-120b-GGUF": {
    "checkpoint": "unsloth/gpt-oss-120b-GGUF:Q4_K_M",
    "recipe": "llamacpp",
    "suggested": true,
    "labels": ["hot", "reasoning", "gpt-oss"]
  }
}
```

### Backend Configuration

Harmony usage can be configured per backend in the `should_use_harmony()` function:

```python
def should_use_harmony(model_info, backend, harmony_formatter):
    # Current logic: Use Harmony for gpt-oss-120b on Vulkan
    if backend == 'vulkan' and 'gpt-oss-120b' in model_info.get('model_name', ''):
        return True
    # Future: Could expand to all GPT-OSS models on all backends
    return False
```

## Troubleshooting

### Harmony Not Available

If you see this warning:
```
GPT-OSS model detected but openai-harmony not available.
Install with: pip install 'lemonade-sdk[harmony]'
```

**Solution**: Install the harmony dependency as shown above.

### Fallback Behavior

When Harmony is not available, Lemonade falls back to:

1. **Standard Templates**: Uses the model's built-in chat template
2. **Legacy Workaround**: For gpt-oss-120b on Vulkan, disables jinja flag
3. **Default Template**: If no other template is available

### Debugging

Enable debug logging to see Harmony decisions:

```bash
lemonade-server-dev serve --log-level debug
```

Look for messages like:
- `"Using Harmony formatting for gpt-oss-120b-GGUF on vulkan backend"`
- `"Harmony formatting not available: ..."`
- `"Error with Harmony formatting: ..."`

## Future Enhancements

Potential improvements include:

1. **Universal GPT-OSS Support**: Use Harmony for all GPT-OSS models regardless of backend
2. **Tool Call Integration**: Enhanced tool calling through Harmony's native support
3. **Reasoning Channels**: Better handling of reasoning vs. final response channels
4. **Custom Role Support**: Support for developer role and custom instructions

## Examples

### Before (Legacy Workaround)

```python
# Hacky approach in llamacpp.py
base_command.extend(["--port", str(port), "--jinja"])
if backend == "vulkan" and "gpt-oss-120b" in model_path:
    base_command.remove("--jinja")  # Ugly!
```

### After (Clean Harmony Integration)

```python
# Clean approach with Harmony
if should_use_harmony(model_info, backend, harmony_formatter):
    # Use Harmony formatting in apply_chat_template
    formatted_text = harmony_formatter.format_messages(messages)
    # Don't add --jinja flag
else:
    base_command.append("--jinja")
```

## References

- [OpenAI Harmony Documentation](https://cookbook.openai.com/articles/openai-harmony)
- [GPT-OSS Models on Hugging Face](https://huggingface.co/openai)
- [llama.cpp Issue #15274](https://github.com/ggml-org/llama.cpp/issues/15274)

"""
Test script to verify Harmony is working in a running server.

This script makes a simple chat completion request to test if Harmony
formatting is being used for GPT-OSS models.
"""

import requests
import json
import sys

def test_chat_completion():
    """Test chat completion with a simple message."""
    print("Testing Harmony formatting with gpt-oss-120b...")
    
    # Server endpoint
    url = "http://localhost:8000/api/v1/chat/completions"
    
    # Simple test message
    payload = {
        "model": "gpt-oss-120b-GGUF",
        "messages": [
            {"role": "user", "content": "Hello, can you help me with a simple math problem? What is 2+2?"}
        ],
        "max_tokens": 50,
        "stream": False
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        print(f"Making request to {url}")
        print(f"Payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(url, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Success! Response:")
            print(f"Model: {result.get('model', 'unknown')}")
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0].get('message', {}).get('content', '')
                print(f"Response: {content}")
            print(f"\nFull response: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ Error: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection error. Is the server running on localhost:8000?")
        print("Start the server with: lemonade-server-dev serve")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

def check_server_status():
    """Check if the server is running."""
    try:
        response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is running")
            return True
        else:
            print(f"❌ Server responded with status {response.status_code}")
            return False
    except:
        print("❌ Server is not responding")
        return False

def main():
    print("Harmony Integration Test")
    print("=" * 40)
    
    # Check server status first
    if not check_server_status():
        print("\nPlease start the server first:")
        print("cd C:\\Users\\ECHO\\Projects\\lemonade-dev-env")
        print(".\\activate-dev-env.ps1")
        print("lemonade-server-dev serve")
        return 1
    
    # Test chat completion
    print("\nTesting chat completion...")
    if test_chat_completion():
        print("\n🎉 Test completed successfully!")
        print("\nTo see if Harmony was used, check the server logs for:")
        print("  '🎯 Harmony Mode: Omitting --jinja flag...'")
        print("  '🎯 Using Harmony formatting...'")
    else:
        print("\n❌ Test failed")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

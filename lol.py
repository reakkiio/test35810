"""
Python client example for the OpenAI-compatible API.

This script demonstrates how to use the OpenAI-compatible API with Python,
both with the OpenAI Python library and with direct requests.
"""

import requests
import json
import os
import sys

# Example 1: Using the requests library directly
def use_requests_library(api_url, api_key=None):
    """Use the requests library to call the API directly."""
    print("\n=== Example 1: Using requests library ===")
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # Add API key if provided
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    # Prepare the request payload
    payload = {
        "model": "gpt-4",  # Use your model name
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, tell me a short joke."}
        ],
        "temperature": 0.7,
        "max_tokens": 150
    }
    
    # Make the API call
    try:
        response = requests.post(
            f"{api_url}/chat/completions",
            headers=headers,
            json=payload
        )
        
        # Check if the request was successful
        if response.status_code == 200:
            result = response.json()
            print("API Response:")
            print(f"Status Code: {response.status_code}")
            
            # Extract and print the assistant's message
            assistant_message = result["choices"][0]["message"]["content"]
            print(f"\nAssistant: {assistant_message}")
            
            # Print token usage if available
            if "usage" in result:
                print(f"\nToken Usage:")
                print(f"  Prompt tokens: {result['usage']['prompt_tokens']}")
                print(f"  Completion tokens: {result['usage']['completion_tokens']}")
                print(f"  Total tokens: {result['usage']['total_tokens']}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"Exception occurred: {e}")

# Example 2: Using the OpenAI Python library
def use_openai_library(api_url, api_key=None):
    """Use the OpenAI Python library to call the API."""
    print("\n=== Example 2: Using OpenAI Python library ===")
    
    try:
        # Check if the OpenAI library is installed
        import openai
    except ImportError:
        print("OpenAI Python library not installed. Install it with:")
        print("pip install openai")
        return
    
    # Configure the client
    client = openai.OpenAI(
        base_url=api_url,
        api_key=api_key or "dummy-api-key"  # API key is required by the library
    )
    
    try:
        # Make the API call
        response = client.chat.completions.create(
            model="gpt-4",  # Use your model name
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What's the capital of France?"}
            ],
            temperature=0.7,
            max_tokens=150
        )
        
        # Extract and print the assistant's message
        assistant_message = response.choices[0].message.content
        print(f"\nAssistant: {assistant_message}")
        
        # Print token usage if available
        if hasattr(response, 'usage'):
            print(f"\nToken Usage:")
            print(f"  Prompt tokens: {response.usage.prompt_tokens}")
            print(f"  Completion tokens: {response.usage.completion_tokens}")
            print(f"  Total tokens: {response.usage.total_tokens}")
    
    except Exception as e:
        print(f"Exception occurred: {e}")

# Example 3: Streaming response
def use_streaming(api_url, api_key=None):
    """Use streaming to get the response token by token."""
    print("\n=== Example 3: Using streaming ===")
    
    headers = {
        "Content-Type": "application/json"
    }
    
    # Add API key if provided
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    # Prepare the request payload with streaming enabled
    payload = {
        "model": "gpt-4",  # Use your model name
        "messages": [
            {"role": "user", "content": "Write a short poem about programming."}
        ],
        "temperature": 0.7,
        "max_tokens": 150,
        "stream": True  # Enable streaming
    }
    
    # Make the API call with streaming
    try:
        response = requests.post(
            f"{api_url}/chat/completions",
            headers=headers,
            json=payload,
            stream=True  # Enable streaming in requests
        )
        
        # Check if the request was successful
        if response.status_code == 200:
            print("Streaming response:")
            print("-------------------")
            
            # Process the streaming response
            for line in response.iter_lines():
                if line:
                    # Skip empty lines
                    line = line.decode('utf-8')
                    
                    # Skip the "data: " prefix
                    if line.startswith('data: '):
                        line = line[6:]
                    
                    # Check for the end of the stream
                    if line == "[DONE]":
                        break
                    
                    try:
                        # Parse the JSON data
                        data = json.loads(line)
                        
                        # Extract and print the content delta if available
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            if "content" in delta:
                                print(delta["content"], end="", flush=True)
                    except json.JSONDecodeError:
                        # Skip lines that aren't valid JSON
                        continue
            
            print("\n-------------------")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    
    except Exception as e:
        print(f"Exception occurred: {e}")

if __name__ == "__main__":
    # Default API URL (change this to your API server URL)
    api_url = "https://ai4free-test.hf.space/v1"
    
    # Get API URL from command line argument if provided
    if len(sys.argv) > 1:
        api_url = sys.argv[1]
    
    # Get API key from environment variable or use None
    api_key = os.environ.get("OPENAI_API_KEY")
    
    print(f"Using API URL: {api_url}")
    print(f"API Key: {'Provided' if api_key else 'Not provided'}")
    
    # Run the examples
    use_requests_library(api_url, api_key)
    use_openai_library(api_url, api_key)
    use_streaming(api_url, api_key)

from webscout.Provider.OPENAI import ExaChat

# Initialize the client
client = ExaChat(timeout=60)

# Create a streaming completion
stream = client.chat.completions.create(
    model="gemini-2.0-pro-exp-02-05",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "Can you tell me more"}


    ],
    stream=True
)

# Process the streaming response
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()  # Add a newline at the end
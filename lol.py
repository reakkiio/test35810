from webscout.Provider.OPENAI import LLMChatCo

# Initialize the client
client = LLMChatCo()

# Create a streaming completion
stream = client.chat.completions.create(
    model="gemini-flash-2.0",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a short poem about programming."}
    ],
    stream=True
)

# Process the streaming response
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()  # Add a newline at the end
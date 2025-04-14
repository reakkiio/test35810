from webscout.Provider.OPENAI import HeckAI

# Initialize the client
client = HeckAI()

# Create a streaming completion
stream = client.chat.completions.create(
    model="google/gemini-2.0-flash-001",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "hi"}
    ],
    stream=True
)

# Process the streaming response
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()  # Add a newline at the end
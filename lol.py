from webscout.Provider.OPENAI import TextPollinations

# Initialize the client
client = TextPollinations()

# Create a completion (non-streaming)
response = client.chat.completions.create(
    model="openai-large",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a short joke."}
    ]
)

# Print the response
print(response.choices[0].message.content)

# Create a streaming completion
stream = client.chat.completions.create(
    model="openai-large",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Count from 1 to 5."}
    ],
    stream=True
)

# Process the streaming response
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()  # Add a newline at the end
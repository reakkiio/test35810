from webscout.Provider.OPENAI import YEPCHAT # Changed import

# Initialize the client
client = YEPCHAT() # Changed client

# Create a streaming completion
stream = client.chat.completions.create(
    model="Mixtral-8x7B-Instruct-v0.1", # Changed model to one available in YEPCHAT
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
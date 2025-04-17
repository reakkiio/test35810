from webscout.Provider.OPENAI import ChatGPT

client = ChatGPT(timeout=60)  # No browser parameter needed anymore

# Create a streaming completion
stream = client.chat.completions.create(
    model="auto",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    stream=True
)

for chunks in stream:
    if chunks.choices[0].delta.content is not None:
        print(chunks.choices[0].delta.content, end="", flush=True)
print()  # Add a newline at the end
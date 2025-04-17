from webscout.Provider.OPENAI import OPKFC

client = OPKFC(timeout=60)  # No browser parameter needed anymore

# Create a streaming completion
stream = client.chat.completions.create(
    model="o4-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    stream=True
)

for chunks in stream:
    print(chunks.choices[0].delta.content, end="", flush=True)
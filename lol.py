from webscout.Provider.OPENAI import WiseCat

# Initialize the client
client = WiseCat()

# Create a streaming completion
stream = client.chat.completions.create(
    model="chat-model-small",  # or use "gpt-4"
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a short poem about programming."}
    ],
    stream=True
)

print()  # Add a newline at the end
from rich import print
# for chunk in stream:
#     print(chunk['choices'][0]['delta'], end='', flush=True)
for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()  # Add a newline at the end 
from webscout.Provider.OPENAI import MultiChatAI

# Initialize the client
client = MultiChatAI()

# Create a completion
response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about Python programming."}
    ],
    temperature=0.7,
    max_tokens=500
)

# Get the response content
print(response.choices[0].message.content)
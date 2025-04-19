from webscout.Provider.OPENAI import E2B
# Initialize the client
client = E2B()

# Create a completion (non-streaming)
response = client.chat.completions.create(
    model="claude-3.5-sonnet",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me about Python programming."}
    ],
    temperature=0.7,
    max_tokens=500
)

# Print the response
print(response.choices[0].message.content)
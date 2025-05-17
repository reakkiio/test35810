from webscout.Provider.OPENAI import NEMOTRON
from rich import print
client = NEMOTRON()
print("\n[bold yellow]Available models:[/bold yellow]")
print(client.models.list())
response = client.chat.completions.create(
    model="NEMOTRON/nemotron70b",
    messages=[{"role": "user", "content": "tell me about india"}],
    stream=False,
)
print(response)
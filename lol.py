from webscout.Provider.OPENAI import BLACKBOXAI
from rich import print
client = BLACKBOXAI()
print("\n[bold yellow]Available models:[/bold yellow]")
print(client.models.list())
response = client.chat.completions.create(
    model="GPT-4.1",
    messages=[{"role": "user", "content": "tell me about india"}],
    stream=False,
)
print(response)
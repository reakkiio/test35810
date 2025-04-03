# Define a tool
from webscout.Provider.yep import YEPCHAT
from webscout.conversation import Fn


weather_tool = Fn(
    name="get_weather", 
    description="Get the current weather", 
    parameters={"location": "string", "unit": "string"}
)

# Initialize YEPCHAT with the tool
ai = YEPCHAT(model="Mixtral-8x7B-Instruct-v0.1", tools=[weather_tool])

resp = ai.chat(input("> "), stream=True)
for chunks in resp:
    print(chunks, end="", flush=True)
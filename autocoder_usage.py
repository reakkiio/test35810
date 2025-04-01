from webscout import autocoder_utiles
from webscout import autocoder
from webscout import C4ai # u can use any webscout or any non webscout provider

use_thinking = True

ai = C4ai(model="command-a-03-2025", timeout=5000, system_prompt=autocoder_utiles.get_intro_prompt())
agent = autocoder.AutoCoder(ai_instance=ai)
ai_thinking = C4ai(model="command-a-03-2025", timeout=5000, system_prompt=autocoder_utiles.get_thinking_intro())
while True:
    if use_thinking:
        print("Thinking...")
        response = ai_thinking.chat(input(">> "))
        print(response)
        response_code = ai.chat("Thought: \n"+str(response)+"\n""Generate the code accordingly above is a brief thought process")
        resp = agent.main(response=response_code)
        if resp:
            print(resp)
    else:
        response = ai.chat(input("> "))
        resp = agent.main(response=response)
        if resp:
            print(resp)

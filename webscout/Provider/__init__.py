# webscout/providers/__init__.py
from .PI import *
from .Cohere import Cohere
from .Reka import REKA
from .Groq import GROQ
from .Groq import AsyncGROQ
from .Openai import OPENAI
from .Openai import AsyncOPENAI
from .Koboldai import KOBOLDAI
from .Koboldai import AsyncKOBOLDAI
from .Blackboxai import BLACKBOXAI
from .ai4chat import *
from .Gemini import GEMINI
from .Deepinfra import DeepInfra
from .typefully import *
from .cleeai import *
from .OLLAMA import OLLAMA
from .Andi import AndiSearch
from .Llama3 import *
from .koala import *
from .meta import *
from .julius import *
from .yep import *
from .Cloudflare import *
from .turboseek import *
from .TeachAnything import *
from .AI21 import *
from .x0gpt import *
from .cerebras import *
from .geminiapi import *
from .elmo import *
from .Netwrck import Netwrck
from .llmchat import *
from .llmchatco import LLMChatCo  # Add new LLMChat.co provider
from .talkai import *
from .llama3mitril import *
from .Marcus import *
from .typegpt import *
from .multichat import *
from .Jadve import *
from .chatglm import *
from .hermes import *
from .TextPollinationsAI import *
from .Glider import *
from .ChatGPTGratis import *
from .QwenLM import *
from .granite import *
from .WiseCat import *
from .freeaichat import FreeAIChat
from .akashgpt import *
from .Perplexitylabs import *
from .AllenAI import *
from .HeckAI import *
from .TwoAI import *
from .Venice import *
from .HuggingFaceChat import *
from .GithubChat import *
from .copilot import *
from .sonus import *
from .uncovr import *
from .LambdaChat import *
from .ChatGPTClone import *
from .VercelAI import *
from .ExaChat import *
from .asksteve import *
from .Aitopia import *
from .searchchat import *
from .ExaAI import ExaAI
from .OpenGPT import OpenGPT
from .scira_chat import *
from .StandardInput import *
from .Writecream import Writecream
from .toolbaz import Toolbaz
from .scnet import SCNet
from .WritingMate import WritingMate
from .MCPCore import MCPCore
from .TypliAI import TypliAI
from .ChatSandbox import ChatSandbox
from .GizAI import GizAI
from .WrDoChat import WrDoChat
from .Nemotron import NEMOTRON
from .FreeGemini import FreeGemini
from .Flowith import Flowith
from .samurai import samurai
__all__ = [
    'SCNet',
    'NEMOTRON',
    'Flowith',
    'samurai',
    'FreeGemini',
    'WrDoChat',
    'GizAI',
    'ChatSandbox',
    'SciraAI',
    'StandardInputAI',
    'OpenGPT',
    'Venice',
    'ExaAI',
    'Copilot',
    'HuggingFaceChat',
    'TwoAI',
    'HeckAI',
    'AllenAI',
    'PerplexityLabs',
    'AkashGPT',
    'WritingMate',
    'WiseCat',
    'IBMGranite',
    'QwenLM',
    'ChatGPTGratis',
    'LambdaChat',
    'TextPollinationsAI',
    'GliderAI',
    'Cohere',
    'REKA',
    'GROQ',
    'AsyncGROQ',
    'OPENAI',
    'AsyncOPENAI',
    'KOBOLDAI',
    'AsyncKOBOLDAI',
    'BLACKBOXAI',
    'GEMINI',
    'DeepInfra',
    'AI4Chat',
    'OLLAMA',
    'AndiSearch',
    'Sambanova',
    'KOALA',
    'Meta',
    'PiAI',
    'Julius',
    'YEPCHAT',
    'Cloudflare',
    'TurboSeek',
    'TeachAnything',
    'AI21',
    'X0GPT',
    'Cerebras',
    'GEMINIAPI',
    'SonusAI',
    'Cleeai',
    'Elmo',
    'ChatGPTClone',
    'TypefullyAI',
    'Netwrck',
    'LLMChat',
    'LLMChatCo',
    'Talkai',
    'Llama3Mitril',
    'Marcus',
    'TypeGPT',
    'Netwrck',
    'MultiChatAI',
    'JadveOpenAI',
    'ChatGLM',
    'NousHermes',
    'FreeAIChat',
    'GithubChat',
    'UncovrAI',
    'VercelAI',
    'ExaChat',
    'AskSteve',
    'Aitopia',
    'SearchChatAI',
    'Writecream',
    'Toolbaz',
    'MCPCore',
    'TypliAI',
]

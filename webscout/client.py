"""
Unified client import for OpenAI-compatible providers.

Usage:
    from webscout.client import FreeAIChat, AI4Chat, ExaChat, MultiChatAI, TwoAI, SciraChat, ChatSandbox, C4AI, Groq, TypeGPT, LLMChat, Cleeai, DeepInfra, BlackboxAI, Cloudflare, Netwrck, OIVSCode, Venice, Writecream, Wisecat, Yep, X0GPT, Sonus, Toolbaz, TextPollinations, StandardInput, Opkfc, Flowith, ExaAI, FreeGemini, GeminiAPI, Gemini, GithubChat, GizAI, Glider, HeckAI, HuggingFaceChat, Hunyuan, Jadve, Julius, Koala, LearnFast, LearnFastAI, NEMOTRON, MCPCore, PydanticImports, TeachAnything, UncovrAI, API, Base
    client = FreeAIChat()
    response = client.chat.completions.create(...)
"""

from webscout.Provider.OPENAI import *

import requests
import random
import string
import base64
from datetime import datetime, timedelta
from typing import Any, Dict, Union, Generator, List
from webscout.AIutel import Optimizers, Conversation, AwesomePrompts
from webscout.AIbase import Provider
from webscout import exceptions
from webscout.litagent import LitAgent
def to_data_uri(image_data):
    """Convert image data to a data URI format"""
    if isinstance(image_data, str):
        # Assume it's already a data URI
        return image_data

    # Encode binary data to base64
    encoded = base64.b64encode(image_data).decode('utf-8')

    # Determine MIME type (simplified)
    mime_type = "image/jpeg"  # Default
    if image_data.startswith(b'\x89PNG'):
        mime_type = "image/png"
    elif image_data.startswith(b'\xff\xd8'):
        mime_type = "image/jpeg"
    elif image_data.startswith(b'GIF'):
        mime_type = "image/gif"

    return f"data:{mime_type};base64,{encoded}"


class BLACKBOXAI(Provider):
    """
    BlackboxAI provider for interacting with the Blackbox API.
    Supports synchronous operations with multiple models.
    """
    url = "https://www.blackbox.ai"
    api_endpoint = "https://www.blackbox.ai/api/chat"


    # Default model (remains the same as per original class)
    default_model = "GPT-4.1"
    default_vision_model = default_model

    # New OpenRouter models list
    openrouter_models = [
        "Deepcoder 14B Preview",
        "DeepHermes 3 Llama 3 8B Preview",
        "DeepSeek R1 Zero",
        "Dolphin3.0 Mistral 24B",
        "Dolphin3.0 R1 Mistral 24B",
        "Flash 3",
        "Gemini 2.0 Flash Experimental",
        "Gemma 2 9B",
        "Gemma 3 12B",
        "Gemma 3 1B",
        "Gemma 3 27B",
        "Gemma 3 4B",
        "Kimi VL A3B Thinking",
        "Llama 3.1 8B Instruct",
        "Llama 3.1 Nemotron Ultra 253B v1",
        "Llama 3.2 11B Vision Instruct",
        "Llama 3.2 1B Instruct",
        "Llama 3.2 3B Instruct",
        "Llama 3.3 70B Instruct",
        "Llama 3.3 Nemotron Super 49B v1",
        "Llama 4 Maverick",
        "Llama 4 Scout",
        "Mistral 7B Instruct",
        "Mistral Nemo",
        "Mistral Small 3",
        "Mistral Small 3.1 24B",
        "Molmo 7B D",
        "Moonlight 16B A3B Instruct",
        "Qwen2.5 72B Instruct",
        "Qwen2.5 7B Instruct",
        "Qwen2.5 Coder 32B Instruct",
        "Qwen2.5 VL 32B Instruct",
        "Qwen2.5 VL 3B Instruct",
        "Qwen2.5 VL 72B Instruct",
        "Qwen2.5-VL 7B Instruct",
        "Qwerky 72B",
        "QwQ 32B",
        "QwQ 32B Preview",
        "QwQ 32B RpR v1",
        "R1",
        "R1 Distill Llama 70B",
        "R1 Distill Qwen 14B",
        "R1 Distill Qwen 32B",
    ]

    # New base models list
    models = [
        default_model,
        "o3-mini",
        "gpt-4.1-nano",
        "Claude-sonnet-3.7",
        "Claude-sonnet-3.5",
        "DeepSeek-R1",
        "Mistral-Small-24B-Instruct-2501",
        *openrouter_models,
        # Trending agent modes (names)
        'Python Agent', 'HTML Agent', 'Builder Agent', 'Java Agent', 'JavaScript Agent',
        'React Agent', 'Android Agent', 'Flutter Agent', 'Next.js Agent', 'AngularJS Agent',
        'Swift Agent', 'MongoDB Agent', 'PyTorch Agent', 'Xcode Agent', 'Azure Agent',
        'Bitbucket Agent', 'DigitalOcean Agent', 'Docker Agent', 'Electron Agent',
        'Erlang Agent', 'FastAPI Agent', 'Firebase Agent', 'Flask Agent', 'Git Agent',
        'Gitlab Agent', 'Go Agent', 'Godot Agent', 'Google Cloud Agent', 'Heroku Agent'
    ]

    # Models that support vision capabilities
    vision_models = [default_vision_model, 'o3-mini', "Llama 3.2 11B Vision Instruct"] # Added Llama vision

    # Models that can be directly selected by users
    userSelectedModel = ['o3-mini','Claude-sonnet-3.7', 'Claude-sonnet-3.5', 'DeepSeek-R1', 'Mistral-Small-24B-Instruct-2501'] + openrouter_models

    # Agent mode configurations
    agentMode = {
        # OpenRouter Free
        'Deepcoder 14B Preview': {'mode': True, 'id': "agentica-org/deepcoder-14b-preview:free", 'name': "Deepcoder 14B Preview"},
        'DeepHermes 3 Llama 3 8B Preview': {'mode': True, 'id': "nousresearch/deephermes-3-llama-3-8b-preview:free", 'name': "DeepHermes 3 Llama 3 8B Preview"},
        'DeepSeek R1 Zero': {'mode': True, 'id': "deepseek/deepseek-r1-zero:free", 'name': "DeepSeek R1 Zero"},
        'Dolphin3.0 Mistral 24B': {'mode': True, 'id': "cognitivecomputations/dolphin3.0-mistral-24b:free", 'name': "Dolphin3.0 Mistral 24B"},
        'Dolphin3.0 R1 Mistral 24B': {'mode': True, 'id': "cognitivecomputations/dolphin3.0-r1-mistral-24b:free", 'name': "Dolphin3.0 R1 Mistral 24B"},
        'Flash 3': {'mode': True, 'id': "rekaai/reka-flash-3:free", 'name': "Flash 3"},
        'Gemini 2.0 Flash Experimental': {'mode': True, 'id': "google/gemini-2.0-flash-exp:free", 'name': "Gemini 2.0 Flash Experimental"},
        'Gemma 2 9B': {'mode': True, 'id': "google/gemma-2-9b-it:free", 'name': "Gemma 2 9B"},
        'Gemma 3 12B': {'mode': True, 'id': "google/gemma-3-12b-it:free", 'name': "Gemma 3 12B"},
        'Gemma 3 1B': {'mode': True, 'id': "google/gemma-3-1b-it:free", 'name': "Gemma 3 1B"},
        'Gemma 3 27B': {'mode': True, 'id': "google/gemma-3-27b-it:free", 'name': "Gemma 3 27B"},
        'Gemma 3 4B': {'mode': True, 'id': "google/gemma-3-4b-it:free", 'name': "Gemma 3 4B"},
        'Kimi VL A3B Thinking': {'mode': True, 'id': "moonshotai/kimi-vl-a3b-thinking:free", 'name': "Kimi VL A3B Thinking"},
        'Llama 3.1 8B Instruct': {'mode': True, 'id': "meta-llama/llama-3.1-8b-instruct:free", 'name': "Llama 3.1 8B Instruct"},
        'Llama 3.1 Nemotron Ultra 253B v1': {'mode': True, 'id': "nvidia/llama-3.1-nemotron-ultra-253b-v1:free", 'name': "Llama 3.1 Nemotron Ultra 253B v1"},
        'Llama 3.2 11B Vision Instruct': {'mode': True, 'id': "meta-llama/llama-3.2-11b-vision-instruct:free", 'name': "Llama 3.2 11B Vision Instruct"},
        'Llama 3.2 1B Instruct': {'mode': True, 'id': "meta-llama/llama-3.2-1b-instruct:free", 'name': "Llama 3.2 1B Instruct"},
        'Llama 3.2 3B Instruct': {'mode': True, 'id': "meta-llama/llama-3.2-3b-instruct:free", 'name': "Llama 3.2 3B Instruct"},
        'Llama 3.3 70B Instruct': {'mode': True, 'id': "meta-llama/llama-3.3-70b-instruct:free", 'name': "Llama 3.3 70B Instruct"},
        'Llama 3.3 Nemotron Super 49B v1': {'mode': True, 'id': "nvidia/llama-3.3-nemotron-super-49b-v1:free", 'name': "Llama 3.3 Nemotron Super 49B v1"},
        'Llama 4 Maverick': {'mode': True, 'id': "meta-llama/llama-4-maverick:free", 'name': "Llama 4 Maverick"},
        'Llama 4 Scout': {'mode': True, 'id': "meta-llama/llama-4-scout:free", 'name': "Llama 4 Scout"},
        'Mistral 7B Instruct': {'mode': True, 'id': "mistralai/mistral-7b-instruct:free", 'name': "Mistral 7B Instruct"},
        'Mistral Nemo': {'mode': True, 'id': "mistralai/mistral-nemo:free", 'name': "Mistral Nemo"},
        'Mistral Small 3': {'mode': True, 'id': "mistralai/mistral-small-24b-instruct-2501:free", 'name': "Mistral Small 3"}, # Matches Mistral-Small-24B-Instruct-2501
        'Mistral Small 3.1 24B': {'mode': True, 'id': "mistralai/mistral-small-3.1-24b-instruct:free", 'name': "Mistral Small 3.1 24B"},
        'Molmo 7B D': {'mode': True, 'id': "allenai/molmo-7b-d:free", 'name': "Molmo 7B D"},
        'Moonlight 16B A3B Instruct': {'mode': True, 'id': "moonshotai/moonlight-16b-a3b-instruct:free", 'name': "Moonlight 16B A3B Instruct"},
        'Qwen2.5 72B Instruct': {'mode': True, 'id': "qwen/qwen-2.5-72b-instruct:free", 'name': "Qwen2.5 72B Instruct"},
        'Qwen2.5 7B Instruct': {'mode': True, 'id': "qwen/qwen-2.5-7b-instruct:free", 'name': "Qwen2.5 7B Instruct"},
        'Qwen2.5 Coder 32B Instruct': {'mode': True, 'id': "qwen/qwen-2.5-coder-32b-instruct:free", 'name': "Qwen2.5 Coder 32B Instruct"},
        'Qwen2.5 VL 32B Instruct': {'mode': True, 'id': "qwen/qwen2.5-vl-32b-instruct:free", 'name': "Qwen2.5 VL 32B Instruct"},
        'Qwen2.5 VL 3B Instruct': {'mode': True, 'id': "qwen/qwen2.5-vl-3b-instruct:free", 'name': "Qwen2.5 VL 3B Instruct"},
        'Qwen2.5 VL 72B Instruct': {'mode': True, 'id': "qwen/qwen2.5-vl-72b-instruct:free", 'name': "Qwen2.5 VL 72B Instruct"},
        'Qwen2.5-VL 7B Instruct': {'mode': True, 'id': "qwen/qwen-2.5-vl-7b-instruct:free", 'name': "Qwen2.5-VL 7B Instruct"},
        'Qwerky 72B': {'mode': True, 'id': "featherless/qwerky-72b:free", 'name': "Qwerky 72B"},
        'QwQ 32B': {'mode': True, 'id': "qwen/qwq-32b:free", 'name': "QwQ 32B"},
        'QwQ 32B Preview': {'mode': True, 'id': "qwen/qwq-32b-preview:free", 'name': "QwQ 32B Preview"},
        'QwQ 32B RpR v1': {'mode': True, 'id': "arliai/qwq-32b-arliai-rpr-v1:free", 'name': "QwQ 32B RpR v1"},
        'R1': {'mode': True, 'id': "deepseek/deepseek-r1:free", 'name': "R1"}, # Matches DeepSeek-R1
        'R1 Distill Llama 70B': {'mode': True, 'id': "deepseek/deepseek-r1-distill-llama-70b:free", 'name': "R1 Distill Llama 70B"},
        'R1 Distill Qwen 14B': {'mode': True, 'id': "deepseek/deepseek-r1-distill-qwen-14b:free", 'name': "R1 Distill Qwen 14B"},
        'R1 Distill Qwen 32B': {'mode': True, 'id': "deepseek/deepseek-r1-distill-qwen-32b:free", 'name': "R1 Distill Qwen 32B"},
        # Default models from the new list
        'Claude-sonnet-3.7': {'mode': True, 'id': "Claude-sonnet-3.7", 'name': "Claude-sonnet-3.7"},
        'Claude-sonnet-3.5': {'mode': True, 'id': "Claude-sonnet-3.5", 'name': "Claude-sonnet-3.5"},
        'DeepSeek-R1': {'mode': True, 'id': "deepseek-reasoner", 'name': "DeepSeek-R1"}, # This is 'R1' in openrouter, but 'DeepSeek-R1' in base models
        'Mistral-Small-24B-Instruct-2501': {'mode': True, 'id': "mistralai/Mistral-Small-24B-Instruct-2501", 'name': "Mistral-Small-24B-Instruct-2501"},
        # Add default_model if it's not covered and has an agent mode
        default_model: {'mode': True, 'id': "openai/gpt-4.1", 'name': default_model}, # Assuming GPT-4.1 is agent-compatible
        'o3-mini': {'mode': True, 'id': "o3-mini", 'name': "o3-mini"}, # Assuming o3-mini is agent-compatible
        'gpt-4.1-nano': {'mode': True, 'id': "gpt-4.1-nano", 'name': "gpt-4.1-nano"}, # Assuming gpt-4.1-nano is agent-compatible
    }

    # Trending agent modes
    trendingAgentMode = {
        'Python Agent': {'mode': True, 'id': "python"},
        'HTML Agent': {'mode': True, 'id': "html"},
        'Builder Agent': {'mode': True, 'id': "builder"},
        'Java Agent': {'mode': True, 'id': "java"},
        'JavaScript Agent': {'mode': True, 'id': "javascript"},
        'React Agent': {'mode': True, 'id': "react"},
        'Android Agent': {'mode': True, 'id': "android"},
        'Flutter Agent': {'mode': True, 'id': "flutter"},
        'Next.js Agent': {'mode': True, 'id': "next.js"},
        'AngularJS Agent': {'mode': True, 'id': "angularjs"},
        'Swift Agent': {'mode': True, 'id': "swift"},
        'MongoDB Agent': {'mode': True, 'id': "mongodb"},
        'PyTorch Agent': {'mode': True, 'id': "pytorch"},
        'Xcode Agent': {'mode': True, 'id': "xcode"},
        'Azure Agent': {'mode': True, 'id': "azure"},
        'Bitbucket Agent': {'mode': True, 'id': "bitbucket"},
        'DigitalOcean Agent': {'mode': True, 'id': "digitalocean"},
        'Docker Agent': {'mode': True, 'id': "docker"},
        'Electron Agent': {'mode': True, 'id': "electron"},
        'Erlang Agent': {'mode': True, 'id': "erlang"},
        'FastAPI Agent': {'mode': True, 'id': "fastapi"},
        'Firebase Agent': {'mode': True, 'id': "firebase"},
        'Flask Agent': {'mode': True, 'id': "flask"},
        'Git Agent': {'mode': True, 'id': "git"},
        'Gitlab Agent': {'mode': True, 'id': "gitlab"},
        'Go Agent': {'mode': True, 'id': "go"},
        'Godot Agent': {'mode': True, 'id': "godot"},
        'Google Cloud Agent': {'mode': True, 'id': "googlecloud"},
        'Heroku Agent': {'mode': True, 'id': "heroku"},
    }

    # Complete list of all models (for authorized users) - used for AVAILABLE_MODELS
    _all_models = list(dict.fromkeys([
        *models,  # Includes default_model, o3-mini, etc., and openrouter_models and agent names
        *list(agentMode.keys()), # Ensure all agentMode keys are included
        *list(trendingAgentMode.keys()) # Ensure all trendingAgentMode keys are included
    ]))

    AVAILABLE_MODELS = {name: name for name in _all_models}
    # Update AVAILABLE_MODELS to use names from agentMode if available
    for model_name_key in agentMode:
        if model_name_key in AVAILABLE_MODELS: # Check if the key from agentMode is in _all_models
            AVAILABLE_MODELS[model_name_key] = agentMode[model_name_key].get('name', model_name_key)


    # Model aliases for easier reference
    model_aliases = {
        "gpt-4": default_model, # default_model is "GPT-4.1"
        "gpt-4.1": default_model,
        "gpt-4o": default_model, # Defaulting to GPT-4.1 as per previous logic if specific GPT-4o handling isn't defined elsewhere
        "gpt-4o-mini": default_model, # Defaulting
        "claude-3.7-sonnet": "Claude-sonnet-3.7",
        "claude-3.5-sonnet": "Claude-sonnet-3.5",
        # "deepseek-r1": "DeepSeek-R1", # This is in base models, maps to R1 or DeepSeek R1 Zero in agentMode
        #
        "deepcoder-14b": "Deepcoder 14B Preview",
        "deephermes-3-8b": "DeepHermes 3 Llama 3 8B Preview",
        "deepseek-r1-zero": "DeepSeek R1 Zero",
        "deepseek-r1": "R1", # Alias for R1 (which is deepseek/deepseek-r1:free)
        "dolphin-3.0-24b": "Dolphin3.0 Mistral 24B",
        "dolphin-3.0-r1-24b": "Dolphin3.0 R1 Mistral 24B",
        "reka-flash": "Flash 3",
        "gemini-2.0-flash": "Gemini 2.0 Flash Experimental",
        "gemma-2-9b": "Gemma 2 9B",
        "gemma-3-12b": "Gemma 3 12B",
        "gemma-3-1b": "Gemma 3 1B",
        "gemma-3-27b": "Gemma 3 27B",
        "gemma-3-4b": "Gemma 3 4B",
        "kimi-vl-a3b-thinking": "Kimi VL A3B Thinking",
        "llama-3.1-8b": "Llama 3.1 8B Instruct",
        "nemotron-253b": "Llama 3.1 Nemotron Ultra 253B v1",
        "llama-3.2-11b": "Llama 3.2 11B Vision Instruct",
        "llama-3.2-1b": "Llama 3.2 1B Instruct",
        "llama-3.2-3b": "Llama 3.2 3B Instruct",
        "llama-3.3-70b": "Llama 3.3 70B Instruct",
        "nemotron-49b": "Llama 3.3 Nemotron Super 49B v1",
        "llama-4-maverick": "Llama 4 Maverick",
        "llama-4-scout": "Llama 4 Scout",
        "mistral-7b": "Mistral 7B Instruct",
        "mistral-nemo": "Mistral Nemo",
        "mistral-small-24b": "Mistral Small 3", # Alias for "Mistral Small 3"
        "mistral-small-24b-instruct-2501": "Mistral-Small-24B-Instruct-2501", # Specific name
        "mistral-small-3.1-24b": "Mistral Small 3.1 24B",
        "molmo-7b": "Molmo 7B D",
        "moonlight-16b": "Moonlight 16B A3B Instruct",
        "qwen-2.5-72b": "Qwen2.5 72B Instruct",
        "qwen-2.5-7b": "Qwen2.5 7B Instruct",
        "qwen-2.5-coder-32b": "Qwen2.5 Coder 32B Instruct",
        "qwen-2.5-vl-32b": "Qwen2.5 VL 32B Instruct",
        "qwen-2.5-vl-3b": "Qwen2.5 VL 3B Instruct",
        "qwen-2.5-vl-72b": "Qwen2.5 VL 72B Instruct",
        "qwen-2.5-vl-7b": "Qwen2.5-VL 7B Instruct",
        "qwerky-72b": "Qwerky 72B",
        "qwq-32b": "QwQ 32B",
        "qwq-32b-preview": "QwQ 32B Preview",
        "qwq-32b-arliai": "QwQ 32B RpR v1",
        "deepseek-r1-distill-llama-70b": "R1 Distill Llama 70B",
        "deepseek-r1-distill-qwen-14b": "R1 Distill Qwen 14B",
        "deepseek-r1-distill-qwen-32b": "R1 Distill Qwen 32B",
    }

    def __init__(
        self,
        is_conversation: bool = True,
        max_tokens: int = 8000,
        timeout: int = 30,
        intro: str = None,
        filepath: str = None,
        update_file: bool = True,
        proxies: dict = {},
        history_offset: int = 10250,
        act: str = None,
        model: str = "gpt-4.1",
        system_message: str = "You are a helpful AI assistant."
    ):
        """Initialize BlackboxAI with enhanced configuration options."""
        self.session = requests.Session()
        self.max_tokens_to_sample = max_tokens
        self.is_conversation = is_conversation
        self.timeout = timeout
        self.last_response = {}
        self.model = self.get_model(model)
        self.system_message = system_message

        self.headers = {
            "Content-Type": "application/json",
            "Accept": "*/*",
        }

        self.__available_optimizers = (
            method for method in dir(Optimizers)
            if callable(getattr(Optimizers, method)) and not method.startswith("__")
        )

        Conversation.intro = (
            AwesomePrompts().get_act(
                act, raise_not_found=True, default=None, case_insensitive=True
            )
            if act
            else intro or Conversation.intro
        )

        self.conversation = Conversation(
            is_conversation, self.max_tokens_to_sample, filepath, update_file
        )
        self.conversation.history_offset = history_offset
        self.session.proxies = proxies

    @classmethod
    def get_model(cls, model: str) -> str:
        """Resolve model name from alias"""
        # Convert to lowercase for case-insensitive matching
        model_lower = model.lower()

        # Check aliases (case-insensitive)
        for alias, target in cls.model_aliases.items():
            if model_lower == alias.lower():
                model = target
                break

        # Check available models (case-insensitive)
        for available_model, target in cls.AVAILABLE_MODELS.items():
            if model_lower == available_model.lower() or model == target:
                return target

        # If we get here, the model wasn't found
        raise ValueError(f"Unknown model: {model}. Available models: {', '.join(cls.AVAILABLE_MODELS)}")

    @classmethod
    def generate_session(cls, email: str, id_length: int = 21, days_ahead: int = 30) -> dict:
        """
        Generate a dynamic session with proper ID and expiry format using a specific email.

        Args:
            email: The email to use for this session
            id_length: Length of the numeric ID (default: 21)
            days_ahead: Number of days ahead for expiry (default: 30)

        Returns:
            dict: A session dictionary with user information and expiry
        """
        # Generate a random name
        first_names = ["Alex", "Jordan", "Taylor", "Morgan", "Casey", "Riley", "Avery", "Quinn", "Skyler", "Dakota"]
        last_names = ["Smith", "Johnson", "Williams", "Brown", "Jones", "Miller", "Davis", "Garcia", "Rodriguez", "Wilson"]
        name = f"{random.choice(first_names)} {random.choice(last_names)}"

        # Generate numeric ID - using Google-like ID format
        numeric_id = ''.join(random.choice('0123456789') for _ in range(id_length))

        # Generate future expiry date
        future_date = datetime.now() + timedelta(days=days_ahead)
        expiry = future_date.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z'

        # Generate random image ID for the new URL format
        chars = string.ascii_letters + string.digits + "-"
        random_img_id = ''.join(random.choice(chars) for _ in range(48))
        image_url = f"https://lh3.googleusercontent.com/a/ACg8oc{random_img_id}=s96-c"

        return {
            "user": {
                "name": name,
                "email": email,
                "image": image_url,
                "id": numeric_id
            },
            "expires": expiry,
            "isNewUser": False
        }

    @classmethod
    def generate_id(cls, length: int = 7) -> str:
        """Generate a random ID of specified length"""
        chars = string.ascii_letters + string.digits
        return ''.join(random.choice(chars) for _ in range(length))

    def _make_request(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        temperature: float = None,
        top_p: float = None,
        max_tokens: int = None,
        media: List = None
    ) -> Generator[str, None, None]:
        """Make synchronous request to BlackboxAI API."""
        # Generate a chat ID for this conversation
        chat_id = self.generate_id()

        # Format messages for the API
        current_messages = []
        for i, msg in enumerate(messages):
            msg_id = chat_id if i == 0 and msg["role"] == "user" else self.generate_id()
            current_msg = {
                "id": msg_id,
                "content": msg["content"],
                "role": msg["role"]
            }
            current_messages.append(current_msg)

        # Add image data if provided
        if media:
            current_messages[-1]['data'] = {
                "imagesData": [
                    {
                        "filePath": f"/{image_name}",
                        "contents": to_data_uri(image)
                    } for image, image_name in media
                ],
                "fileText": "",
                "title": ""
            }

        # Generate a random email for the session
        chars = string.ascii_lowercase + string.digits
        random_team = ''.join(random.choice(chars) for _ in range(8))
        request_email = f"{random_team}@blackbox.ai"

        # Generate a session with the email
        session_data = self.generate_session(request_email)

        # Prepare the request data based on the working example
        data = {
            "messages": current_messages,
            "agentMode": self.agentMode.get(self.model, {}) if self.model in self.agentMode else {},
            "id": chat_id,
            "previewToken": None,
            "userId": None,
            "codeModelMode": True,
            "trendingAgentMode": {},
            "isMicMode": False,
            "userSystemPrompt": self.system_message,
            "maxTokens": max_tokens or self.max_tokens_to_sample,
            "playgroundTopP": top_p,
            "playgroundTemperature": temperature,
            "isChromeExt": False,
            "githubToken": "",
            "clickedAnswer2": False,
            "clickedAnswer3": False,
            "clickedForceWebSearch": False,
            "visitFromDelta": False,
            "isMemoryEnabled": False,
            "mobileClient": False,
            "userSelectedModel": self.model if self.model in self.userSelectedModel else None,
            "validated": "00f37b34-a166-4efb-bce5-1312d87f2f94",  # Using a fixed validated value from the example
            "imageGenerationMode": False,
            "webSearchModePrompt": False,
            "deepSearchMode": False,
            "designerMode": False,
            "domains": None,
            "vscodeClient": False,
            "codeInterpreterMode": False,
            "customProfile": {
                "name": "",
                "occupation": "",
                "traits": [],
                "additionalInfo": "",
                "enableNewChats": False
            },
            "webSearchModeOption": {
                "autoMode": True,
                "webMode": False,
                "offlineMode": False
            },
            "session": session_data,
            "isPremium": True,
            "subscriptionCache": {
                "status": "PREMIUM",
                "customerId": "cus_" + ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(14)),
                "expiryTimestamp": int((datetime.now() + timedelta(days=30)).timestamp()),
                "lastChecked": int(datetime.now().timestamp() * 1000),
                "isTrialSubscription": True
            },
            "beastMode": False,
            "reasoningMode": False,
            "designerMode": False,
            "workspaceId": ""
        }

        # Use LitAgent to generate a realistic browser fingerprint for headers
        agent = LitAgent()
        fingerprint = agent.generate_fingerprint("chrome")
        headers = {
            'accept': fingerprint['accept'],
            'accept-encoding': 'gzip, deflate, br, zstd',
            'accept-language': fingerprint['accept_language'],
            'content-type': 'application/json',
            'origin': 'https://www.blackbox.ai',
            'referer': 'https://www.blackbox.ai/',
            'sec-ch-ua': fingerprint['sec_ch_ua'],
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': f'"{fingerprint["platform"]}"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-origin',
            'user-agent': fingerprint['user_agent']
        }

        try:
            response = self.session.post(
                self.api_endpoint,
                json=data,
                headers=headers,
                stream=stream,
                timeout=self.timeout
            )

            if not response.ok:
                error_msg = f"API request failed: {response.status_code} - {response.text}"

                # Check for service suspension
                if response.status_code == 503 and "service has been suspended" in response.text.lower():
                    error_msg = "BlackboxAI service has been suspended by its owner. Please try again later or use a different provider."

                # Check for API endpoint issues
                if response.status_code == 403 and "replace" in response.text.lower() and "api.blackbox.ai" in response.text:
                    error_msg = "BlackboxAI API endpoint issue. Please check the API endpoint configuration."

                raise exceptions.FailedToGenerateResponseError(error_msg)

            if stream:
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        if "You have reached your request limit for the hour" in line:
                            raise exceptions.RateLimitError("Rate limit exceeded")
                        yield line
            else:
                response_text = response.text
                if "You have reached your request limit for the hour" in response_text:
                    raise exceptions.RateLimitError("Rate limit exceeded")
                yield response_text

        except requests.exceptions.RequestException as e:
            raise exceptions.ProviderConnectionError(f"Connection error: {str(e)}")

    def ask(
        self,
        prompt: str,
        stream: bool = False,
        temperature: float = None,
        top_p: float = None,
        max_tokens: int = None,
        optimizer: str = None,
        conversationally: bool = False,
        media: List = None
    ) -> Union[Dict[str, str], Generator[Dict[str, str], None, None]]:
        """Send a prompt to BlackboxAI API and return the response."""
        conversation_prompt = self.conversation.gen_complete_prompt(prompt)
        if optimizer:
            if optimizer in self.__available_optimizers:
                conversation_prompt = getattr(Optimizers, optimizer)(
                    conversation_prompt if conversationally else prompt
                )
            else:
                raise ValueError(f"Optimizer is not one of {self.__available_optimizers}")

        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": conversation_prompt}
        ]

        def for_stream():
            for text in self._make_request(
                messages,
                stream=True,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                media=media
            ):
                yield {"text": text}

        def for_non_stream():
            response_text = next(self._make_request(
                messages,
                stream=False,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                media=media
            ))
            self.last_response = {"text": response_text}
            return self.last_response

        return for_stream() if stream else for_non_stream()

    def chat(
        self,
        prompt: str,
        stream: bool = False,
        temperature: float = None,
        top_p: float = None,
        max_tokens: int = None,
        optimizer: str = None,
        conversationally: bool = False,
        media: List = None
    ) -> Union[str, Generator[str, None, None]]:
        """Generate response as string."""

        def for_stream():
            for response in self.ask(
                prompt,
                stream=True,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                optimizer=optimizer,
                conversationally=conversationally,
                media=media
            ):
                yield self.get_message(response)

        def for_non_stream():
            return self.get_message(
                self.ask(
                    prompt,
                    stream=False,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    optimizer=optimizer,
                    conversationally=conversationally,
                    media=media
                )
            )

        return for_stream() if stream else for_non_stream()

    def get_message(self, response: Dict[str, Any]) -> str:
        """Extract message from response dictionary."""
        assert isinstance(response, dict), "Response should be of dict data-type only"
        return response["text"].replace('\\n', '\n').replace('\\n\\n', '\n\n')

if __name__ == "__main__":
    print("-" * 80)
    print(f"{'Model':<50} {'Status':<10} {'Response'}")
    print("-" * 80)

    for model in BLACKBOXAI.AVAILABLE_MODELS:
        try:
            test_ai = BLACKBOXAI(model=model, timeout=60)
            response = test_ai.chat("Say 'Hello' in one word")
            response_text = response
            
            if response_text and len(response_text.strip()) > 0:
                status = "✓"
                # Truncate response if too long
                display_text = response_text.strip()[:50] + "..." if len(response_text.strip()) > 50 else response_text.strip()
            else:
                status = "✗"
                display_text = "Empty or invalid response"
            print(f"{model:<50} {status:<10} {display_text}")
        except Exception as e:
            print(f"{model:<50} {'✗':<10} {str(e)}")
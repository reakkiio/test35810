import requests
import json

def fetch_together_models():
    """Fetch models from Together.xyz API"""
    api_key = "56c8eeff9971269d7a7e625ff88e8a83a34a556003a5c87c289ebe9a3d8a3d2c"
    endpoint = "https://api.together.xyz/v1/models"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }
    
    try:
        response = requests.get(endpoint, headers=headers, timeout=30)
        response.raise_for_status()
        
        models_data = response.json()
        
        # Extract and categorize models
        chat_models = []
        image_models = []
        language_models = []
        all_models = []
        
        print(f"Total models found: {len(models_data)}")
        print("\n" + "="*80)
        
        for model in models_data:
            if isinstance(model, dict):
                model_id = model.get("id", "")
                model_type = model.get("type", "").lower()
                context_length = model.get("context_length", 0)
                
                if not model_id:
                    continue
                
                all_models.append(model_id)
                
                # Categorize by type
                if model_type == "chat":
                    chat_models.append(model_id)
                elif model_type == "image":
                    image_models.append(model_id)
                elif model_type == "language":
                    language_models.append(model_id)
                
                # Print model details
                print(f"Model: {model_id}")
                print(f"  Type: {model_type}")
                print(f"  Context Length: {context_length}")
                # if model.get("config"):
                #     config = model["config"]
                #     if config.get("stop"):
                #         print(f"  Stop Tokens: {config['stop']}")
                # print("-" * 40)
        
        print(f"\nSUMMARY:")
        print(f"Chat Models: {len(chat_models)}")
        print(f"Image Models: {len(image_models)}")
        print(f"Language Models: {len(language_models)}")
        print(f"Total Models: {len(all_models)}")
        
        # Generate Python list for code
        print("\n" + "="*80)
        print("AVAILABLE_MODELS = [")
        for model in sorted(all_models):
            print(f'    "{model}",')
        print("]")
        
        return {
            "all_models": all_models,
            "chat_models": chat_models,
            "image_models": image_models,
            "language_models": language_models,
            "raw_data": models_data
        }
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching models: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON response: {e}")
        return None

if __name__ == "__main__":
    result = fetch_together_models()
    
    if result:
        print(f"\n📊 Successfully fetched {len(result['all_models'])} models from Together.xyz")
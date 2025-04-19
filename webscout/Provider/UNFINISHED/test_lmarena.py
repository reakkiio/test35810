import cloudscraper

def main():
    print("Testing cloudscraper access to LMArena...")
    try:
        scraper = cloudscraper.create_scraper(browser={
            'browser': 'chrome',
            'platform': 'windows',
            'desktop': True
        })
        
        # Test basic access
        response = scraper.get("https://lmarena.ai")
        print(f"Status code: {response.status_code}")
        print(f"Response length: {len(response.text)}")
        print("Cloudscraper test successful!")
        
        # Generate a session hash
        import uuid
        session_hash = str(uuid.uuid4()).replace("-", "")
        print(f"Session hash: {session_hash}")
        
        # Create payloads
        model_id = "gpt-4o"
        prompt = "Hello, what is your name?"
        
        first_payload = {
            "data": [
                None,
                model_id,
                {"text": prompt, "files": []},
                {
                    "text_models": [model_id],
                    "all_text_models": [model_id],
                    "vision_models": [],
                    "all_vision_models": [],
                    "image_gen_models": [],
                    "all_image_gen_models": [],
                    "search_models": [],
                    "all_search_models": [],
                    "models": [model_id],
                    "all_models": [model_id],
                    "arena_type": "text-arena"
                }
            ],
            "event_data": None,
            "fn_index": 117,
            "trigger_id": 159,
            "session_hash": session_hash
        }
        
        second_payload = {
            "data": [],
            "event_data": None,
            "fn_index": 118,
            "trigger_id": 159,
            "session_hash": session_hash
        }
        
        third_payload = {
            "data": [None, 0.7, 1, 2048],
            "event_data": None,
            "fn_index": 119,
            "trigger_id": 159,
            "session_hash": session_hash
        }
        
        # Set up headers
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        
        # Make requests
        print("Sending first request...")
        response = scraper.post(
            "https://lmarena.ai/queue/join?",
            json=first_payload,
            headers=headers
        )
        print(f"First response status: {response.status_code}")
        
        print("Sending second request...")
        response = scraper.post(
            "https://lmarena.ai/queue/join?",
            json=second_payload,
            headers=headers
        )
        print(f"Second response status: {response.status_code}")
        
        print("Sending third request...")
        response = scraper.post(
            "https://lmarena.ai/queue/join?",
            json=third_payload,
            headers=headers
        )
        print(f"Third response status: {response.status_code}")
        
        # Stream the response
        stream_url = f"https://lmarena.ai/queue/data?session_hash={session_hash}"
        print(f"Streaming from: {stream_url}")
        
        with scraper.get(stream_url, headers={"Accept": "text/event-stream"}, stream=True) as response:
            print(f"Stream response status: {response.status_code}")
            text_position = 0
            response_text = ""
            
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    print(line)
 
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

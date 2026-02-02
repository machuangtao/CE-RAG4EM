import os
import time
from functools import partial
from typing import List

from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm
import multiprocessing as mp


load_dotenv()

def _process_single_request(
    messages: List, *,
    model: str,
    temperature: float,
    max_retries: int = 3
) -> str:
    """
    Executes a single chat request using Gemini API via OpenAI-compatible interface.
    
    Env vars:
      - Google_API_KEY: required for Gemini API access
      - GEMINI_BASE_URL (optional): base URL for Gemini API, defaults to Google's endpoint
    """

    google_api_key = os.getenv("Google_API_KEY")
    if not google_api_key:
        return "ERROR: No API key found. Set Google_API_KEY environment variable."


    gemini_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        
        # Map OpenAI roles to Gemini roles
        if role == "system":
            # Prepend system message to first user message
            gemini_messages.insert(0, types.Content(role="user", parts=[types.Part(text=content)]))
        elif role == "assistant":
            gemini_messages.append(types.Content(role="model", parts=[types.Part(text=content)]))
        else:  # user
            gemini_messages.append(types.Content(role="user", parts=[types.Part(text=content)]))
    
   
    for attempt in range(max_retries):
        try:
            client = genai.Client(api_key=google_api_key)
            response = client.models.generate_content(
                model = model,
                contents = gemini_messages,
                config = types.GenerateContentConfig(
                    temperature = temperature,
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                )
            )
            return response.text.strip()
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
                continue
            else:
                return f"ERROR: Failed after {max_retries} attempts: {str(e)}"


def process_gemini_requests(
        model: str,
        messages_list: list,
        temperature: float = 0.5,
        chunk_size: int = 128,
        max_processes: int = 120,
) -> list:
    """
    Execute multiple requests, while the requests will be split into chunks first.
    Within each chunk, parallel requests will be sent in respect with the cpu_count.

    Args:
        model (str): OpenAI model to use
        temperature (float): Temperature scaling factor
        messages_list (list): List of messages, each message is a EntityMatchPrompt instance
        chunk_size (int): Number of messages to process at a time (default 25 to stay within free tier limit)
        max_processes (int): Maximum number of processes to use, to avoid hit the rate limit (default 5 for free tier)
    """
    num_processes = min(mp.cpu_count(), max_processes)
    print(f"Using {num_processes} processes to handle {len(messages_list)} requests")
    bound_fn = partial(_process_single_request, model=model, temperature=temperature)
    results = []

    start = time.time()
    for i in tqdm(range(0, len(messages_list), chunk_size)):
        chunk = messages_list[i : i + chunk_size]
        try:
            with mp.Pool(num_processes) as pool:
                chunk_results = list(pool.map_async(bound_fn, chunk).get(timeout=360))
                results.extend(chunk_results)
        except mp.TimeoutError:
            print(f"\nTimeout processing chunk {i // chunk_size + 1}. Moving to next chunk...")
            continue
        except Exception as e:
            print(f"\nError processing chunk {i // chunk_size + 1}: {str(e)}")
            print("Continuing with next chunk...")
            continue

    end = time.time()
    print(f"\nProcessed {len(messages_list)} requests in {end - start} seconds")
    return results


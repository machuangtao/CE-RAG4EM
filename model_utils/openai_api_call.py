import os
import time
from functools import partial
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
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
        Executes a single chat request using OpenAI official API.
        Chooses a proxy if PROXY_OPENAI_API_KEY is set; otherwise uses OPENAI_API_KEY.

        Env vars:
          - PROXY_OPENAI_API_KEY (optional): if set, route via proxy
          - OPENAI_API_KEY (fallback): used if proxy key not set
          - OPENAI_BASE_URL (optional): override base_url for proxy; default shown below
    """
    proxy_key = os.getenv("PROXY_OPENAI_API_KEY")
    direct_key = os.getenv("OPENAI_API_KEY")
    if not (proxy_key or direct_key):
        return "ERROR: No API key found. Set PROXY_OPENAI_API_KEY or OPENAI_API_KEY."

    if proxy_key:
        base_url = os.getenv("OPENAI_PROXY_URL")
        client = OpenAI(
            api_key=proxy_key,
            base_url=base_url,
            max_retries=max_retries
        )
    else:
        client = OpenAI(
            api_key=direct_key,
            max_retries=max_retries
        )

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    response = completion.choices[0].message.content.strip()
    return response


def process_openai_requests(
        model: str,
        messages_list: list,
        temperature: float = 0.5,
        chunk_size: int = 128,
        max_processes: int = 128,
) -> list:
    """
    Execute multiple requests, while the requests will be split into chunks first.
    Within each chunk, parallel requests will be sent in respect with the cpu_count.

    Args:
        model (str): OpenAI model to use
        temperature (float): Temperature scaling factor
        messages_list (list): List of messages, each message is a EntityMatchPrompt instance
        chunk_size (int): Number of messages to process at a time
        max_processes (int): Maximum number of processes to use, to avoid hit the rate limit
    """
    try:
        n_cpus = len(os.sched_getaffinity(0))   # safer way to use on a cluster
        num_processes = min(max_processes, n_cpus)
    except AttributeError:
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


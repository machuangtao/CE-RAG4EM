import time
import os
import asyncio
from typing import List, Dict, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# vLLM optional imports
# =========================
try:
    # vLLM-related env vars (set BEFORE importing vllm)
    os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
    os.environ["VLLM_ENGINE_ITERATION_TIMEOUT_S"] = "60"

    from vllm import SamplingParams
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs

    VLLM_AVAILABLE = True
except ImportError:
    AsyncLLMEngine = None
    SamplingParams = None
    AsyncEngineArgs = None
    VLLM_AVAILABLE = False


# =========================
# Model name → IDs / paths
# =========================

def _resolve_hf_model_id(model_name: str) -> str:
    if model_name == "jellyfish-8b":
        return "NECOUDBFM/Jellyfish-8B"
    elif model_name == "jellyfish-7b":
        return "NECOUDBFM/Jellyfish-7B"
    elif model_name == "mistral-7b":
        return "mistralai/Mistral-7B-Instruct-v0.3"
    elif model_name == "qwen-7b":
        return "Qwen/Qwen2.5-7B-Instruct"
    elif model_name == "qwen3-4b":
        return "Qwen/Qwen3-4B-Instruct-2507"
    elif model_name == "qwen3-8b":
        return "Qwen/Qwen3-8B"
    else:
        raise ValueError(
            f"Unsupported model: '{model_name}'. "
            "Supported models: jellyfish-8b, jellyfish-7b, mistral-7b, qwen-7b, qwen3-4b, qwen3-8b"
        )

def _resolve_vllm_model_uri(model_name: str) -> str:
    """
    For vLLM, you can either:
      - Use the same HF repo ID, or
      - Point to a local checkpoint directory.
    Edit this mapping if you have local paths.
    """
    # By default, just use the HF IDs
    return _resolve_hf_model_id(model_name)
    # Example if you want local paths instead:
    # mapping = {
    #     "jellyfish-8b": "/path/to/jellyfish-8b-ckpt",
    #     "jellyfish-7b": "/path/to/jellyfish-7b-ckpt",
    #     ...
    # }
    # return mapping[model_name]


# =========================
# Hugging Face backend
# =========================

def _load_hf_model(model_name: str):
    """
    Load HF model + tokenizer for text generation.
    Automatically uses GPU if available, else CPU.
    """
    model_id = _resolve_hf_model_id(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id)
        model = model.to(device)

    return model, tokenizer, device


def _hf_generate_one(
    model,
    tokenizer,
    device,
    message: str,
    temperature: float,
    max_new_tokens: int = 4096,
) -> str:
    inputs = tokenizer(str(message), return_tensors="pt").to(device)
    # Map OpenAI-style messages to model input
    if isinstance(message, list):
        # Handle OpenAI-style message format [{"role": "user", "content": "..."}]
        formatted_messages = []
        for msg in message:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                formatted_messages.append({"role": "system", "content": content})
            elif role == "user":
                formatted_messages.append({"role": "user", "content": content})
            elif role == "assistant":
                formatted_messages.append({"role": "assistant", "content": content})
        
        # Use chat template if available
        if hasattr(tokenizer, "apply_chat_template"):
            inputs = tokenizer.apply_chat_template(
                formatted_messages, 
                return_tensors="pt",
                add_generation_prompt=True
            )
            inputs = {"input_ids": inputs.to(device)}
        else:
            # Fallback: concatenate messages
            text = "\n".join([f"{m['role']}: {m['content']}" for m in formatted_messages])
            inputs = tokenizer(text, return_tensors="pt").to(device)
    else:
        inputs = tokenizer(str(message), return_tensors="pt").to(device)
    input_length = inputs['input_ids'].shape[1]
    
    # Ensure attention_mask is set
    if 'attention_mask' not in inputs:
        inputs['attention_mask'] = torch.ones_like(inputs['input_ids']).to(device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
    )
    # Get the generated tokens *after* the input prompt
    input_length = inputs["input_ids"].shape[-1]
    generated_tokens = outputs[0][input_length:]

    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def _hf_process_local_request(
    model_name: str,
    messages_list: List[str],
    temperature: float = 0.5,
) -> List[str]:
    model, tokenizer, device = _load_hf_model(model_name)
    responses: List[str] = []

    # Simple sequential loop
    for message in messages_list:
        resp = _hf_generate_one(
            model, tokenizer, device, message, temperature=temperature
        )
        responses.append(resp)

    return responses


# =========================
# vLLM backend
# =========================

def _initialize_vllm_engine(model_name: str):
    """
    Initialize a vLLM AsyncLLMEngine for the given model.
    Uses max_model_len=4096 to fit within available GPU memory (11.5 GiB).
    """
    model_uri = _resolve_vllm_model_uri(model_name)

    engine_args = AsyncEngineArgs(
        model=model_uri,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        kv_cache_dtype="fp8_e5m2",
        enable_chunked_prefill=True,
        trust_remote_code=True,
        max_model_len=4096  # Limit to fit available GPU memory
    )
    return AsyncLLMEngine.from_engine_args(engine_args)


async def _vllm_generate_batch_async(
    model_name: str,
    prompts: List[str],
    temperature: float,
    max_tokens: int = 4096,
) -> List[str]:
    """
    Async helper that uses vLLM to generate for all prompts in parallel
    and returns a list of strings in the same order.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    batch_start_time = time.time()
    logger.info(f"Initializing vLLM engine for model: {model_name}")
    engine = _initialize_vllm_engine(model_name)
    init_time = time.time() - batch_start_time
    logger.info(f"Engine initialization took {init_time:.2f}s")
    
    try:
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.05,
            detokenize=True,
            max_tokens=max_tokens,
        )

        results: List[Optional[str]] = [None] * len(prompts)
        request_times: Dict[str, float] = {}

        async def run_one(i: int, prompt: str) -> None:
            req_id = f"req-{i}"
            req_start_time = time.time()
            final = None
            
            logger.debug(f"Processing request {req_id}, prompt length: {len(prompt)}")
            
            async for out in engine.generate(prompt, sampling_params, req_id):
                final = out  # keep last chunk as the final result

            if final is None:
                logger.warning(f"[{req_id}] No output from engine")
                results[i] = ""
                request_times[req_id] = time.time() - req_start_time
                return

            outputs = getattr(final, "outputs", []) or []
            text = outputs[0].text if outputs else ""
            logger.debug(f"[{req_id}] Output text length: {len(text)}")
            results[i] = text
            request_times[req_id] = time.time() - req_start_time

        gather_start_time = time.time()
        await asyncio.gather(*(run_one(i, p) for i, p in enumerate(prompts)))
        gather_time = time.time() - gather_start_time

        # Replace any None with empty string to be safe
        total_time = time.time() - batch_start_time
        avg_time = total_time / len(prompts) if prompts else 0
        logger.info(f"vLLM batch complete: {len(results)} results in {total_time:.2f}s (avg: {avg_time:.2f}s per request)")
        logger.info(f"Parallel batch processing took {gather_time:.2f}s")
        
        return [r if r is not None else "" for r in results]
    finally:
        engine.shutdown()


def _vllm_process_local_request(
    model_name: str,
    messages_list: List[str],
    temperature: float = 0.5,
) -> List[str]:
    """
    Synchronous wrapper around the async vLLM batch generation.
    Uses asyncio.run() which properly handles event loop creation and cleanup.
    
    Handles both:
    - List of strings (plain prompts)
    - List of message lists (OpenAI-style format: [[{"role": "user", "content": "..."}], ...])
    - List of message dicts (single message: [{"role": "user", "content": "..."}, ...])
    """
    import logging
    logger = logging.getLogger(__name__)
    
    if not messages_list:
        return []
    
    overall_start = time.time()
    
    # Convert message formats to plain strings
    prompts = []
    for msg in messages_list:
        if isinstance(msg, list):
            # Handle list of message dicts format: [{"role": "user", "content": "..."}]
            prompt_text = ""
            for m in msg:
                if isinstance(m, dict):
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    prompt_text += f"{role}: {content}\n"
            prompts.append(prompt_text.strip())
        elif isinstance(msg, dict):
            # Handle single message dict: {"role": "user", "content": "..."}
            content = msg.get("content", "")
            prompts.append(str(content))
        else:
            # Already a string
            prompts.append(str(msg))
    
    conversion_time = time.time() - overall_start
    logger.info(f"Message conversion took {conversion_time:.2f}s for {len(prompts)} prompts")
    
    try:
        result = asyncio.run(
            _vllm_generate_batch_async(
                model_name=model_name,
                prompts=prompts,
                temperature=temperature,
            )
        )
        total_time = time.time() - overall_start
        logger.info(f"Total vLLM processing time: {total_time:.2f}s")
        return result
    except Exception as e:
        logging.error(f"Error in vLLM processing: {str(e)}", exc_info=True)
        # Return empty list of same length as input for graceful degradation
        return [""] * len(messages_list)


# =========================
# Public API (what we will call)
# =========================

def process_local_request(
    model_name: str,
    messages_list: List[str],
    temperature: float = 0.5,
    backend: Optional[str] = None,
) -> List[str]:
    """
    Process text generation requests using local language models.

    This function handles text generation by automatically selecting the appropriate backend:

    - (suggested) If vLLM is installed, uses vLLM for fast, batched inference.
    - Otherwise falls back to the original Hugging Face implementation.
    
    Args:
        model_name: The name of the model to use
        messages_list: List of prompts to process
        temperature: Sampling temperature (default 0.5)
        backend: Force a specific backend ("vllm" or "hf"). If None, uses vLLM if available, else HF.
    """
    # Default to vLLM if available, otherwise fall back to HF
    if backend is None:
        backend = "vllm" if VLLM_AVAILABLE else "hf"
    
    if backend == "vllm" and VLLM_AVAILABLE:
        return _vllm_process_local_request(
            model_name=model_name,
            messages_list=messages_list,
            temperature=temperature,
        )
    else:
        # Fallback: original HF-based slow path
        return _hf_process_local_request(
            model_name=model_name,
            messages_list=messages_list,
            temperature=temperature,
        )
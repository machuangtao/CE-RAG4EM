"""
Send the prompts built by prepare_prompts.py (prompts_outputs/<dataset>/<dataset>_<partition>_{falcon,wikidata}_prompts.json)
to the OpenAI API and save the raw responses + parsed Yes/No decisions to disk.

Key parameters are kept consistent with ce_rag4em_main.py / model_utils/openai_api_call.py:
  - model: gpt-4o-mini
  - temperature: 0.5
  - max_retries: 3

Designed to be resumable (only pending pairs are re-sent) and submittable via SLURM
(see slurm/run_openai_inference.slurm).

Usage:
    python run_openai_inference.py --dataset beer --source wikidata
"""

import argparse
import glob
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_TOP_K_PER_ENTITY = 2 
DATASET_DIR_MAP = {
    "abt": "abt_buy",
    "amgo": "amazon_google",
    "beer": "beer",
    "dbac": "dblp_acm",
    "dbgo": "dblp_scholar",
    "foza": "fodors_zagat",
    "itam": "itunes_amazon",
    "waam": "walmart_amazon",
    "wdc": "wdc",
}


def make_client(max_retries: int, no_proxy: bool = False) -> OpenAI:
    proxy_key = None if no_proxy else os.getenv("PROXY_OPENAI_API_KEY")
    direct_key = os.getenv("OPENAI_API_KEY")
    if not (proxy_key or direct_key):
        raise RuntimeError("No API key found. Set PROXY_OPENAI_API_KEY or OPENAI_API_KEY in .env.")
    if proxy_key:
        return OpenAI(api_key=proxy_key, base_url=os.getenv("OPENAI_PROXY_URL"), max_retries=max_retries)
    return OpenAI(api_key=direct_key, max_retries=max_retries)


def parse_decision(response: str):
    """Mirrors the robust Yes/No parsing in ce_rag4em_main.evaluate_predictions, so
    predictions here stay comparable to the main pipeline's results."""
    response_lower = (response or "").strip().lower()

    if "match decision" in response_lower:
        decision_start = response_lower.find("match decision")
        decision_part = response_lower[decision_start + len("match decision"):]
        decision_part_cleaned = (
            decision_part.replace(":", "").replace("*", "").replace('"', "").replace("'", "").replace("\\n", " ").strip()
        )
        words = decision_part_cleaned.split()
        first_word = words[0] if words else ""
        if first_word == "yes":
            return 1
        if first_word == "no":
            return 0

    if "yes" in response_lower or "matched" in response_lower:
        return 1
    if "no" in response_lower or "not matched" in response_lower:
        return 0
    return 0  # ambiguous -> default to no match, same as the main pipeline


def call_openai(client: OpenAI, model: str, messages: list, temperature: float, seed: int = None) -> str:
    kwargs = {"seed": seed} if seed is not None else {}
    completion = client.chat.completions.create(model=model, messages=messages, temperature=temperature, **kwargs)
    return completion.choices[0].message.content.strip()


def process_pair(client, model, temperature, seed, record, logger):
    try:
        response = call_openai(client, model, record["messages"], temperature, seed)
        return {**record, "response": response, "predicted_label": parse_decision(response)}
    except Exception as exc:
        logger.error("Pair %s failed: %s", record["pair_index"], exc)
        return {**record, "error": str(exc)}


def atomic_write_json(path: str, data: dict):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def run_dataset_source(dataset, source, partition, prompts_dir, output_dir, model, temperature,
                        max_retries, workers, checkpoint_every, extra_retry_rounds, retry_round_wait,
                        limit, logger, reranked=False, seed=None, top_k_per_entity=DEFAULT_TOP_K_PER_ENTITY,
                        no_proxy=False):
    top_k_tag = f"_top{top_k_per_entity}" if top_k_per_entity != DEFAULT_TOP_K_PER_ENTITY else ""
    source_tag = source + top_k_tag + ("_reranked" if reranked else "")
    prompts_path = os.path.join(prompts_dir, dataset, f"{dataset}_{partition}_{source_tag}_prompts.json")
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts_data = json.load(f)
    prompts = prompts_data["prompts"]

    run_tag = source_tag + (f"_seed{seed}" if seed is not None else "")
    out_dir = os.path.join(output_dir, dataset)
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f"{dataset}_{partition}_{run_tag}_results.json")

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            output = json.load(f)
        logger.info("Resuming from existing output at %s", output_path)
    else:
        output = {
            "metadata": {
                "dataset": dataset,
                "source": source,
                "reranked": reranked,
                "seed": seed,
                "partition": partition,
                "model": model,
                "temperature": temperature,
                "prompt_template": prompts_data["metadata"].get("prompt_template"),
            },
            "results": {},  # keyed by pair_index (str)
        }

    pending = [p for p in prompts if "error" in output["results"].get(str(p["pair_index"]), {"error": True})]
    completed_count = len(prompts) - len(pending)
    if limit is not None:
        pending = pending[:limit]
    logger.info("[%s/%s] completed: %d, pending: %d (processing %d this run)",
                dataset, run_tag, completed_count, len(prompts) - completed_count, len(pending))

    client = make_client(max_retries, no_proxy)

    def run_pass(records, desc):
        since_checkpoint = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_pair, client, model, temperature, seed, record, logger): record
                for record in records
            }
            for future in as_completed(futures):
                record = futures[future]
                result = future.result()
                output["results"][str(record["pair_index"])] = result

                since_checkpoint += 1
                if since_checkpoint >= checkpoint_every:
                    atomic_write_json(output_path, output)
                    since_checkpoint = 0
        logger.info("Finished pass: %s (%d records)", desc, len(records))

    run_pass(pending, f"{dataset}/{run_tag} main pass")

    for round_num in range(1, extra_retry_rounds + 1):
        still_failed = [p for p in pending if "error" in output["results"].get(str(p["pair_index"]), {})]
        if not still_failed:
            break
        logger.warning("[%s/%s] round %d: %d pairs still failed, waiting %.0fs before retrying",
                        dataset, run_tag, round_num, len(still_failed), retry_round_wait)
        atomic_write_json(output_path, output)
        time.sleep(retry_round_wait)
        run_pass(still_failed, f"{dataset}/{run_tag} retry-round-{round_num}")

    output["metadata"]["total_pairs"] = len(prompts)
    output["metadata"]["num_success"] = sum(1 for v in output["results"].values() if "error" not in v)
    output["metadata"]["num_failed"] = sum(1 for v in output["results"].values() if "error" in v)
    atomic_write_json(output_path, output)

    logger.info("[%s/%s] done. success=%d failed=%d total=%d -> %s",
                dataset, run_tag, output["metadata"]["num_success"], output["metadata"]["num_failed"],
                len(prompts), output_path)


def main():
    parser = argparse.ArgumentParser(description="Run saved RAG entity-matching prompts through the OpenAI API.")
    parser.add_argument("-d", "--dataset", default="all", choices=sorted(DATASET_DIR_MAP.keys()) + ["all"])
    parser.add_argument("-p", "--partition", default="test")
    parser.add_argument("--source", default="both", choices=["falcon", "wikidata", "pair", "both"])
    parser.add_argument("--prompts-dir", default="prompts_outputs")
    parser.add_argument("--output-dir", default="api_outputs")
    parser.add_argument("--reranked", action="store_true",
                         help="Read the *_reranked_prompts.json files produced by "
                              "prepare_prompts.py --reranked instead of the raw ones.")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--extra-retry-rounds", type=int, default=3)
    parser.add_argument("--retry-round-wait", type=float, default=60.0)
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N pending pairs (for testing).")
    parser.add_argument("--seed", type=int, default=None,
                         help="OpenAI API 'seed' param, for repeated runs across random seeds.")
    parser.add_argument("--top-k-per-entity", type=int, default=DEFAULT_TOP_K_PER_ENTITY,
                         help=f"Must match the --top-k-per-entity used to build the prompts "
                              f"(default: {DEFAULT_TOP_K_PER_ENTITY}).")
    parser.add_argument("--no-proxy", action="store_true",
                         help="Ignore PROXY_OPENAI_API_KEY and always use OPENAI_API_KEY directly "
                              "against the official OpenAI endpoint.")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger("openai_inference")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    top_k_tag = f"_top{args.top_k_per_entity}" if args.top_k_per_entity != DEFAULT_TOP_K_PER_ENTITY else ""
    log_tag = args.source + top_k_tag + ("_reranked" if args.reranked else "") + (f"_seed{args.seed}" if args.seed is not None else "")
    file_handler = logging.FileHandler(f"logs/openai_inference_{args.dataset}_{log_tag}.log")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    datasets = list(DATASET_DIR_MAP.keys()) if args.dataset == "all" else [args.dataset]
    sources = ["falcon", "wikidata"] if args.source == "both" else [args.source]

    for dataset in datasets:
        for source in sources:
            run_dataset_source(
                dataset, source, args.partition, args.prompts_dir, args.output_dir,
                args.model, args.temperature, args.max_retries, args.workers,
                args.checkpoint_every, args.extra_retry_rounds, args.retry_round_wait,
                args.limit, logger, args.reranked, args.seed, args.top_k_per_entity, args.no_proxy,
            )


if __name__ == "__main__":
    main()

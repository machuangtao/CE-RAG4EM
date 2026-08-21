"""
Run entity-matching LLM inference using each batch's shared top-2 QID context
(from batch_retrieval_outputs/) as the "Additional Information" in the same
rag4em prompt template used by prepare_prompts.py / run_openai_inference.py,
so prediction error can be measured as a function of batch_size.

Usage:
    python run_batch_inference.py --dataset beer --batch-size 6
"""

import argparse
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from data_utils.constants import PROMPT_TEMPLATES

load_dotenv()

CONTEXT_TOP_K = 2
NO_CONTEXT_FALLBACK = "No avaliable relevant knowledge, please make the decision on your own."


def format_qid_entry(qid: str, label, description) -> str:
    label = label or qid
    if description and description != label:
        return f"{qid} ({label}: {description})"
    return f"{qid} ({label})"


def build_context(relevant_qids: list) -> str:
    parts = [format_qid_entry(q["QID"], q.get("label"), q.get("description")) for q in relevant_qids[:CONTEXT_TOP_K]]
    return "; ".join(parts) if parts else NO_CONTEXT_FALLBACK


def parse_decision(response: str) -> int:
    """Same robust Yes/No parsing as run_openai_inference.py:parse_decision, kept
    as an independent copy so this script has no import-time coupling to it."""
    response_lower = (response or "").strip().lower()

    if "match decision" in response_lower:
        decision_part = response_lower[response_lower.find("match decision") + len("match decision"):]
        decision_part = decision_part.replace(":", "").replace("*", "").replace('"', "").replace("'", "").strip()
        first_word = decision_part.split()[0] if decision_part.split() else ""
        if first_word == "yes":
            return 1
        if first_word == "no":
            return 0

    if "yes" in response_lower or "matched" in response_lower:
        return 1
    if "no" in response_lower or "not matched" in response_lower:
        return 0
    return 0


def make_client(max_retries: int) -> OpenAI:
    proxy_key = os.getenv("PROXY_OPENAI_API_KEY")
    direct_key = os.getenv("OPENAI_API_KEY")
    if proxy_key:
        return OpenAI(api_key=proxy_key, base_url=os.getenv("OPENAI_PROXY_URL"), max_retries=max_retries)
    return OpenAI(api_key=direct_key, max_retries=max_retries)


def call_openai(client: OpenAI, model: str, content: str, temperature: float, seed: int = None) -> str:
    kwargs = {"seed": seed} if seed is not None else {}
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": content}],
        temperature=temperature,
        **kwargs,
    )
    return completion.choices[0].message.content.strip()


def process_pair(client, model, temperature, seed, pair_id, content, ground_truth, logger):
    try:
        response = call_openai(client, model, content, temperature, seed)
        return {"predicted_label": parse_decision(response), "response": response, "ground_truth": ground_truth}
    except Exception as exc:
        logger.error("Pair %s failed: %s", pair_id, exc)
        return {"error": str(exc), "ground_truth": ground_truth}


def atomic_write_json(path: str, data: dict):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def build_prompts(dataset: str, batch_size: int) -> dict:
    """pair_id -> content (fully formatted rag4em prompt), ground_truth"""
    pairs_df = pd.read_csv(f"batch_outputs/{dataset}/{dataset}_test_QG_{batch_size}_batches.csv")
    with open(f"batch_retrieval_outputs/{dataset}/{dataset}_test_QG_{batch_size}_batch_retrieval.json") as f:
        batches = json.load(f)["batches"]

    template = PROMPT_TEMPLATES["rag4em"]["user"]
    prompts = {}
    for row in pairs_df.itertuples():
        batch = batches[str(row.batch_id)]
        context_text = build_context(batch["relevant_qids"] or [])
        content = template.format(row.record_1, row.record_2, context_text)
        prompts[row.pair_id] = {"content": content, "ground_truth": int(row.ground_truth)}
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Run batch-retrieval-context entity-matching inference.")
    parser.add_argument("-d", "--dataset", required=True, choices=["beer", "foza", "itam"])
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--output-dir", default="batch_inference_outputs")
    parser.add_argument("--seed", type=int, default=None,
                         help="OpenAI API 'seed' param, for repeated runs across random seeds.")
    args = parser.parse_args()

    seed_tag = f"_seed{args.seed}" if args.seed is not None else ""

    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger(f"{args.dataset}_batch_inference{seed_tag}")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(f"logs/{args.dataset}_QG_{args.batch_size}_batch_inference{seed_tag}.log")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    out_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f"{args.dataset}_test_QG_{args.batch_size}{seed_tag}_results.json")

    prompts = build_prompts(args.dataset, args.batch_size)
    logger.info("Built %d prompts for dataset=%s batch_size=%d", len(prompts), args.dataset, args.batch_size)

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            output = json.load(f)
        logger.info("Resuming from existing output at %s", output_path)
    else:
        output = {
            "metadata": {"dataset": args.dataset, "batch_size": args.batch_size, "model": args.model,
                         "temperature": args.temperature, "seed": args.seed},
            "results": {},
        }

    pending = [pid for pid in prompts if "error" in output["results"].get(pid, {"error": True})]
    logger.info("Already completed: %d, pending: %d", len(prompts) - len(pending), len(pending))

    client = make_client(args.max_retries)

    since_checkpoint = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_pair, client, args.model, args.temperature, args.seed, pid,
                             prompts[pid]["content"], prompts[pid]["ground_truth"], logger): pid
            for pid in pending
        }
        for future in as_completed(futures):
            pid = futures[future]
            output["results"][pid] = future.result()
            since_checkpoint += 1
            if since_checkpoint >= args.checkpoint_every:
                atomic_write_json(output_path, output)
                since_checkpoint = 0

    output["metadata"]["total_pairs"] = len(prompts)
    output["metadata"]["num_success"] = sum(1 for v in output["results"].values() if "error" not in v)
    output["metadata"]["num_failed"] = sum(1 for v in output["results"].values() if "error" in v)
    atomic_write_json(output_path, output)

    # ---- metrics ----
    results = [v for v in output["results"].values() if "error" not in v]
    tp = sum(1 for r in results if r["predicted_label"] == 1 and r["ground_truth"] == 1)
    fp = sum(1 for r in results if r["predicted_label"] == 1 and r["ground_truth"] == 0)
    fn = sum(1 for r in results if r["predicted_label"] == 0 and r["ground_truth"] == 1)
    tn = sum(1 for r in results if r["predicted_label"] == 0 and r["ground_truth"] == 0)
    accuracy = (tp + tn) / len(results) if results else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")

    logger.info(
        "[%s/QG_%d] done. success=%d failed=%d acc=%.4f P=%.4f R=%.4f F1=%.4f -> %s",
        args.dataset, args.batch_size, output["metadata"]["num_success"], output["metadata"]["num_failed"],
        accuracy, precision, recall, f1, output_path,
    )
    print(f"{args.dataset}/QG_{args.batch_size}: acc={accuracy:.4f} P={precision:.4f} R={recall:.4f} F1={f1:.4f}")


if __name__ == "__main__":
    main()

"""
Evidence coverage/noise evaluation for batch retrieval (see subsubsec:acc_bs in the paper).

For each pair, pool its top-5 retrieved QIDs across batch sizes {1,2,4,6,8}, LLM-judge each
unique (pair, QID) exactly once ("Relevant"/"Irrelevant"), then reuse those judgments to compute
per-(pair, batch_size) coverage/noise against the top-2 QIDs actually used as context.

Usage:
    python evaluate_evidence_quality.py --dataset beer
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
from tqdm import tqdm

load_dotenv()

BATCH_SIZES = [1, 2, 4, 6, 8]
POOL_TOP_K = 5
CONTEXT_TOP_K = 2

JUDGE_PROMPT = """Record A:
{record_1}

Record B:
{record_2}

Knowledge evidence:
{evidence}

Question:
Is this evidence useful for determining whether Record A and Record B
refer to the same real-world entity?

Answer only:
Relevant
or
Irrelevant"""


def format_qid_entry(qid: str, label, description) -> str:
    label = label or qid
    if description and description != label:
        return f"{qid} ({label}: {description})"
    return f"{qid} ({label})"


def make_client(model: str) -> OpenAI:
    proxy_key = os.getenv("PROXY_OPENAI_API_KEY")
    direct_key = os.getenv("OPENAI_API_KEY")
    if proxy_key:
        return OpenAI(api_key=proxy_key, base_url=os.getenv("OPENAI_PROXY_URL"), max_retries=5)
    return OpenAI(api_key=direct_key, max_retries=5)


def judge_relevant(client: OpenAI, model: str, record_1: str, record_2: str, evidence: str, logger) -> bool:
    prompt = JUDGE_PROMPT.format(record_1=record_1, record_2=record_2, evidence=evidence)
    kwargs = {} if model.startswith("gpt-5") else {"temperature": 0}
    for attempt in range(1, 6):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            answer = (completion.choices[0].message.content or "").strip().lower()
            if "irrelevant" in answer:
                return False
            if "relevant" in answer:
                return True
            logger.warning("Unparseable judge answer %r, defaulting to Irrelevant", answer)
            return False
        except Exception as exc:
            if attempt == 5:
                raise
            delay = min(2.0 * (2 ** (attempt - 1)), 30.0)
            logger.warning("Judge call attempt %d failed (%s); retrying in %.1fs", attempt, exc, delay)
            time.sleep(delay)


def atomic_write_json(path: str, data: dict):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def load_pairs(dataset: str) -> pd.DataFrame:
    """Canonical pair_id -> record_1/record_2/ground_truth map (identical across batch sizes)."""
    path = f"batch_outputs/{dataset}/{dataset}_test_QG_1_batches.csv"
    df = pd.read_csv(path)[["pair_id", "record_1", "record_2", "ground_truth"]]
    return df.drop_duplicates(subset="pair_id").set_index("pair_id")


def load_batch_retrieval(dataset: str, batch_size: int) -> dict:
    path = f"batch_retrieval_outputs/{dataset}/{dataset}_test_QG_{batch_size}_batch_retrieval.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["batches"]


def build_pair_to_batch(dataset: str, batch_size: int) -> dict:
    path = f"batch_outputs/{dataset}/{dataset}_test_QG_{batch_size}_batches.csv"
    df = pd.read_csv(path)[["pair_id", "batch_id"]]
    return dict(zip(df["pair_id"], df["batch_id"].astype(str)))


def build_pool(dataset: str, pairs: pd.DataFrame, logger) -> tuple:
    """Returns (pool: pair_id -> {QID: {label, description}}, per_size_context: (b, pair_id) -> [QID,...] top-5)."""
    pool = {pid: {} for pid in pairs.index}
    per_size_top5 = {}
    for b in BATCH_SIZES:
        pair_to_batch = build_pair_to_batch(dataset, b)
        batches = load_batch_retrieval(dataset, b)
        for pid in pairs.index:
            batch_id = pair_to_batch[pid]
            top5 = (batches[batch_id]["relevant_qids"] or [])[:POOL_TOP_K]
            per_size_top5[(b, pid)] = [q["QID"] for q in top5]
            for q in top5:
                pool[pid][q["QID"]] = {"label": q.get("label"), "description": q.get("description")}
    total_pool_items = sum(len(v) for v in pool.values())
    logger.info("Built candidate pool: %d pairs, %d unique (pair,QID) items", len(pool), total_pool_items)
    return pool, per_size_top5


def main():
    parser = argparse.ArgumentParser(description="Evaluate evidence coverage/noise of batch retrieval via LLM-as-judge.")
    parser.add_argument("-d", "--dataset", required=True, choices=["beer", "foza", "itam"])
    parser.add_argument("--model", default="gpt-5-mini")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--output-dir", default="evidence_quality_outputs")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger(f"{args.dataset}_evidence_quality")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(f"logs/{args.dataset}_evidence_quality.log")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    out_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    judgments_path = os.path.join(out_dir, f"{args.dataset}_pair_evidence_judgments.json")
    metrics_path = os.path.join(out_dir, f"{args.dataset}_evidence_quality_metrics.csv")

    pairs = load_pairs(args.dataset)
    logger.info("Loaded %d pairs for dataset=%s", len(pairs), args.dataset)

    pool, per_size_top5 = build_pool(args.dataset, pairs, logger)

    # ---- Step 2: LLM-judge each unique (pair_id, QID) once, with resume ----
    if os.path.exists(judgments_path):
        with open(judgments_path, "r", encoding="utf-8") as f:
            output = json.load(f)
        logger.info("Resuming judgments from %s", judgments_path)
    else:
        output = {
            "metadata": {"dataset": args.dataset, "model": args.model},
            "judgments": {},
        }

    todo = []
    for pid, evidence_items in pool.items():
        for qid in evidence_items:
            key = f"{pid}|{qid}"
            if key not in output["judgments"]:
                todo.append((pid, qid))
    logger.info("Judgments already cached: %d, pending: %d", len(output["judgments"]), len(todo))

    client = make_client(args.model)

    def judge_one(pid, qid):
        record_1 = pairs.loc[pid, "record_1"]
        record_2 = pairs.loc[pid, "record_2"]
        info = pool[pid][qid]
        evidence = format_qid_entry(qid, info.get("label"), info.get("description"))
        relevant = judge_relevant(client, args.model, record_1, record_2, evidence, logger)
        return pid, qid, relevant

    since_checkpoint = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(judge_one, pid, qid): (pid, qid) for pid, qid in todo}
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{args.dataset}/judge"):
            pid, qid = futures[future]
            try:
                _, _, relevant = future.result()
                output["judgments"][f"{pid}|{qid}"] = relevant
            except Exception as exc:
                logger.error("Judgment for pair=%s qid=%s failed permanently: %s", pid, qid, exc)

            since_checkpoint += 1
            if since_checkpoint >= args.checkpoint_every:
                atomic_write_json(judgments_path, output)
                since_checkpoint = 0

    output["metadata"]["total_judgments"] = len(pool and {f"{p}|{q}" for p, e in pool.items() for q in e})
    atomic_write_json(judgments_path, output)
    logger.info("Judgments done: %d cached total -> %s", len(output["judgments"]), judgments_path)

    # ---- Steps 3-4: R_p* proxy + per-(pair, batch_size) coverage/noise ----
    rows = []
    for pid in pairs.index:
        r_star = {qid for qid in pool[pid] if output["judgments"].get(f"{pid}|{qid}") is True}
        r_star_size = len(r_star)

        per_b = {}
        for b in BATCH_SIZES:
            context_qids = set(per_size_top5[(b, pid)][:CONTEXT_TOP_K])
            rel = len(context_qids & r_star)
            noise = 1 - rel / CONTEXT_TOP_K
            cov = (rel / r_star_size) if r_star_size > 0 else 1.0
            per_b[b] = {"rel": rel, "noise": noise, "cov": cov}

        cov_pair = per_b[1]["cov"]
        noise_pair = per_b[1]["noise"]
        for b in BATCH_SIZES:
            cov_batch = per_b[b]["cov"]
            noise_batch = per_b[b]["noise"]
            rows.append({
                "dataset": args.dataset,
                "batch_size": b,
                "pair_id": pid,
                "ground_truth": int(pairs.loc[pid, "ground_truth"]),
                "R_star_size": r_star_size,
                "Rel_batch": per_b[b]["rel"],
                "cov_pair": cov_pair,
                "cov_batch": cov_batch,
                "noise_pair": noise_pair,
                "noise_batch": noise_batch,
                "delta_cov": max(0.0, cov_pair - cov_batch),
                "delta_noise": max(0.0, noise_batch - noise_pair),
            })

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(metrics_path, index=False)
    logger.info("Wrote %d metric rows -> %s", len(metrics_df), metrics_path)

    summary = metrics_df.groupby("batch_size")[["delta_cov", "delta_noise"]].mean()
    logger.info("Mean degradation by batch size:\n%s", summary.to_string())
    print(f"\n=== {args.dataset}: mean Δcov(b) / Δnoise(b) ===")
    print(summary.to_string())


if __name__ == "__main__":
    main()

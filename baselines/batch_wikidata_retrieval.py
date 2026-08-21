"""
Batch retrieval: for every batch produced by generate_batches.py, concatenate all
its pair queries ("What are {record_1}, and {record_2}?") into a single query string
and retrieve top-K Wikidata items/properties once per batch (mirrors batch_retrieval.py's
block-level batching, but reads the QG batch files under batch_outputs/).

Usage:
    python batch_wikidata_retrieval.py --dataset beer --batch-size 6
"""

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from wikidata_entity_retrieval import (
    API_BASE,
    make_session,
    query_endpoint,
    atomic_write_json,
)


def build_batch_queries(batches_csv: str) -> dict:
    df = pd.read_csv(batches_csv)
    batch_queries = {}
    for batch_id, group in df.groupby("batch_id"):
        pair_queries = [f"What are {row.record_1}, and {row.record_2}?" for row in group.itertuples()]
        batch_queries[str(batch_id)] = {
            "pair_ids": group["pair_id"].tolist(),
            "query_text": " ; ".join(pair_queries),
        }
    return batch_queries


def process_batch(batch_id, info, session, args, logger):
    query_text = info["query_text"]
    if len(query_text) > args.max_query_chars:
        query_text = query_text[: args.max_query_chars]
    qids = query_endpoint(session, "item", query_text, args, logger, f"batch_{batch_id}/item")
    pids = query_endpoint(session, "property", query_text, args, logger, f"batch_{batch_id}/property")
    return {
        "pair_ids": info["pair_ids"],
        "query_text": query_text,
        "relevant_qids": qids,
        "relevant_pids": pids,
    }


def main():
    parser = argparse.ArgumentParser(description="Retrieve top-K Wikidata items/properties per batch of pairs.")
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("-p", "--partition", default="test")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--batches-dir", default="batch_outputs")
    parser.add_argument("--output-dir", default="batch_retrieval_outputs")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-base-delay", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--max-query-chars", type=int, default=3000,
                         help="Truncate the concatenated batch query text to this many characters "
                              "(the API returns HTTP 414 for very long query strings).")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger(f"{args.dataset}_batch_retrieval")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(f"logs/{args.dataset}_{args.partition}_QG_{args.batch_size}_batch_retrieval.log")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    batches_csv = os.path.join(
        args.batches_dir, args.dataset, f"{args.dataset}_{args.partition}_QG_{args.batch_size}_batches.csv"
    )
    out_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(
        out_dir, f"{args.dataset}_{args.partition}_QG_{args.batch_size}_batch_retrieval.json"
    )

    logger.info("Building batch queries from %s", batches_csv)
    batch_queries = build_batch_queries(batches_csv)
    logger.info("Total batches: %d", len(batch_queries))

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            output = json.load(f)
        logger.info("Resuming from existing output at %s", output_path)
    else:
        output = {
            "metadata": {
                "dataset": args.dataset,
                "partition": args.partition,
                "batch_size": args.batch_size,
                "api_base": API_BASE,
                "top_k": args.top_k,
                "lang": args.lang,
                "rerank": False,
            },
            "batches": {},
        }

    pending_ids = [b for b in batch_queries if "error" in output["batches"].get(b, {"error": True})]
    logger.info(
        "Batches already completed: %d, pending: %d",
        len(batch_queries) - len(pending_ids), len(pending_ids),
    )

    session = make_session(args.workers)

    since_last_checkpoint = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_batch, batch_id, batch_queries[batch_id], session, args, logger): batch_id
            for batch_id in pending_ids
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{args.dataset}/QG_{args.batch_size}"):
            batch_id = futures[future]
            try:
                result = future.result()
                output["batches"][batch_id] = result
            except Exception as exc:
                logger.error("Batch %s failed: %s", batch_id, exc)
                output["batches"][batch_id] = {
                    "pair_ids": batch_queries[batch_id]["pair_ids"],
                    "error": str(exc),
                }

            since_last_checkpoint += 1
            if since_last_checkpoint >= args.checkpoint_every:
                atomic_write_json(output_path, output)
                since_last_checkpoint = 0

    output["metadata"]["total_batches"] = len(batch_queries)
    output["metadata"]["num_success"] = sum(1 for v in output["batches"].values() if "error" not in v)
    output["metadata"]["num_failed"] = sum(1 for v in output["batches"].values() if "error" in v)
    atomic_write_json(output_path, output)

    logger.info(
        "Done. success=%d failed=%d total=%d. Output: %s",
        output["metadata"]["num_success"], output["metadata"]["num_failed"], len(batch_queries), output_path,
    )
    if output["metadata"]["num_failed"] > 0:
        logger.warning("Some batches failed. Re-run the same command to retry only the failed ones.")


if __name__ == "__main__":
    main()

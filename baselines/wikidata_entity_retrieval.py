"""
Baseline retrieval: for every test-set entity of a dataset, query the Wikidata
Vector Search service (https://wd-vectordb.wmcloud.org) for the top-K most
relevant Wikidata items (QIDs) and properties (PIDs).

Usage:
    python wikidata_entity_retrieval.py --dataset abt
"""

import argparse
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm

API_BASE = "https://wd-vectordb.wmcloud.org"
USER_AGENT = "CE-RAG4EM-EntityRetrieval/1.0 (research use; contact: YOUR_EMAIL_ADDR)"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

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


def entity_to_text(entity_row: pd.Series, with_semantic: bool = True) -> str:
    parts = []
    for col_name, value in entity_row.items():
        if col_name == "id":
            continue
        text_value = str(value) if pd.notnull(value) else "nan"
        if with_semantic:
            parts.append(f"{col_name}: {text_value}")
        else:
            parts.append(text_value)
    return "; ".join(parts)


def build_entities(raw_dir: str, partition: str) -> dict:
    pairs = pd.read_csv(os.path.join(raw_dir, f"{partition}.csv"))
    table_a = pd.read_csv(os.path.join(raw_dir, "tableA.csv")).set_index("id")
    table_b = pd.read_csv(os.path.join(raw_dir, "tableB.csv")).set_index("id")

    entities = {}
    for eid in sorted(pairs["ltable_id"].unique().tolist()):
        entities[f"tableA_{eid}"] = {"table": "A", "id": int(eid), "row": table_a.loc[eid]}
    for eid in sorted(pairs["rtable_id"].unique().tolist()):
        entities[f"tableB_{eid}"] = {"table": "B", "id": int(eid), "row": table_b.loc[eid]}
    return entities


def make_session(pool_size: int) -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def query_endpoint(session, endpoint, query_text, args, logger, label):
    url = f"{API_BASE}/{endpoint}/query/"
    params = {"query": query_text, "lang": args.lang, "K": args.top_k, "rerank": "false"}
    last_exc = None
    for attempt in range(1, args.max_retries + 2):
        try:
            resp = session.get(url, params=params, timeout=args.timeout)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                 requests.exceptions.ChunkedEncodingError) as exc:
            last_exc = exc
        else:
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in RETRYABLE_STATUS:
                last_exc = requests.exceptions.HTTPError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            else:
                raise requests.exceptions.HTTPError(
                    f"HTTP {resp.status_code} (non-retryable): {resp.text[:200]}"
                )
        if attempt <= args.max_retries:
            delay = min(args.retry_base_delay * (2 ** (attempt - 1)) + random.uniform(0, args.retry_base_delay), 30.0)
            logger.warning("[%s] attempt %d failed (%s); retrying in %.1fs", label, attempt, last_exc, delay)
            time.sleep(delay)
    raise last_exc


def process_entity(key, info, session, args, logger):
    query_text = entity_to_text(info["row"], with_semantic=True)
    if len(query_text) > args.max_query_chars:
        query_text = query_text[: args.max_query_chars]
    qids = query_endpoint(session, "item", query_text, args, logger, f"{key}/item")
    pids = query_endpoint(session, "property", query_text, args, logger, f"{key}/property")
    return {
        "table": info["table"],
        "id": info["id"],
        "query_text": query_text,
        "relevant_qids": qids,
        "relevant_pids": pids,
    }


def atomic_write_json(path: str, data: dict):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def main():
    parser = argparse.ArgumentParser(description="Retrieve top-K Wikidata items/properties per test-set entity.")
    parser.add_argument("-d", "--dataset", required=True, choices=sorted(DATASET_DIR_MAP.keys()))
    parser.add_argument("-p", "--partition", default="test")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="retrieval_outputs")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--lang", default="en")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-base-delay", type=float, default=1.0)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--max-query-chars", type=int, default=3000,
                         help="Truncate serialized entity text to this many characters before querying "
                              "(the API returns HTTP 414 for very long query strings).")
    parser.add_argument("--extra-retry-rounds", type=int, default=3,
                         help="Extra full retry rounds over still-failed entities after the main pass.")
    parser.add_argument("--retry-round-wait", type=float, default=90.0,
                         help="Seconds to wait before each extra retry round, to let rate limits reset.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N pending entities (for testing).")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger(args.dataset)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(f"logs/{args.dataset}_{args.partition}_wikidata_retrieval.log")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    raw_dir = os.path.join(args.raw_dir, DATASET_DIR_MAP[args.dataset])
    out_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f"{args.dataset}_{args.partition}_entity_retrieval.json")

    logger.info("Building entity set for dataset=%s partition=%s from %s", args.dataset, args.partition, raw_dir)
    entities = build_entities(raw_dir, args.partition)
    logger.info("Total unique entities: %d", len(entities))

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            output = json.load(f)
        logger.info("Resuming from existing output at %s", output_path)
    else:
        output = {
            "metadata": {
                "dataset": args.dataset,
                "dataset_dir": DATASET_DIR_MAP[args.dataset],
                "partition": args.partition,
                "api_base": API_BASE,
                "top_k": args.top_k,
                "lang": args.lang,
                "rerank": False,
            },
            "entities": {},
        }

    pending_keys = [k for k in entities if "error" in output["entities"].get(k, {"error": True})]
    completed_count = len(entities) - len(pending_keys)
    if args.limit is not None:
        pending_keys = pending_keys[: args.limit]
    logger.info(
        "Entities already completed: %d, pending: %d (processing %d this run)",
        completed_count, len(entities) - completed_count, len(pending_keys),
    )

    session = make_session(args.workers)

    def run_pass(keys, desc):
        since_last_checkpoint = 0
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(process_entity, key, entities[key], session, args, logger): key
                for key in keys
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                key = futures[future]
                try:
                    result = future.result()
                    output["entities"][key] = result
                except Exception as exc:
                    logger.error("Entity %s failed this round: %s", key, exc)
                    output["entities"][key] = {
                        "table": entities[key]["table"],
                        "id": entities[key]["id"],
                        "error": str(exc),
                    }

                since_last_checkpoint += 1
                if since_last_checkpoint >= args.checkpoint_every:
                    atomic_write_json(output_path, output)
                    since_last_checkpoint = 0

    run_pass(pending_keys, f"{args.dataset}/{args.partition}")

    for round_num in range(1, args.extra_retry_rounds + 1):
        still_failed = [k for k in pending_keys if "error" in output["entities"].get(k, {})]
        if not still_failed:
            break
        logger.warning(
            "Round %d: %d entities still failed, waiting %.0fs before retrying them",
            round_num, len(still_failed), args.retry_round_wait,
        )
        atomic_write_json(output_path, output)
        time.sleep(args.retry_round_wait)
        run_pass(still_failed, f"{args.dataset}/{args.partition} retry-round-{round_num}")

    success_count = sum(1 for k in pending_keys if "error" not in output["entities"].get(k, {}))
    fail_count = len(pending_keys) - success_count

    output["metadata"]["total_entities"] = len(entities)
    output["metadata"]["num_success"] = sum(1 for v in output["entities"].values() if "error" not in v)
    output["metadata"]["num_failed"] = sum(1 for v in output["entities"].values() if "error" in v)
    atomic_write_json(output_path, output)

    logger.info(
        "Done. This run: success=%d failed=%d. Overall: success=%d failed=%d total=%d. Output: %s",
        success_count, fail_count,
        output["metadata"]["num_success"], output["metadata"]["num_failed"], len(entities),
        output_path,
    )
    if output["metadata"]["num_failed"] > 0:
        logger.warning("Some entities failed. Re-run the same command to retry only the failed ones.")


if __name__ == "__main__":
    main()

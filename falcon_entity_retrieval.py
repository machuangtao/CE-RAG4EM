"""
Baseline retrieval: for every test-set entity of a dataset, query the Falcon 2.0
entity/relation linking service (https://labs.tib.eu/falcon/falcon2) for the
Wikidata items (QIDs) and properties (PIDs) mentioned in the entity's text.

Falcon 2.0 spots named-entity-like spans, so long free-text attributes (e.g.
"description") are excluded by default and the query text is truncated, since
long inputs make the service slow or return a 502.

Usage:
    python falcon_entity_retrieval.py --dataset abt
"""

import argparse
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm

API_URL = "https://labs.tib.eu/falcon/falcon2/api"
USER_AGENT = "CE-RAG4EM-EntityRetrieval/1.0 (research use; contact: xxx)"
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


def entity_to_text(entity_row: pd.Series, exclude_cols: set, max_chars: int) -> str:
    parts = []
    for col_name, value in entity_row.items():
        if col_name == "id" or col_name.lower() in exclude_cols:
            continue
        text_value = str(value) if pd.notnull(value) else "nan"
        parts.append(text_value)
    text = " ".join(parts)
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars]
    return text


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
    session.headers.update({"User-Agent": USER_AGENT, "Content-Type": "application/json"})
    adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def query_falcon(session, query_text, args, logger, label):
    params = {"mode": args.mode, "k": args.top_k}
    if args.db:
        params["db"] = 1
    payload = json.dumps({"text": query_text})
    last_exc = None
    for attempt in range(1, args.max_retries + 2):
        try:
            resp = session.post(API_URL, params=params, data=payload, timeout=args.timeout)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout,
                 requests.exceptions.ChunkedEncodingError) as exc:
            last_exc = exc
        else:
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError as exc:
                    last_exc = exc
            elif resp.status_code in RETRYABLE_STATUS:
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


def parse_response(resp_json, logger, label):
    entities = resp_json.get("entities_wikidata")
    if entities is None:
        logger.warning("[%s] unexpected response shape, treating as empty: %s", label, str(resp_json)[:200])
        entities = []
    relations = resp_json.get("relations_wikidata", [])

    qids = [
        {
            "qid": e.get("URI", "").rstrip("/").rsplit("/", 1)[-1],
            "uri": e.get("URI"),
            "surface_form": e.get("surface form"),
        }
        for e in entities
    ]
    pids = [
        {
            "pid": r.get("URI", "").rstrip("/").rsplit("/", 1)[-1],
            "uri": r.get("URI"),
            "surface_form": r.get("surface form"),
        }
        for r in relations
    ]
    return qids, pids


def process_entity(key, info, session, args, logger):
    query_text = entity_to_text(info["row"], args.exclude_cols_set, args.max_chars)
    if not query_text.strip():
        return {
            "table": info["table"],
            "id": info["id"],
            "query_text": query_text,
            "relevant_qids": [],
            "relevant_pids": [],
        }
    resp_json = query_falcon(session, query_text, args, logger, key)
    qids, pids = parse_response(resp_json, logger, key)
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


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description="Retrieve Wikidata items/properties per test-set entity via Falcon 2.0.")
    parser.add_argument("-d", "--dataset", required=True, choices=sorted(DATASET_DIR_MAP.keys()))
    parser.add_argument("-p", "--partition", default="test")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="retrieval_outputs")
    parser.add_argument("--top-k", type=int, default=5, help="Falcon's 'k' param: top candidates per detected mention.")
    parser.add_argument("--mode", default="long", choices=["long", "short"],
                         help="short returns only entities; long returns entities and relations.")
    parser.add_argument("--db", action="store_true", help="Also request DBpedia results (ignored in output).")
    parser.add_argument("--exclude-cols", default="description",
                         help="Comma-separated column names to drop from the query text (case-insensitive). "
                              "Falcon is a NER-style tagger and struggles with long free-text fields.")
    parser.add_argument("--max-chars", type=int, default=300,
                         help="Truncate the concatenated query text to this many characters.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=75.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--retry-base-delay", type=float, default=2.0)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--extra-retry-rounds", type=int, default=3,
                         help="Extra full retry rounds over still-failed entities after the main pass.")
    parser.add_argument("--retry-round-wait", type=float, default=90.0,
                         help="Seconds to wait before each extra retry round, to let rate limits reset.")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N pending entities (for testing).")
    args = parser.parse_args()
    args.exclude_cols_set = {c.strip().lower() for c in args.exclude_cols.split(",") if c.strip()}

    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger(args.dataset)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(f"logs/{args.dataset}_{args.partition}_falcon_retrieval.log")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    raw_dir = os.path.join(args.raw_dir, DATASET_DIR_MAP[args.dataset])
    out_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    output_path = os.path.join(out_dir, f"{args.dataset}_{args.partition}_falcon_retrieval.json")

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
                "api_url": API_URL,
                "mode": args.mode,
                "top_k": args.top_k,
                "exclude_cols": sorted(args.exclude_cols_set),
                "max_chars": args.max_chars,
                "created_at": now_iso(),
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
                    output["metadata"]["last_updated"] = now_iso()
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

    output["metadata"]["last_updated"] = now_iso()
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

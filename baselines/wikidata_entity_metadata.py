"""
Collect every unique QID/PID referenced in retrieval_outputs/*/*_test_entity_retrieval.json
and fetch each one's English label + description from the official Wikidata API,
storing everything in one unified JSON file.

Usage:
    python wikidata_entity_metadata.py
"""

import argparse
import glob
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm

API_URL = "https://www.wikidata.org/w/api.php"
USER_AGENT = "CE-RAG4EM-EntityRetrieval/1.0 (research use; contact: YOUR_EMAIL_ADDR)"
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def collect_ids(input_glob: str):
    qids, pids = set(), set()
    files = sorted(set().union(*(glob.glob(p.strip()) for p in input_glob.split(","))))
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        for v in d.get("entities", {}).values():
            for q in (v.get("relevant_qids") or []):
                qids.add(q["QID"])
            for p in (v.get("relevant_pids") or []):
                pids.add(p["PID"])
    return files, qids, pids


def make_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    adapter = HTTPAdapter(pool_connections=32, pool_maxsize=32)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def request_with_retry(session, params, args, logger, label):
    last_exc = None
    for attempt in range(1, args.max_retries + 2):
        try:
            resp = session.get(API_URL, params=params, timeout=args.timeout)
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
            delay = min(1.0 * (2 ** (attempt - 1)) + random.uniform(0, 1.0), 30.0)
            logger.warning("[%s] attempt %d failed (%s); retrying in %.1fs", label, attempt, last_exc, delay)
            time.sleep(delay)
    raise last_exc


def fetch_batch(session, ids, args, logger):
    """Fetch label/description for a batch of ids. Handles the API's
    all-or-nothing error behavior when one id in the batch doesn't exist."""
    remaining = list(ids)
    results = {}
    while remaining:
        params = {
            "action": "wbgetentities",
            "ids": "|".join(remaining),
            "props": "labels|descriptions",
            "languages": args.lang,
            "format": "json",
        }
        data = request_with_retry(session, params, args, logger, remaining[0])
        if "error" in data:
            err = data["error"]
            bad_id = err.get("id")
            if err.get("code") == "no-such-entity" and bad_id in remaining:
                remaining.remove(bad_id)
                results[bad_id] = {"type": None, "label": None, "description": None, "not_found": True}
                continue
            raise RuntimeError(f"Unhandled API error for batch starting {remaining[0]}: {err}")

        entities = data.get("entities", {})
        for eid in remaining:
            edata = entities.get(eid)
            if edata is None or "missing" in edata:
                results[eid] = {"type": None, "label": None, "description": None, "not_found": True}
                continue
            label = edata.get("labels", {}).get(args.lang, {}).get("value")
            desc = edata.get("descriptions", {}).get(args.lang, {}).get("value")
            results[eid] = {"type": edata.get("type"), "label": label, "description": desc}
        remaining = []
    return results


def atomic_write_json(path: str, data: dict):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def chunk(seq, size):
    seq = list(seq)
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def main():
    parser = argparse.ArgumentParser(description="Fetch Wikidata labels/descriptions for retrieved QIDs/PIDs.")
    parser.add_argument(
        "--input-glob",
        default="retrieval_outputs/*/*_test_entity_retrieval.json,retrieval_outputs/*/*_test_pair_retrieval.json",
        help="Comma-separated glob patterns.",
    )
    parser.add_argument("--output", default="retrieval_outputs/wikidata_entity_metadata.json")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--checkpoint-every", type=int, default=20, help="Checkpoint every N batches.")
    parser.add_argument("--limit", type=int, default=None, help="Only process first N pending ids (for testing).")
    args = parser.parse_args()

    os.makedirs("logs", exist_ok=True)
    logger = logging.getLogger("wikidata_metadata")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler("logs/wikidata_entity_metadata.log")
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    files, qids, pids = collect_ids(args.input_glob)
    all_ids = sorted(qids) + sorted(pids)
    logger.info("Scanned %d files, found %d unique QIDs and %d unique PIDs (%d total)",
                len(files), len(qids), len(pids), len(all_ids))

    if os.path.exists(args.output):
        with open(args.output, "r", encoding="utf-8") as f:
            output = json.load(f)
        logger.info("Resuming from existing output at %s", args.output)
    else:
        output = {
            "metadata": {
                "lang": args.lang,
                "source_files": files,
            },
            "entities": {},
        }

    pending_ids = [i for i in all_ids if output["entities"].get(i, {"error": True}).get("error")]
    already_fetched = len(all_ids) - len(pending_ids)
    if args.limit is not None:
        pending_ids = pending_ids[: args.limit]
    logger.info("Already fetched: %d, pending: %d (processing %d this run)",
                already_fetched, len(all_ids) - already_fetched, len(pending_ids))

    session = make_session()
    batches = list(chunk(pending_ids, args.batch_size))
    since_checkpoint = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch_batch, session, b, args, logger): b for b in batches}
        for future in tqdm(as_completed(futures), total=len(futures), desc="wikidata_metadata"):
            batch = futures[future]
            try:
                result = future.result()
                output["entities"].update(result)
            except Exception as exc:
                logger.error("Batch starting with %s failed permanently: %s", batch[0], exc)
                for eid in batch:
                    if eid not in output["entities"]:
                        output["entities"][eid] = {"type": None, "label": None, "description": None, "error": str(exc)}

            since_checkpoint += 1
            if since_checkpoint >= args.checkpoint_every:
                atomic_write_json(args.output, output)
                since_checkpoint = 0

    output["metadata"]["total_ids"] = len(all_ids)
    output["metadata"]["num_found"] = sum(
        1 for v in output["entities"].values() if not v.get("not_found") and not v.get("error")
    )
    output["metadata"]["num_not_found"] = sum(1 for v in output["entities"].values() if v.get("not_found"))
    output["metadata"]["num_error"] = sum(1 for v in output["entities"].values() if v.get("error"))
    atomic_write_json(args.output, output)

    logger.info(
        "Done. total=%d found=%d not_found=%d error=%d. Output: %s",
        output["metadata"]["total_ids"], output["metadata"]["num_found"],
        output["metadata"]["num_not_found"], output["metadata"]["num_error"], args.output,
    )
    if output["metadata"]["num_error"] > 0:
        logger.warning("Some batches failed. Re-run the same command to retry only the missing ids.")


if __name__ == "__main__":
    main()

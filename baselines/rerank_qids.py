"""
Rerank each retrieval record's candidate QIDs with a cross-encoder, so that
prepare_prompts.py can pick a better top-2 instead of just the first 2 as
returned by the retrieval API.

Works generically over any retrieval_outputs/<dataset>/<dataset>_<partition>_<suffix>.json
file that has the shape {"entities": {key: {"query_text": ..., "relevant_qids": [...]}}} --
this covers the per-entity falcon/wikidata retrieval files as well as the per-pair
test_pair_retrieval.json files (same schema, "query_text" is just the pair's combined query).

For each record:
  - query = record["query_text"]
  - candidates = record["relevant_qids"], each resolved via the metadata file to
    "label: description" (or just "label" if no description). Candidates whose QID
    isn't found in the metadata file are dropped (they can't be shown in a prompt anyway).
  - all (query, candidate) pairs are scored in batches with the cross-encoder, and
    relevant_qids is rewritten sorted by score descending, with a "rerank_score" field added.

Usage:
    python rerank_qids.py --dataset beer --source wikidata
"""

import argparse
import json
import os
from datetime import datetime, timezone

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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

RETRIEVAL_SOURCES = {
    "falcon": {
        "retrieval_suffix": "falcon_retrieval",
        "metadata_path": "retrieval_outputs/falcon_entity_metadata.json",
        "qid_key": "qid",
    },
    "wikidata": {
        "retrieval_suffix": "entity_retrieval",
        "metadata_path": "retrieval_outputs/wikidata_entity_metadata.json",
        "qid_key": "QID",
    },
    "pair": {
        "retrieval_suffix": "pair_retrieval",
        # pair-level retrieval draws from the same Wikidata vector index as the
        # entity-level "wikidata" source, so its QIDs/PIDs are already fully
        # covered there (verified: 100% of pair QIDs/PIDs found in this file).
        "metadata_path": "retrieval_outputs/wikidata_entity_metadata.json",
        "qid_key": "QID",
    },
}


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def candidate_text(meta: dict) -> str:
    label = meta.get("label") or ""
    description = meta.get("description")
    if description and description != label:
        return f"{label}: {description}"
    return label


def score_pairs(model, tokenizer, device, queries, texts, batch_size, max_length):
    scores = []
    with torch.no_grad():
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i : i + batch_size]
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(
                batch_queries, batch_texts,
                padding=True, truncation=True, max_length=max_length,
                return_tensors="pt",
            ).to(device)
            logits = model(**inputs).logits.squeeze(-1)
            scores.extend(logits.float().cpu().tolist())
    return scores


def rerank_dataset_source(dataset, source, partition, model, tokenizer, device, batch_size, max_length):
    cfg = RETRIEVAL_SOURCES[source]
    qid_key = cfg["qid_key"]
    retrieval_path = os.path.join(
        "retrieval_outputs", dataset, f"{dataset}_{partition}_{cfg['retrieval_suffix']}.json"
    )
    data = load_json(retrieval_path)
    metadata = load_json(cfg["metadata_path"]).get("entities", {})

    # Flatten all (record_key, candidate_index, query, candidate_text) tuples across the
    # whole dataset so the cross-encoder can be run in large GPU batches, not per-record.
    flat_keys, flat_queries, flat_texts = [], [], []
    kept_candidates = {}  # record_key -> list of candidate dicts (metadata-resolved subset)

    for key, record in data["entities"].items():
        query = record.get("query_text") or ""
        kept = []
        for cand in record.get("relevant_qids") or []:
            qid = cand.get(qid_key)
            meta = metadata.get(qid)
            if not meta or meta.get("not_found") or meta.get("error"):
                continue
            text = candidate_text(meta)
            if not text:
                continue
            kept.append(dict(cand))
            flat_keys.append(key)
            flat_queries.append(query)
            flat_texts.append(text)
        kept_candidates[key] = kept

    print(f"[{dataset}/{source}] scoring {len(flat_queries)} (query, candidate) pairs "
          f"across {len(data['entities'])} records...")
    scores = score_pairs(model, tokenizer, device, flat_queries, flat_texts, batch_size, max_length)

    # Distribute scores back and re-sort each record's candidates.
    idx_by_key = {}
    for i, key in enumerate(flat_keys):
        idx_by_key.setdefault(key, []).append(i)

    for key, record in data["entities"].items():
        candidates = kept_candidates[key]
        indices = idx_by_key.get(key, [])
        for cand, idx in zip(candidates, indices):
            cand["rerank_score"] = scores[idx]
        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
        record["relevant_qids"] = candidates

    data["metadata"]["reranker"] = {
        "model": model.name_or_path,
        "query_field": "query_text",
        "candidate_field": "label: description",
        "reranked_at": datetime.now(timezone.utc).isoformat(),
    }

    out_path = os.path.join(
        "retrieval_outputs", dataset, f"{dataset}_{partition}_{cfg['retrieval_suffix']}_reranked.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[{dataset}/{source}] wrote {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Rerank retrieval candidates with a cross-encoder.")
    parser.add_argument("-d", "--dataset", default="all", choices=sorted(DATASET_DIR_MAP.keys()) + ["all"])
    parser.add_argument("-p", "--partition", default="test")
    parser.add_argument("--source", default="both", choices=["falcon", "wikidata", "pair", "both"])
    parser.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L4-v2")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(device).eval()

    datasets = list(DATASET_DIR_MAP.keys()) if args.dataset == "all" else [args.dataset]
    sources = ["falcon", "wikidata"] if args.source == "both" else [args.source]

    for dataset in datasets:
        for source in sources:
            rerank_dataset_source(
                dataset, source, args.partition, model, tokenizer, device,
                args.batch_size, args.max_length,
            )


if __name__ == "__main__":
    main()

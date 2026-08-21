"""
Build entity-matching (RAG) prompts for the Falcon and Wikidata Vector Search
retrieval baselines and save them to disk, so they can later be processed
uniformly via the OpenAI API.

For each test-set entity pair, the "Additional Information" section is built
from the top-2 retrieved QIDs of entity A and the top-2 retrieved QIDs of
entity B (at most 4 entries total), each resolved to its label/description
via the corresponding *_entity_metadata.json file.

Usage:
    python prepare_prompts.py --dataset beer
    python prepare_prompts.py --dataset all --source both
"""

import argparse
import json
import os

import pandas as pd

from data_utils.constants import PROMPT_TEMPLATES

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
        # pair-level retrieval draws from the same Wikidata vector index as "wikidata",
        # so its QIDs/PIDs are already fully covered by this metadata file.
        "metadata_path": "retrieval_outputs/wikidata_entity_metadata.json",
        "qid_key": "QID",
    },
}

TOP_K_PER_ENTITY = 2
NO_CONTEXT_FALLBACK = "No avaliable relevant knowledge, please make the decision on your own."


def row_to_entity_text(row: pd.Series) -> str:
    parts = []
    for col_name, value in row.items():
        if col_name == "id":
            continue
        text_value = str(value) if pd.notnull(value) else "nan"
        parts.append(f"{col_name}: {text_value}")
    return "; ".join(parts)


def format_qid_entry(qid: str, meta: dict) -> str:
    label = meta.get("label") or qid
    description = meta.get("description")
    if description and description != label and not description.startswith(f"Wikidata entity {qid}"):
        return f"{qid} ({label}: {description})"
    return f"{qid} ({label})"


def top_qid_context(entity_key: str, retrieval_entities: dict, qid_key: str, metadata: dict, top_k: int) -> list:
    entry = retrieval_entities.get(entity_key, {})
    relevant_qids = (entry.get("relevant_qids") or [])[:top_k]
    formatted = []
    for item in relevant_qids:
        qid = item.get(qid_key)
        if not qid:
            continue
        meta = metadata.get(qid)
        if not meta or meta.get("not_found") or meta.get("error"):
            continue
        formatted.append(format_qid_entry(qid, meta))
    return formatted


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_prompts_for_dataset(dataset: str, source: str, partition: str, raw_dir_base: str, output_dir_base: str,
                               reranked: bool = False, top_k_per_entity: int = TOP_K_PER_ENTITY):
    cfg = RETRIEVAL_SOURCES[source]
    dataset_dir = DATASET_DIR_MAP[dataset]
    raw_dir = os.path.join(raw_dir_base, dataset_dir)

    retrieval_suffix = cfg["retrieval_suffix"] + ("_reranked" if reranked else "")
    retrieval_path = os.path.join(
        "retrieval_outputs", dataset, f"{dataset}_{partition}_{retrieval_suffix}.json"
    )
    retrieval_entities = load_json(retrieval_path).get("entities", {})
    metadata = load_json(cfg["metadata_path"]).get("entities", {})

    pairs = pd.read_csv(os.path.join(raw_dir, f"{partition}.csv"))
    table_a = pd.read_csv(os.path.join(raw_dir, "tableA.csv")).set_index("id")
    table_b = pd.read_csv(os.path.join(raw_dir, "tableB.csv")).set_index("id")

    template = PROMPT_TEMPLATES["rag4em"]["user"]

    records = []
    missing_context_count = 0
    for i, pair in pairs.iterrows():
        ltable_id = int(pair["ltable_id"])
        rtable_id = int(pair["rtable_id"])
        key_a = f"tableA_{ltable_id}"
        key_b = f"tableB_{rtable_id}"

        entity_a_text = row_to_entity_text(table_a.loc[ltable_id])
        entity_b_text = row_to_entity_text(table_b.loc[rtable_id])

        if source == "pair":
            # One shared query/candidate list per pair (not per entity): take its top-k directly.
            pair_key = f"{ltable_id}-{rtable_id}"
            context_parts = top_qid_context(pair_key, retrieval_entities, cfg["qid_key"], metadata, top_k_per_entity)
        else:
            context_parts = top_qid_context(key_a, retrieval_entities, cfg["qid_key"], metadata, top_k_per_entity)
            context_parts += top_qid_context(key_b, retrieval_entities, cfg["qid_key"], metadata, top_k_per_entity)

        if context_parts:
            context_text = "; ".join(context_parts)
        else:
            context_text = NO_CONTEXT_FALLBACK
            missing_context_count += 1

        content = template.format(entity_a_text, entity_b_text, context_text)
        records.append({
            "pair_index": int(i),
            "ltable_id": ltable_id,
            "rtable_id": rtable_id,
            "label": int(pair["label"]),
            "entity_a_key": key_a,
            "entity_b_key": key_b,
            "messages": [{"role": "user", "content": content}],
        })

    out_dir = os.path.join(output_dir_base, dataset)
    os.makedirs(out_dir, exist_ok=True)
    top_k_tag = f"_top{top_k_per_entity}" if top_k_per_entity != TOP_K_PER_ENTITY else ""
    source_tag = source + top_k_tag + ("_reranked" if reranked else "")
    out_path = os.path.join(out_dir, f"{dataset}_{partition}_{source_tag}_prompts.json")
    output = {
        "metadata": {
            "dataset": dataset,
            "dataset_dir": dataset_dir,
            "partition": partition,
            "source": source,
            "reranked": reranked,
            "prompt_template": "rag4em",
            "top_k_per_entity": top_k_per_entity,
            "num_pairs": len(records),
            "num_pairs_without_context": missing_context_count,
        },
        "prompts": records,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    return out_path, len(records), missing_context_count


def main():
    parser = argparse.ArgumentParser(
        description="Build and save RAG entity-matching prompts for the falcon/wikidata retrieval baselines."
    )
    parser.add_argument("-d", "--dataset", default="all", choices=sorted(DATASET_DIR_MAP.keys()) + ["all"])
    parser.add_argument("-p", "--partition", default="test")
    parser.add_argument("--source", default="both", choices=["falcon", "wikidata", "pair", "both"])
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="prompts_outputs")
    parser.add_argument("--reranked", action="store_true",
                         help="Read the cross-encoder-reranked retrieval files (produced by rerank_qids.py) "
                              "instead of the raw retrieval files.")
    parser.add_argument("--top-k-per-entity", type=int, default=TOP_K_PER_ENTITY,
                         help=f"How many candidate QIDs to keep per entity/pair (default: {TOP_K_PER_ENTITY}). "
                              "Non-default values get a '_topK' tag in the output filename so they "
                              "never overwrite the default-K prompts.")
    args = parser.parse_args()

    datasets = list(DATASET_DIR_MAP.keys()) if args.dataset == "all" else [args.dataset]
    sources = ["falcon", "wikidata"] if args.source == "both" else [args.source]

    for dataset in datasets:
        for source in sources:
            out_path, n, missing = build_prompts_for_dataset(
                dataset, source, args.partition, args.raw_dir, args.output_dir,
                args.reranked, args.top_k_per_entity,
            )
            print(f"[{dataset}/{source}] wrote {n} prompts ({missing} without retrieval context) -> {out_path}")


if __name__ == "__main__":
    main()

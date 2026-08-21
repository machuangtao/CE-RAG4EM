#!/usr/bin/env python3
"""Run direct per-query RAG on split bucket datasets.

This script consumes bucket CSVs produced by the similarity split script,
retrieves pair-level Wikidata context through the existing direct retrieval
pipeline, runs entity matching with an RAG prompt, and reports both matching
metrics and retrieval coverage for each bucket.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model_utils import process_requests


def _load_constants_module() -> object:
    constants_path = REPO_ROOT / "data_utils" / "constants.py"
    spec = importlib.util.spec_from_file_location("rag_em_constants", constants_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load constants module from {constants_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CONSTANTS = _load_constants_module()
BUCKET_NAMES = ["easy_non_match", "ambiguous", "easy_match"]


def _parse_binary_decision(text: str) -> int:
    lowered = (text or "").strip().lower()
    decision_match = re.search(r"match decision\s*[:\-]?\s*(yes|no)", lowered)
    if decision_match:
        return 1 if decision_match.group(1) == "yes" else 0

    yes_positions = [match.start() for match in re.finditer(r"\byes\b", lowered)]
    no_positions = [match.start() for match in re.finditer(r"\bno\b", lowered)]
    if not yes_positions and not no_positions:
        return 0

    last_yes = yes_positions[-1] if yes_positions else -1
    last_no = no_positions[-1] if no_positions else -1
    return 1 if last_yes > last_no else 0


def _load_prompt_template(prompt_name: str) -> str:
    prompt_templates = CONSTANTS.PROMPT_TEMPLATES
    if prompt_name not in prompt_templates:
        raise ValueError(f"Prompt '{prompt_name}' not found. Available prompts: {list(prompt_templates.keys())}")
    return prompt_templates[prompt_name]["user"]


def _resolve_bucket_csv_paths(dataset_key: str, bucket_csv_dir: Path) -> Dict[str, Path]:
    candidates = [
        bucket_csv_dir,
        REPO_ROOT / "output" / f"{dataset_key}_match_entity_records_split",
        REPO_ROOT / "output" / "match_entity_records_split",
    ]

    for candidate_dir in candidates:
        candidate_paths = {
            bucket_name: candidate_dir / f"{dataset_key}_{bucket_name}.csv"
            for bucket_name in BUCKET_NAMES
        }
        if all(path.exists() for path in candidate_paths.values()):
            return candidate_paths

    report_candidates = [
        bucket_csv_dir / f"{dataset_key}_test_match_entity_records_split.json",
        REPO_ROOT / "output" / f"{dataset_key}_match_entity_records_split" / f"{dataset_key}_test_match_entity_records_split.json",
        REPO_ROOT / "output" / "match_entity_records_split" / f"{dataset_key}_test_match_entity_records_split.json",
    ]
    for report_path in report_candidates:
        if not report_path.exists():
            continue
        with report_path.open("r", encoding="utf-8") as file_handle:
            report = json.load(file_handle)
        csv_paths = report.get("bucket_csv_paths", {})
        resolved_paths = {
            bucket_name: REPO_ROOT / str(csv_paths[bucket_name])
            for bucket_name in BUCKET_NAMES
            if bucket_name in csv_paths
        }
        if len(resolved_paths) == len(BUCKET_NAMES) and all(path.exists() for path in resolved_paths.values()):
            return resolved_paths

    searched = ", ".join(str(path) for path in candidates + [path.parent for path in report_candidates])
    raise FileNotFoundError(
        f"Could not locate split bucket CSVs for dataset '{dataset_key}'. Searched: {searched}"
    )


def _load_bucket_csvs(dataset_key: str, bucket_csv_dir: Path) -> Dict[str, pd.DataFrame]:
    bucket_csv_paths = _resolve_bucket_csv_paths(dataset_key, bucket_csv_dir)
    bucket_frames: Dict[str, pd.DataFrame] = {}
    for bucket_name in BUCKET_NAMES:
        csv_path = bucket_csv_paths[bucket_name]
        bucket_frame = pd.read_csv(csv_path)
        if "pair_id" not in bucket_frame.columns:
            bucket_frame["pair_id"] = bucket_frame.apply(
                lambda row: f"{row.ltable_id}-{row.rtable_id}",
                axis=1,
            )
        if "bucket" not in bucket_frame.columns:
            bucket_frame["bucket"] = bucket_name
        bucket_frames[bucket_name] = bucket_frame
    return bucket_frames


def _build_pair_query_frame(bucket_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat([bucket_frames[bucket_name] for bucket_name in BUCKET_NAMES], ignore_index=True)
    pair_query_df = combined[["pair_id", "entityA", "entityB"]].drop_duplicates().copy()
    pair_query_df["query"] = pair_query_df.apply(
        lambda row: f"What are {row.entityA}, and {row.entityB}?",
        axis=1,
    )
    return pair_query_df[["query", "pair_id"]]


def _merge_relevant_ids_with_info(
    relevant_ids_path: Path,
    pids_path: Path,
    qids_path: Path,
    triplets_path: Path,
) -> None:
    with relevant_ids_path.open("r", encoding="utf-8") as file_handle:
        relevant_ids = json.load(file_handle)
    with pids_path.open("r", encoding="utf-8") as file_handle:
        pids = json.load(file_handle)
    with qids_path.open("r", encoding="utf-8") as file_handle:
        qids = json.load(file_handle)
    with triplets_path.open("r", encoding="utf-8") as file_handle:
        triplets = json.load(file_handle)

    def enrich_item(item: Dict[str, object]) -> Dict[str, object]:
        key_type = "QID" if "QID" in item else "PID"
        id_value = str(item.get(key_type, ""))
        info = qids.get(id_value, {}) if key_type == "QID" else pids.get(id_value, {})
        label = info.get("label", "")
        description = info.get("description", "")
        pretty_list = triplets.get(id_value, [])
        pretty_str = "\n".join(entry.get("pretty_string", "") for entry in pretty_list)
        return {
            **item,
            "label": label,
            "description": description,
            "pretty_string": pretty_str,
        }

    for section_key in ["relevant_qids", "relevant_pids"]:
        for query_id, items in relevant_ids.get(section_key, {}).items():
            relevant_ids[section_key][query_id] = [enrich_item(item) for item in (items or [])]

    with relevant_ids_path.open("w", encoding="utf-8") as file_handle:
        json.dump(relevant_ids, file_handle, ensure_ascii=False, indent=2)


def _extract_ids_from_payload(payload: Dict[str, Dict[str, List[Dict[str, object]]]]) -> Tuple[List[str], List[str]]:
    pids = [
        str(item["PID"])
        for items in payload.get("relevant_pids", {}).values()
        if items is not None
        for item in items
        if isinstance(item, dict) and item.get("PID")
    ]
    qids = [
        str(item["QID"])
        for items in payload.get("relevant_qids", {}).values()
        if items is not None
        for item in items
        if isinstance(item, dict) and item.get("QID")
    ]
    return list(set(pids)), list(set(qids))


def _ensure_retrieval_enrichment(pair_path: Path, partition: str, query_dir: Path) -> None:
    from data_utils.wiki_query import fetch_and_save_triplets, fetch_and_save_wikidata

    with pair_path.open("r", encoding="utf-8") as file_handle:
        pair_retrieval = json.load(file_handle)

    pids, qids = _extract_ids_from_payload(pair_retrieval)
    pids_path = query_dir / f"pids_{partition}.json"
    qids_path = query_dir / f"qids_{partition}.json"
    triplets_path = query_dir / f"triplets_{partition}.json"

    if qids:
        fetch_and_save_wikidata(qids, str(qids_path))
    else:
        qids_path.write_text("{}", encoding="utf-8")

    if pids:
        fetch_and_save_wikidata(pids, str(pids_path))
    else:
        pids_path.write_text("{}", encoding="utf-8")

    ids = pids + qids
    if ids:
        fetch_and_save_triplets(ids, str(triplets_path))
    else:
        triplets_path.write_text("{}", encoding="utf-8")

    _merge_relevant_ids_with_info(pair_path, pids_path, qids_path, triplets_path)


def _ensure_pair_retrieval(bucket_frames: Dict[str, pd.DataFrame], partition: str, query_dir: Path) -> Path:
    pair_path = query_dir / f"pair_relevant_ids_{partition}.json"
    if pair_path.exists():
        _ensure_retrieval_enrichment(pair_path, partition, query_dir)
        return pair_path

    from data_utils.wiki_query import (
        fetch_and_save_relevant_ids,
        fetch_and_save_triplets,
        fetch_and_save_wikidata,
    )

    query_dir.mkdir(parents=True, exist_ok=True)
    pair_query_df = _build_pair_query_frame(bucket_frames)
    fetch_and_save_relevant_ids(pair_query_df, "pair_id", "query", str(pair_path))
    _ensure_retrieval_enrichment(pair_path, partition, query_dir)

    if not pair_path.exists():
        raise FileNotFoundError(f"Expected pair retrieval file was not created: {pair_path}")
    return pair_path


def _load_pair_retrieval(pair_path: Path) -> Dict[str, Dict[str, List[Dict[str, object]]]]:
    with pair_path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def _format_retrieval_item(item: Dict[str, object], item_type: str) -> str:
    id_key = "QID" if item_type == "qid" else "PID"
    identifier = str(item.get(id_key, ""))
    label = str(item.get("label", "") or "")
    description = str(item.get("description", "") or "")
    pretty_string = str(item.get("pretty_string", "") or "").strip()

    if label:
        prefix = "Wikidata entity" if item_type == "qid" else "Wikidata property"
        if description and description != label and not description.startswith(f"{prefix} {identifier}"):
            return f"{identifier} ({label}: {description})"
        return f"{identifier} ({label})"

    if pretty_string:
        return pretty_string

    return identifier


def _select_context_items(
    retrieval_payload: Dict[str, Dict[str, List[Dict[str, object]]]],
    pair_id: str,
    rag_type: str,
    top_k: int,
) -> Tuple[List[str], List[str], Dict[str, int]]:
    qid_context_items: List[str] = []
    pid_context_items: List[str] = []
    all_qid_items = retrieval_payload.get("relevant_qids", {}).get(pair_id, []) or []
    all_pid_items = retrieval_payload.get("relevant_pids", {}).get(pair_id, []) or []

    raw_qid_count = len(all_qid_items)
    raw_pid_count = len(all_pid_items)
    selected_qid_count = 0
    selected_pid_count = 0
    usable_qid_count = 0
    usable_pid_count = 0

    def _extend(items: Sequence[Dict[str, object]], item_type: str) -> None:
        nonlocal selected_qid_count, selected_pid_count, usable_qid_count, usable_pid_count
        selected = list(items[:top_k])
        if item_type == "qid":
            selected_qid_count += len(selected)
        else:
            selected_pid_count += len(selected)

        for item in selected:
            formatted = _format_retrieval_item(item, item_type)
            if formatted:
                if item_type == "qid":
                    qid_context_items.append(formatted)
                    usable_qid_count += 1
                else:
                    pid_context_items.append(formatted)
                    usable_pid_count += 1

    if rag_type in {"qid", "both"}:
        _extend(all_qid_items, "qid")
    if rag_type in {"pid", "both"}:
        _extend(all_pid_items, "pid")

    return qid_context_items, pid_context_items, {
        "raw_qid_retrieved_count": raw_qid_count,
        "raw_pid_retrieved_count": raw_pid_count,
        "raw_retrieved_count": raw_qid_count + raw_pid_count,
        "selected_qid_count": selected_qid_count,
        "selected_pid_count": selected_pid_count,
        "selected_count": selected_qid_count + selected_pid_count,
        "usable_qid_context_count": usable_qid_count,
        "usable_pid_context_count": usable_pid_count,
        "usable_context_count": usable_qid_count + usable_pid_count,
    }


def _build_bucket_messages(
    bucket_frame: pd.DataFrame,
    retrieval_payload: Dict[str, Dict[str, List[Dict[str, object]]]],
    prompt_name: str,
    rag_type: str,
    top_k: int,
) -> Tuple[List[List[Dict[str, str]]], Dict[str, float], List[Dict[str, object]]]:
    prompt_template = _load_prompt_template(prompt_name)
    messages: List[List[Dict[str, str]]] = []
    retrieved_pairs: List[Dict[str, object]] = []

    total_pairs = int(len(bucket_frame))
    pairs_with_any_retrieval = 0
    pairs_with_usable_context = 0
    total_raw_retrieved = 0
    total_raw_qid_retrieved = 0
    total_raw_pid_retrieved = 0
    total_selected_items = 0
    total_selected_qid_items = 0
    total_selected_pid_items = 0
    total_usable_context_items = 0
    total_usable_qid_context_items = 0
    total_usable_pid_context_items = 0
    pairs_with_qid_retrieval = 0
    pairs_with_pid_retrieval = 0

    for row in bucket_frame.itertuples():
        pair_id = getattr(row, "pair_id")
        qid_context_items, pid_context_items, counts = _select_context_items(retrieval_payload, pair_id, rag_type, top_k)

        total_raw_retrieved += counts["raw_retrieved_count"]
        total_raw_qid_retrieved += counts["raw_qid_retrieved_count"]
        total_raw_pid_retrieved += counts["raw_pid_retrieved_count"]
        total_selected_items += counts["selected_count"]
        total_selected_qid_items += counts["selected_qid_count"]
        total_selected_pid_items += counts["selected_pid_count"]
        total_usable_context_items += counts["usable_context_count"]
        total_usable_qid_context_items += counts["usable_qid_context_count"]
        total_usable_pid_context_items += counts["usable_pid_context_count"]
        if counts["raw_retrieved_count"] > 0:
            pairs_with_any_retrieval += 1
        if counts["raw_qid_retrieved_count"] > 0:
            pairs_with_qid_retrieval += 1
        if counts["raw_pid_retrieved_count"] > 0:
            pairs_with_pid_retrieval += 1
        if counts["usable_context_count"] > 0:
            pairs_with_usable_context += 1

        context_sections: List[str] = []
        if qid_context_items:
            context_sections.append("Entities: " + "; ".join(qid_context_items))
        if pid_context_items:
            context_sections.append("Properties: " + "; ".join(pid_context_items))
        context_text = " | ".join(context_sections) if context_sections else "No available relevant knowledge, please make the decision on your own."
        retrieved_pairs.append(
            {
                "pair_id": pair_id,
                "ltable_id": int(row.ltable_id),
                "rtable_id": int(row.rtable_id),
                "raw_qid_retrieved_count": counts["raw_qid_retrieved_count"],
                "raw_pid_retrieved_count": counts["raw_pid_retrieved_count"],
                "raw_retrieved_count": counts["raw_retrieved_count"],
                "selected_qid_count": counts["selected_qid_count"],
                "selected_pid_count": counts["selected_pid_count"],
                "selected_count": counts["selected_count"],
                "usable_qid_context_count": counts["usable_qid_context_count"],
                "usable_pid_context_count": counts["usable_pid_context_count"],
                "usable_context_count": counts["usable_context_count"],
                "qid_context_items": qid_context_items,
                "pid_context_items": pid_context_items,
                "context_text": context_text,
            }
        )
        user_content = prompt_template.format(row.entityA, row.entityB, context_text)
        messages.append([{"role": "user", "content": user_content}])

    retrieval_stats = {
        "pair_count": total_pairs,
        "pairs_with_any_retrieval": pairs_with_any_retrieval,
        "pairs_with_qid_retrieval": pairs_with_qid_retrieval,
        "pairs_with_pid_retrieval": pairs_with_pid_retrieval,
        "pairs_with_usable_context": pairs_with_usable_context,
        "total_raw_retrieved": total_raw_retrieved,
        "total_raw_qid_retrieved": total_raw_qid_retrieved,
        "total_raw_pid_retrieved": total_raw_pid_retrieved,
        "total_selected_items": total_selected_items,
        "total_selected_qid_items": total_selected_qid_items,
        "total_selected_pid_items": total_selected_pid_items,
        "total_usable_context_items": total_usable_context_items,
        "total_usable_qid_context_items": total_usable_qid_context_items,
        "total_usable_pid_context_items": total_usable_pid_context_items,
        "avg_raw_retrieved": float(total_raw_retrieved / total_pairs) if total_pairs else 0.0,
        "avg_raw_qid_retrieved": float(total_raw_qid_retrieved / total_pairs) if total_pairs else 0.0,
        "avg_raw_pid_retrieved": float(total_raw_pid_retrieved / total_pairs) if total_pairs else 0.0,
        "avg_selected_items": float(total_selected_items / total_pairs) if total_pairs else 0.0,
        "avg_selected_qid_items": float(total_selected_qid_items / total_pairs) if total_pairs else 0.0,
        "avg_selected_pid_items": float(total_selected_pid_items / total_pairs) if total_pairs else 0.0,
        "avg_usable_context_items": float(total_usable_context_items / total_pairs) if total_pairs else 0.0,
        "avg_usable_qid_context_items": float(total_usable_qid_context_items / total_pairs) if total_pairs else 0.0,
        "avg_usable_pid_context_items": float(total_usable_pid_context_items / total_pairs) if total_pairs else 0.0,
    }
    return messages, retrieval_stats, retrieved_pairs


def _evaluate_predictions(frame: pd.DataFrame, predictions: List[str]) -> Dict[str, float]:
    shared_len = min(len(frame), len(predictions))
    frame = frame.iloc[:shared_len].copy()
    predictions = predictions[:shared_len]

    predicted_labels = np.array([_parse_binary_decision(prediction) for prediction in predictions], dtype=int)
    gold_labels = frame["label"].astype(int).to_numpy()

    return {
        "count": int(shared_len),
        "accuracy": float(accuracy_score(gold_labels, predicted_labels)),
        "precision": float(precision_score(gold_labels, predicted_labels, zero_division=0)),
        "recall": float(recall_score(gold_labels, predicted_labels, zero_division=0)),
        "f1": float(f1_score(gold_labels, predicted_labels, zero_division=0)),
        "positive_rate": float(gold_labels.mean()) if shared_len else 0.0,
    }


def _evaluate_overall(bucket_frames: Dict[str, pd.DataFrame], bucket_results: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
    combined_frames = [bucket_frames[bucket_name] for bucket_name in BUCKET_NAMES]
    combined_predictions: List[str] = []
    for bucket_name in BUCKET_NAMES:
        combined_predictions.extend(bucket_results[bucket_name])

    overall = _evaluate_predictions(pd.concat(combined_frames, ignore_index=True), combined_predictions)
    per_bucket = {
        bucket_name: _evaluate_predictions(bucket_frames[bucket_name], bucket_results[bucket_name])
        for bucket_name in BUCKET_NAMES
    }
    return {
        "overall": overall,
        "per_bucket": per_bucket,
    }


def _write_prompt_files(
    dataset_key: str,
    partition: str,
    prompt_name: str,
    messages_by_bucket: Dict[str, List[List[Dict[str, str]]]],
    output_dir: Path,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_paths: Dict[str, str] = {}
    for bucket_name, messages in messages_by_bucket.items():
        prompt_path = output_dir / f"{dataset_key}_{partition}_{bucket_name}_{prompt_name}_direct_rag_messages.json"
        with prompt_path.open("w", encoding="utf-8") as file_handle:
            json.dump(
                {
                    "metadata": {
                        "dataset_key": dataset_key,
                        "partition": partition,
                        "bucket": bucket_name,
                        "prompt_name": prompt_name,
                        "count": len(messages),
                    },
                    "messages": messages,
                },
                file_handle,
                ensure_ascii=False,
                indent=2,
            )
        prompt_paths[bucket_name] = str(prompt_path)
    return prompt_paths


def _ascii_safe_text(text: str) -> str:
    replacements = {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
    }
    normalized = text
    for src, dst in replacements.items():
        normalized = normalized.replace(src, dst)
    return normalized.encode("ascii", errors="ignore").decode("ascii")


def _sanitize_messages_ascii(messages: List[List[Dict[str, str]]]) -> List[List[Dict[str, str]]]:
    sanitized_messages: List[List[Dict[str, str]]] = []
    for message_group in messages:
        sanitized_group: List[Dict[str, str]] = []
        for message in message_group:
            sanitized_message = dict(message)
            content = str(sanitized_message.get("content", ""))
            sanitized_message["content"] = _ascii_safe_text(content)
            sanitized_group.append(sanitized_message)
        sanitized_messages.append(sanitized_group)
    return sanitized_messages


def _expected_prompt_paths(dataset_key: str, partition: str, prompt_name: str, prompt_output_dir: Path) -> Dict[str, Path]:
    return {
        bucket_name: prompt_output_dir / f"{dataset_key}_{partition}_{bucket_name}_{prompt_name}_direct_rag_messages.json"
        for bucket_name in BUCKET_NAMES
    }


def _load_existing_prompt_files(
    dataset_key: str,
    partition: str,
    prompt_name: str,
    prompt_output_dir: Path,
) -> Tuple[Dict[str, List[List[Dict[str, str]]]], Dict[str, str]]:
    expected_paths = _expected_prompt_paths(dataset_key, partition, prompt_name, prompt_output_dir)
    if not all(path.exists() for path in expected_paths.values()):
        return {}, {}

    messages_by_bucket: Dict[str, List[List[Dict[str, str]]]] = {}
    prompt_paths: Dict[str, str] = {}
    for bucket_name, prompt_path in expected_paths.items():
        with prompt_path.open("r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)

        messages = payload.get("messages", [])
        if not isinstance(messages, list):
            return {}, {}

        messages_by_bucket[bucket_name] = messages
        prompt_paths[bucket_name] = str(prompt_path)

    return messages_by_bucket, prompt_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Run direct per-query RAG on split bucket datasets.")
    parser.add_argument("--dataset-key", default="amgo", help="Dataset key to evaluate (default: amgo).")
    parser.add_argument("--partition", default="test", help="Dataset partition to use (default: test).")
    parser.add_argument(
        "--bucket-csv-dir",
        default="output/match_entity_records_split",
        help="Directory containing split bucket CSVs named {dataset}_{bucket}.csv.",
    )
    parser.add_argument(
        "--query-dir",
        default="",
        help="Directory containing pair_relevant_ids_{partition}.json. Defaults to data/query/{dataset_key}.",
    )
    parser.add_argument(
        "--rag-type",
        default="qid",
        choices=["qid", "pid", "both"],
        help="Type of retrieved context to use.",
    )
    parser.add_argument("--top-k", type=int, default=2, help="Number of retrieved context items to include per type.")
    parser.add_argument(
        "--prompt-name",
        default="enforced10rag",
        help="Prompt template used for direct RAG prompting.",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="Model used for generation.")
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature.")
    parser.add_argument("--chunk-size", type=int, default=128, help="Number of requests per OpenAI batch.")
    parser.add_argument("--max-processes", type=int, default=128, help="Maximum worker processes for OpenAI requests.")
    parser.add_argument(
        "--output-dir",
        default="output/split_direct_rag_results",
        help="Directory where bucket results and the summary report will be written.",
    )
    parser.add_argument(
        "--save-prompts",
        action="store_true",
        help="Write bucket prompt JSON files before generation.",
    )
    parser.add_argument(
        "--prompt-output-dir",
        default="output/split_direct_rag_prompts",
        help="Directory for saved bucket prompt JSON files.",
    )
    args = parser.parse_args()

    bucket_csv_dir = Path(args.bucket_csv_dir)
    query_dir = Path(args.query_dir) if args.query_dir else REPO_ROOT / "data" / "query" / args.dataset_key
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_output_dir = Path(args.prompt_output_dir)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    bucket_frames = _load_bucket_csvs(args.dataset_key, bucket_csv_dir)
    messages_by_bucket, prompt_paths = _load_existing_prompt_files(
        args.dataset_key,
        args.partition,
        args.prompt_name,
        prompt_output_dir,
    )

    used_prompt_cache = bool(messages_by_bucket)
    retrieval_stats_by_bucket: Dict[str, Dict[str, float]] = {}
    pair_retrieval_path_str = ""
    retrieved_results_output_path: Optional[Path] = None

    if used_prompt_cache:
        print("Found existing prompt files for all buckets. Skipping retrieval/enrichment and reusing prompts.")
        for bucket_name in BUCKET_NAMES:
            message_count = len(messages_by_bucket.get(bucket_name, []))
            retrieval_stats_by_bucket[bucket_name] = {
                "pair_count": int(len(bucket_frames[bucket_name])),
                "cached_prompt_count": int(message_count),
                "cached_prompts_used": 1,
            }
    else:
        pair_retrieval_path = _ensure_pair_retrieval(bucket_frames, args.partition, query_dir)
        pair_retrieval_path_str = str(pair_retrieval_path)
        retrieval_payload = _load_pair_retrieval(pair_retrieval_path)

        retrieved_pairs_by_bucket: Dict[str, List[Dict[str, object]]] = {}
        for bucket_name in BUCKET_NAMES:
            messages, retrieval_stats, retrieved_pairs = _build_bucket_messages(
                bucket_frames[bucket_name],
                retrieval_payload,
                args.prompt_name,
                args.rag_type,
                args.top_k,
            )
            messages_by_bucket[bucket_name] = messages
            retrieval_stats_by_bucket[bucket_name] = retrieval_stats
            retrieved_pairs_by_bucket[bucket_name] = retrieved_pairs

        retrieved_results_output_path = output_dir / (
            f"{args.dataset_key}_{args.partition}_{args.rag_type}_{args.top_k}_{args.model}_"
            f"direct_retrieval_results_{run_timestamp}.json"
        )
        with retrieved_results_output_path.open("w", encoding="utf-8") as file_handle:
            json.dump(
                {
                    "metadata": {
                        "dataset_key": args.dataset_key,
                        "partition": args.partition,
                        "rag_type": args.rag_type,
                        "top_k": args.top_k,
                        "query_type": "pair",
                        "query_dir": str(query_dir),
                        "pair_retrieval_path": pair_retrieval_path_str,
                    },
                    "buckets": {
                        bucket_name: {
                            "retrieval_stats": retrieval_stats_by_bucket[bucket_name],
                            "pairs": retrieved_pairs_by_bucket[bucket_name],
                        }
                        for bucket_name in BUCKET_NAMES
                    },
                },
                file_handle,
                ensure_ascii=False,
                indent=2,
            )

        if args.save_prompts:
            prompt_paths = _write_prompt_files(
                args.dataset_key,
                args.partition,
                args.prompt_name,
                messages_by_bucket,
                prompt_output_dir,
            )

    bucket_results: Dict[str, List[str]] = {}
    bucket_output_paths: Dict[str, str] = {}
    bucket_error_counts: Dict[str, int] = {}
    for bucket_name in BUCKET_NAMES:
        messages = _sanitize_messages_ascii(messages_by_bucket[bucket_name])
        print(f"Generating {len(messages)} direct-RAG responses for {bucket_name} using {args.model}...")
        responses = process_requests(
            model=args.model,
            messages_list=messages,
            temperature=args.temperature,
            chunk_size=args.chunk_size,
            max_processes=args.max_processes,
        )
        bucket_results[bucket_name] = responses
        error_count = sum(1 for response in responses if isinstance(response, str) and response.startswith("ERROR:"))
        bucket_error_counts[bucket_name] = int(error_count)
        if error_count:
            print(f"Warning: {error_count}/{len(responses)} responses failed in bucket '{bucket_name}'.")
        if responses and error_count == len(responses):
            raise RuntimeError(
                f"All responses failed in bucket '{bucket_name}'. "
                "Check OpenAI credentials/proxy settings and message encoding."
            )

        output_path = output_dir / (
            f"{args.dataset_key}_{args.partition}_{bucket_name}_{args.rag_type}_{args.top_k}_"
            f"{args.prompt_name}_{args.model}_direct_rag_{run_timestamp}.json"
        )
        with output_path.open("w", encoding="utf-8") as file_handle:
            json.dump(
                {
                    "metadata": {
                        "dataset_key": args.dataset_key,
                        "partition": args.partition,
                        "bucket": bucket_name,
                        "prompt_name": args.prompt_name,
                        "model": args.model,
                        "temperature": args.temperature,
                        "rag_type": args.rag_type,
                        "top_k": args.top_k,
                        "query_dir": str(query_dir),
                        "pair_retrieval_path": pair_retrieval_path_str,
                        "retrieval_stats": retrieval_stats_by_bucket[bucket_name],
                        "error_count": bucket_error_counts[bucket_name],
                        "used_prompt_cache": used_prompt_cache,
                    },
                    "results": responses,
                },
                file_handle,
                ensure_ascii=False,
                indent=2,
            )
        bucket_output_paths[bucket_name] = str(output_path)

    metrics_report = _evaluate_overall(bucket_frames, bucket_results)
    bucket_summary = {
        bucket_name: {
            "count": int(len(bucket_frames[bucket_name])),
            "positive_count": int(bucket_frames[bucket_name]["label"].sum()),
            "negative_count": int(len(bucket_frames[bucket_name]) - bucket_frames[bucket_name]["label"].sum()),
            "positive_rate": float(bucket_frames[bucket_name]["label"].mean()) if len(bucket_frames[bucket_name]) else 0.0,
        }
        for bucket_name in BUCKET_NAMES
    }

    report = {
        "metadata": {
            "dataset_key": args.dataset_key,
            "partition": args.partition,
            "model": args.model,
            "temperature": args.temperature,
            "rag_type": args.rag_type,
            "top_k": args.top_k,
            "prompt_name": args.prompt_name,
            "bucket_csv_dir": str(bucket_csv_dir),
            "query_dir": str(query_dir),
            "pair_retrieval_path": pair_retrieval_path_str,
            "used_prompt_cache": used_prompt_cache,
            "context_config": {
                "enabled": True,
                "query_type": "pair",
                "rag_type": args.rag_type,
                "num": args.top_k,
            },
        },
        "buckets": bucket_summary,
        "prompt_paths": prompt_paths,
        "bucket_output_paths": bucket_output_paths,
        "retrieval_stats": retrieval_stats_by_bucket,
        "bucket_error_counts": bucket_error_counts,
        "retrieved_results_path": str(retrieved_results_output_path) if retrieved_results_output_path else "",
        "metrics": metrics_report,
    }

    report_path = output_dir / (
        f"{args.dataset_key}_{args.partition}_{args.rag_type}_{args.top_k}_{args.model}_"
        f"direct_rag_report_{run_timestamp}.json"
    )
    with report_path.open("w", encoding="utf-8") as file_handle:
        json.dump(report, file_handle, ensure_ascii=False, indent=2)

    print(f"Saved direct-RAG bucket report to {report_path}")
    print("Retrieval stats:")
    for bucket_name in BUCKET_NAMES:
        stats = retrieval_stats_by_bucket[bucket_name]
        if "pairs_with_any_retrieval" in stats:
            print(
                f"- {bucket_name}: pairs={stats['pair_count']}, any_retrieval={stats['pairs_with_any_retrieval']}, "
                f"usable_context={stats['pairs_with_usable_context']}, avg_selected={stats['avg_selected_items']:.2f}"
            )
        else:
            print(
                f"- {bucket_name}: pairs={stats.get('pair_count', 0)}, "
                f"cached_prompt_count={stats.get('cached_prompt_count', 0)}, "
                f"cached_prompts_used={stats.get('cached_prompts_used', 0)}"
            )
    print("Per-bucket metrics:")
    for bucket_name in BUCKET_NAMES:
        metrics = metrics_report["per_bucket"][bucket_name]
        print(
            f"- {bucket_name}: accuracy={metrics['accuracy']:.4f}, precision={metrics['precision']:.4f}, "
            f"recall={metrics['recall']:.4f}, f1={metrics['f1']:.4f}"
        )


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""Run split-bucket RAG using blocking-based batch retrieval outputs.

This script mirrors the split bucket evaluation flow used by the direct RAG
script, but builds context from block-level retrieval artifacts produced by the
blocking retrieval pipeline.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


def _compute_binary_metrics(gold_labels: np.ndarray, predicted_labels: np.ndarray) -> Dict[str, float]:
    if len(gold_labels) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    true_positive = int(np.sum((gold_labels == 1) & (predicted_labels == 1)))
    true_negative = int(np.sum((gold_labels == 0) & (predicted_labels == 0)))
    false_positive = int(np.sum((gold_labels == 0) & (predicted_labels == 1)))
    false_negative = int(np.sum((gold_labels == 1) & (predicted_labels == 0)))

    total = len(gold_labels)
    accuracy = float((true_positive + true_negative) / total) if total else 0.0
    precision = float(true_positive / (true_positive + false_positive)) if (true_positive + false_positive) else 0.0
    recall = float(true_positive / (true_positive + false_negative)) if (true_positive + false_negative) else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


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
                lambda row: f"{int(row.ltable_id)}-{int(row.rtable_id)}",
                axis=1,
            )
        if "bucket" not in bucket_frame.columns:
            bucket_frame["bucket"] = bucket_name
        bucket_frames[bucket_name] = bucket_frame
    return bucket_frames


def _normalize_entity_id(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text == "":
        return ""
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def _normalize_pair_id(ltable_id: object, rtable_id: object) -> str:
    return f"{_normalize_entity_id(ltable_id)}-{_normalize_entity_id(rtable_id)}"


def _resolve_subblocks_path(dataset_key: str, partition: str, blocking_method: str, max_block_size: int, explicit_path: str) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Provided subblocks path not found: {path}")
        return path

    candidate = REPO_ROOT / "blocking_outputs" / dataset_key / f"{dataset_key}_{partition}_{blocking_method}_{max_block_size}_subblocks_with_pairs.json"
    if candidate.exists():
        return candidate

    raise FileNotFoundError(
        "Could not find subblocks_with_pairs file. "
        f"Expected at {candidate}. Run 2_block_retrieval.py first or pass --subblocks-path."
    )


def _resolve_group_retrieval_path(
    dataset_key: str,
    partition: str,
    blocking_method: str,
    max_block_size: int,
    kg_source: str,
    explicit_path: str,
) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Provided group retrieval path not found: {path}")
        return path

    candidates = [
        REPO_ROOT / "retrieval_outputs" / dataset_key / f"{dataset_key}_{partition}_{blocking_method}_{max_block_size}_{kg_source}_rgroup_retrieval_results.json",
        REPO_ROOT / "retrieval_outputs" / dataset_key / f"{dataset_key}_{partition}_{blocking_method}_{max_block_size}_{kg_source}_group_retrieval_results.json",
        REPO_ROOT / "retrieval_outputs" / dataset_key / f"{dataset_key}_{partition}_{blocking_method}_{max_block_size}_group_retrieval_results.json",
    ]
    for path in candidates:
        if path.exists():
            return path

    joined = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find blocking group retrieval file. "
        f"Checked: {joined}. Run 2_block_retrieval.py first or pass --group-retrieval-path."
    )


def _load_pair_to_block_map(subblocks_path: Path) -> Dict[str, str]:
    with subblocks_path.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    pair_to_block: Dict[str, str] = {}
    blocks = payload.get("blocks", {})
    for block_id, block_data in blocks.items():
        for pair in block_data.get("pairs", []):
            pair_id = _normalize_pair_id(pair.get("ltable_id"), pair.get("rtable_id"))
            if pair_id:
                pair_to_block[pair_id] = block_id
    return pair_to_block


def _load_group_retrieval(group_retrieval_path: Path) -> Dict[str, object]:
    with group_retrieval_path.open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def _extract_retrieval_ids(retrieval_blocks: Dict[str, Dict[str, object]]) -> Tuple[List[str], List[str]]:
    qids: set[str] = set()
    pids: set[str] = set()

    for block_data in retrieval_blocks.values():
        for item in block_data.get("relevant_qids", []) or []:
            if isinstance(item, dict) and item.get("QID"):
                qids.add(str(item["QID"]))
        for item in block_data.get("relevant_pids", []) or []:
            if isinstance(item, dict) and item.get("PID"):
                pids.add(str(item["PID"]))

    return sorted(qids), sorted(pids)


def _load_wikidata_info(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    if not isinstance(payload, dict):
        return {}
    return payload


def _enrich_retrieval_blocks_with_wikidata(
    retrieval_blocks: Dict[str, Dict[str, object]],
    cache_dir: Path,
    cache_key: str,
) -> Dict[str, Dict[str, object]]:
    from data_utils.wiki_query import fetch_and_save_wikidata

    cache_dir.mkdir(parents=True, exist_ok=True)
    qids, pids = _extract_retrieval_ids(retrieval_blocks)

    qids_path = cache_dir / f"qids_{cache_key}.json"
    pids_path = cache_dir / f"pids_{cache_key}.json"

    if qids and not qids_path.exists():
        fetch_and_save_wikidata(qids, str(qids_path))
    elif not qids and not qids_path.exists():
        qids_path.write_text("{}", encoding="utf-8")

    if pids and not pids_path.exists():
        fetch_and_save_wikidata(pids, str(pids_path))
    elif not pids and not pids_path.exists():
        pids_path.write_text("{}", encoding="utf-8")

    qid_info = _load_wikidata_info(qids_path)
    pid_info = _load_wikidata_info(pids_path)

    enriched_blocks: Dict[str, Dict[str, object]] = {}
    for block_id, block_data in retrieval_blocks.items():
        enriched_block = dict(block_data)

        enriched_qids = []
        for item in block_data.get("relevant_qids", []) or []:
            if not isinstance(item, dict):
                continue
            enriched_item = dict(item)
            qid = str(item.get("QID", ""))
            info = qid_info.get(qid, {})
            enriched_item["label"] = str(info.get("label", "") or "")
            enriched_item["description"] = str(info.get("description", "") or "")
            enriched_qids.append(enriched_item)

        enriched_pids = []
        for item in block_data.get("relevant_pids", []) or []:
            if not isinstance(item, dict):
                continue
            enriched_item = dict(item)
            pid = str(item.get("PID", ""))
            info = pid_info.get(pid, {})
            enriched_item["label"] = str(info.get("label", "") or "")
            enriched_item["description"] = str(info.get("description", "") or "")
            enriched_pids.append(enriched_item)

        enriched_block["relevant_qids"] = enriched_qids
        enriched_block["relevant_pids"] = enriched_pids
        enriched_blocks[block_id] = enriched_block

    return enriched_blocks


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
    block_retrieval: Dict[str, object],
    rag_type: str,
    top_k: int,
) -> Tuple[List[str], List[str], Dict[str, int]]:
    qid_context_items: List[str] = []
    pid_context_items: List[str] = []
    all_qid_items = block_retrieval.get("relevant_qids", []) or []
    all_pid_items = block_retrieval.get("relevant_pids", []) or []

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


def _build_bucket_messages_with_blocking_context(
    bucket_frame: pd.DataFrame,
    pair_to_block: Dict[str, str],
    retrieval_blocks: Dict[str, Dict[str, object]],
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
    pairs_missing_block_mapping = 0
    pairs_missing_retrieval_block = 0
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
        pair_id = str(getattr(row, "pair_id"))
        normalized_pair_id = _normalize_pair_id(row.ltable_id, row.rtable_id)

        block_id = pair_to_block.get(normalized_pair_id)
        block_retrieval = retrieval_blocks.get(block_id, {}) if block_id else {}

        if not block_id:
            pairs_missing_block_mapping += 1
        elif block_id not in retrieval_blocks:
            pairs_missing_retrieval_block += 1

        qid_context_items, pid_context_items, counts = _select_context_items(block_retrieval, rag_type, top_k)

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
                "normalized_pair_id": normalized_pair_id,
                "ltable_id": int(float(row.ltable_id)),
                "rtable_id": int(float(row.rtable_id)),
                "block_id": block_id if block_id else "",
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
        "pairs_missing_block_mapping": pairs_missing_block_mapping,
        "pairs_missing_retrieval_block": pairs_missing_retrieval_block,
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
    metrics = _compute_binary_metrics(gold_labels, predicted_labels)

    return {
        "count": int(shared_len),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
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
            sanitized_message["content"] = _ascii_safe_text(str(sanitized_message.get("content", "")))
            sanitized_group.append(sanitized_message)
        sanitized_messages.append(sanitized_group)
    return sanitized_messages


def _expected_prompt_paths(
    dataset_key: str,
    partition: str,
    prompt_name: str,
    blocking_method: str,
    max_block_size: int,
    prompt_output_dir: Path,
) -> Dict[str, Path]:
    return {
        bucket_name: (
            prompt_output_dir /
            f"{dataset_key}_{partition}_{bucket_name}_{prompt_name}_{blocking_method}_{max_block_size}_blocking_rag_messages.json"
        )
        for bucket_name in BUCKET_NAMES
    }


def _load_existing_prompt_files(
    dataset_key: str,
    partition: str,
    prompt_name: str,
    blocking_method: str,
    max_block_size: int,
    prompt_output_dir: Path,
) -> Tuple[Dict[str, List[List[Dict[str, str]]]], Dict[str, str]]:
    expected_paths = _expected_prompt_paths(
        dataset_key,
        partition,
        prompt_name,
        blocking_method,
        max_block_size,
        prompt_output_dir,
    )
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


def _write_prompt_files(
    dataset_key: str,
    partition: str,
    prompt_name: str,
    blocking_method: str,
    max_block_size: int,
    messages_by_bucket: Dict[str, List[List[Dict[str, str]]]],
    output_dir: Path,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_paths: Dict[str, str] = {}
    for bucket_name, messages in messages_by_bucket.items():
        prompt_path = output_dir / (
            f"{dataset_key}_{partition}_{bucket_name}_{prompt_name}_{blocking_method}_{max_block_size}_blocking_rag_messages.json"
        )
        with prompt_path.open("w", encoding="utf-8") as file_handle:
            json.dump(
                {
                    "metadata": {
                        "dataset_key": dataset_key,
                        "partition": partition,
                        "bucket": bucket_name,
                        "prompt_name": prompt_name,
                        "blocking_method": blocking_method,
                        "max_block_size": max_block_size,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run split bucket RAG with blocking-based batch retrieval context.")
    parser.add_argument("--dataset-key", default="amgo", help="Dataset key to evaluate (default: amgo).")
    parser.add_argument("--partition", default="test", help="Dataset partition to use (default: test).")
    parser.add_argument(
        "--bucket-csv-dir",
        default="output/match_entity_records_split",
        help="Directory containing split bucket CSVs named {dataset}_{bucket}.csv.",
    )
    parser.add_argument("--blocking-method", default="QG", help="Blocking method used to generate subblocks/retrieval.")
    parser.add_argument("--max-block-size", type=int, default=6, help="Max subblock size used in blocking retrieval outputs.")
    parser.add_argument("--kg-source", default="wikidata", help="Knowledge source tag used in retrieval output names.")
    parser.add_argument(
        "--subblocks-path",
        default="",
        help="Optional explicit path to {dataset}_{partition}_{blocking}_{size}_subblocks_with_pairs.json.",
    )
    parser.add_argument(
        "--group-retrieval-path",
        default="",
        help="Optional explicit path to blocking group retrieval results JSON.",
    )
    parser.add_argument(
        "--rag-type",
        default="pid",
        choices=["qid", "pid", "both"],
        help="Type of retrieved context to use.",
    )
    parser.add_argument("--top-k", type=int, default=2, help="Number of retrieved context items to include per type.")
    parser.add_argument(
        "--prompt-name",
        default="enforced10rag_no_reranking",
        help="Prompt template used for RAG prompting (default: enforced10rag_no_reranking).",
    )
    parser.add_argument("--model", default="gpt-4o-mini", help="Model used for generation.")
    parser.add_argument("--temperature", type=float, default=0.5, help="Sampling temperature.")
    parser.add_argument("--chunk-size", type=int, default=128, help="Number of requests per OpenAI batch.")
    parser.add_argument("--max-processes", type=int, default=128, help="Maximum worker processes for OpenAI requests.")
    parser.add_argument(
        "--output-dir",
        default="output/split_blocking_rag_results",
        help="Directory where bucket results and summary report will be written.",
    )
    parser.add_argument(
        "--no-save-prompts",
        action="store_true",
        help="Disable writing bucket prompt JSON files before generation.",
    )
    parser.add_argument(
        "--prompt-output-dir",
        default="output/split_blocking_rag_prompts",
        help="Directory for saved bucket prompt JSON files.",
    )
    parser.add_argument(
        "--ignore-prompt-cache",
        action="store_true",
        help="Regenerate prompts even if cached prompt files already exist.",
    )
    args = parser.parse_args()
    save_prompts = not args.no_save_prompts

    bucket_csv_dir = Path(args.bucket_csv_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_output_dir = Path(args.prompt_output_dir)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    subblocks_path = _resolve_subblocks_path(
        args.dataset_key,
        args.partition,
        args.blocking_method,
        args.max_block_size,
        args.subblocks_path,
    )
    group_retrieval_path = _resolve_group_retrieval_path(
        args.dataset_key,
        args.partition,
        args.blocking_method,
        args.max_block_size,
        args.kg_source,
        args.group_retrieval_path,
    )

    bucket_frames = _load_bucket_csvs(args.dataset_key, bucket_csv_dir)
    messages_by_bucket, prompt_paths = _load_existing_prompt_files(
        args.dataset_key,
        args.partition,
        args.prompt_name,
        args.blocking_method,
        args.max_block_size,
        prompt_output_dir,
    )

    if args.ignore_prompt_cache:
        messages_by_bucket = {}
        prompt_paths = {}

    used_prompt_cache = bool(messages_by_bucket)
    retrieval_stats_by_bucket: Dict[str, Dict[str, float]] = {}
    retrieved_results_output_path: Optional[Path] = None

    if used_prompt_cache:
        print("Found existing blocking prompt files for all buckets. Skipping retrieval mapping and reusing prompts.")
        for bucket_name in BUCKET_NAMES:
            message_count = len(messages_by_bucket.get(bucket_name, []))
            retrieval_stats_by_bucket[bucket_name] = {
                "pair_count": int(len(bucket_frames[bucket_name])),
                "cached_prompt_count": int(message_count),
                "cached_prompts_used": 1,
            }
    else:
        pair_to_block = _load_pair_to_block_map(subblocks_path)
        group_retrieval_payload = _load_group_retrieval(group_retrieval_path)
        retrieval_blocks = group_retrieval_payload.get("blocks", {})
        retrieval_blocks = _enrich_retrieval_blocks_with_wikidata(
            retrieval_blocks,
            output_dir / "wikidata_cache",
            f"{args.dataset_key}_{args.partition}_{args.blocking_method}_{args.max_block_size}",
        )

        retrieved_pairs_by_bucket: Dict[str, List[Dict[str, object]]] = {}
        for bucket_name in BUCKET_NAMES:
            messages, retrieval_stats, retrieved_pairs = _build_bucket_messages_with_blocking_context(
                bucket_frames[bucket_name],
                pair_to_block,
                retrieval_blocks,
                args.prompt_name,
                args.rag_type,
                args.top_k,
            )
            messages_by_bucket[bucket_name] = messages
            retrieval_stats_by_bucket[bucket_name] = retrieval_stats
            retrieved_pairs_by_bucket[bucket_name] = retrieved_pairs

        retrieved_results_output_path = output_dir / (
            f"{args.dataset_key}_{args.partition}_{args.blocking_method}_{args.max_block_size}_{args.rag_type}_{args.top_k}_"
            f"{args.model}_blocking_retrieval_results_{run_timestamp}.json"
        )
        with retrieved_results_output_path.open("w", encoding="utf-8") as file_handle:
            json.dump(
                {
                    "metadata": {
                        "dataset_key": args.dataset_key,
                        "partition": args.partition,
                        "blocking_method": args.blocking_method,
                        "max_block_size": args.max_block_size,
                        "kg_source": args.kg_source,
                        "rag_type": args.rag_type,
                        "top_k": args.top_k,
                        "subblocks_path": str(subblocks_path),
                        "group_retrieval_path": str(group_retrieval_path),
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

        if save_prompts:
            prompt_paths = _write_prompt_files(
                args.dataset_key,
                args.partition,
                args.prompt_name,
                args.blocking_method,
                args.max_block_size,
                messages_by_bucket,
                prompt_output_dir,
            )

    bucket_results: Dict[str, List[str]] = {}
    bucket_output_paths: Dict[str, str] = {}
    bucket_error_counts: Dict[str, int] = {}

    from model_utils import process_requests

    for bucket_name in BUCKET_NAMES:
        messages = _sanitize_messages_ascii(messages_by_bucket[bucket_name])
        print(f"Generating {len(messages)} blocking-RAG responses for {bucket_name} using {args.model}...")
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
            f"{args.dataset_key}_{args.partition}_{bucket_name}_{args.blocking_method}_{args.max_block_size}_"
            f"{args.rag_type}_{args.top_k}_{args.prompt_name}_{args.model}_blocking_rag_{run_timestamp}.json"
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
                        "blocking_method": args.blocking_method,
                        "max_block_size": args.max_block_size,
                        "kg_source": args.kg_source,
                        "rag_type": args.rag_type,
                        "top_k": args.top_k,
                        "subblocks_path": str(subblocks_path),
                        "group_retrieval_path": str(group_retrieval_path),
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
            "blocking_method": args.blocking_method,
            "max_block_size": args.max_block_size,
            "kg_source": args.kg_source,
            "rag_type": args.rag_type,
            "top_k": args.top_k,
            "prompt_name": args.prompt_name,
            "bucket_csv_dir": str(bucket_csv_dir),
            "subblocks_path": str(subblocks_path),
            "group_retrieval_path": str(group_retrieval_path),
            "used_prompt_cache": used_prompt_cache,
            "context_config": {
                "enabled": True,
                "query_type": "blocking_group",
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
        f"{args.dataset_key}_{args.partition}_{args.blocking_method}_{args.max_block_size}_{args.rag_type}_{args.top_k}_"
        f"{args.model}_blocking_rag_report_{run_timestamp}.json"
    )
    with report_path.open("w", encoding="utf-8") as file_handle:
        json.dump(report, file_handle, ensure_ascii=False, indent=2)

    print(f"Saved blocking-RAG bucket report to {report_path}")
    print("Retrieval stats:")
    for bucket_name in BUCKET_NAMES:
        stats = retrieval_stats_by_bucket[bucket_name]
        if "pairs_with_any_retrieval" in stats:
            print(
                f"- {bucket_name}: pairs={stats['pair_count']}, any_retrieval={stats['pairs_with_any_retrieval']}, "
                f"usable_context={stats['pairs_with_usable_context']}, missing_pair_map={stats['pairs_missing_block_mapping']}, "
                f"missing_retrieval_block={stats['pairs_missing_retrieval_block']}, avg_selected={stats['avg_selected_items']:.2f}"
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

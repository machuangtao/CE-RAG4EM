#!/usr/bin/env python3
"""Generate LLM-only prompts and metrics for similarity buckets.

This script rebuilds the selected dataset test split, assigns the same similarity buckets
used by the split report, and evaluates stored LLM-only entity-matching outputs
with the non-RAG `enforced10` prompt path.

It is intended to mirror the LLM-only flow used when `context_config` is
disabled in the existing entity matching pipeline.
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


def _prepare_text_data(dataset_key: str, partition: str, with_semantic: bool = True) -> pd.DataFrame:
    paths = CONSTANTS.DATASET_PATHS[dataset_key]

    table_a = pd.read_csv(paths["tableA"])
    table_b = pd.read_csv(paths["tableB"])
    partition_path = paths.get(partition)
    if partition_path is None:
        raise ValueError(f"No path found for partition '{partition}' in dataset '{dataset_key}'.")

    pairs = pd.read_csv(partition_path)

    rename_map = CONSTANTS.COLUMN_RENAMES.get(dataset_key, {})
    if rename_map:
        table_a = table_a.rename(columns=rename_map)
        table_b = table_b.rename(columns=rename_map)

    drop_cols = CONSTANTS.COLUMN_DROP.get(dataset_key, [])
    if drop_cols:
        table_a = table_a.drop(columns=[col for col in drop_cols if col in table_a.columns])
        table_b = table_b.drop(columns=[col for col in drop_cols if col in table_b.columns])

    table_a = table_a.add_suffix("_a")
    table_b = table_b.add_suffix("_b")

    table_a = table_a.rename(columns={"id_a": "ltable_id"})
    table_b = table_b.rename(columns={"id_b": "rtable_id"})

    merged = pairs.merge(table_a, on="ltable_id", how="left")
    merged = merged.merge(table_b, on="rtable_id", how="left")

    def _serialize_entity(row: pd.Series) -> str:
        parts = []
        for col_name, value in row.items():
            text_value = str(value) if pd.notnull(value) else "nan"
            parts.append(f"{col_name[:-2]}: {text_value}" if with_semantic else text_value)
        return "; ".join(parts)

    a_cols = [col for col in merged.columns if col.endswith("_a")]
    b_cols = [col for col in merged.columns if col.endswith("_b")]
    merged["entityA"] = merged[a_cols].apply(lambda row: _serialize_entity(row), axis=1)
    merged["entityB"] = merged[b_cols].apply(lambda row: _serialize_entity(row), axis=1)

    return merged[["ltable_id", "rtable_id", "entityA", "entityB", "label"]]


def _load_split_report_thresholds(split_report_path: Path) -> Dict[str, float]:
    with split_report_path.open("r", encoding="utf-8") as file_handle:
        split_report = json.load(file_handle)
    thresholds = split_report.get("metadata", {}).get("thresholds", {})
    if not thresholds:
        raise ValueError(f"Missing thresholds in split report: {split_report_path}")
    return {
        "low_threshold": float(thresholds["low_threshold"]),
        "high_threshold": float(thresholds["high_threshold"]),
    }


def _build_similarity_frame(
    dataset_key: str,
    partition: str,
    *,
    model_name: str,
    batch_size: int,
    split_report_path: Path,
) -> pd.DataFrame:
    text_df = _prepare_text_data(dataset_key, partition, with_semantic=True).copy()
    thresholds = _load_split_report_thresholds(split_report_path)

    text_df["pair_id"] = text_df.apply(lambda row: f"{row.ltable_id}-{row.rtable_id}", axis=1)

    # Similarity is only used for bucket assignment. If bucket CSVs already exist,
    # they will be used directly and this score is not needed downstream.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    entity_a_embeddings = model.encode(
        text_df["entityA"].tolist(),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    entity_b_embeddings = model.encode(
        text_df["entityB"].tolist(),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    text_df["similarity_score"] = np.sum(entity_a_embeddings * entity_b_embeddings, axis=1).astype(float)

    low_threshold = float(thresholds["low_threshold"])
    high_threshold = float(thresholds["high_threshold"])

    def bucket_for_score(score: float) -> str:
        if score <= low_threshold:
            return "easy_non_match"
        if score >= high_threshold:
            return "easy_match"
        return "ambiguous"

    text_df["bucket"] = text_df["similarity_score"].map(bucket_for_score)
    return text_df


def _bucket_summary(frame: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for bucket_name in BUCKET_NAMES:
        subset = frame[frame["bucket"] == bucket_name]
        if subset.empty:
            summary[bucket_name] = {
                "count": 0,
                "positive_count": 0,
                "negative_count": 0,
                "positive_rate": 0.0,
                "mean_similarity": 0.0,
                "min_similarity": 0.0,
                "max_similarity": 0.0,
            }
            continue

        positive_count = int(subset["label"].sum())
        count = int(len(subset))
        summary[bucket_name] = {
            "count": count,
            "positive_count": positive_count,
            "negative_count": int(count - positive_count),
            "positive_rate": float(positive_count / count),
            "mean_similarity": float(subset["similarity_score"].mean()),
            "min_similarity": float(subset["similarity_score"].min()),
            "max_similarity": float(subset["similarity_score"].max()),
        }
    return summary


def _load_bucket_csvs(dataset_key: str, bucket_csv_dir: Path) -> Dict[str, pd.DataFrame]:
    bucket_frames: Dict[str, pd.DataFrame] = {}
    for bucket_name in BUCKET_NAMES:
        csv_path = bucket_csv_dir / f"{dataset_key}_{bucket_name}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing bucket CSV: {csv_path}")
        bucket_frames[bucket_name] = pd.read_csv(csv_path)
    return bucket_frames


def _load_prompt_template(prompt_name: str) -> str:
    prompt_templates = CONSTANTS.PROMPT_TEMPLATES
    if prompt_name not in prompt_templates:
        raise ValueError(f"Prompt '{prompt_name}' not found. Available prompts: {list(prompt_templates.keys())}")
    return prompt_templates[prompt_name]["user"]


def _build_messages(frame: pd.DataFrame, prompt_name: str) -> List[List[Dict[str, str]]]:
    prompt_template = _load_prompt_template(prompt_name)
    messages: List[List[Dict[str, str]]] = []

    for row in frame.itertuples():
        user_content = prompt_template.format(row.entityA, row.entityB)
        messages.append([{"role": "user", "content": user_content}])

    return messages


def _write_bucket_prompts(
    frame: pd.DataFrame,
    output_dir: Path,
    dataset_key: str,
    prompt_name: str,
    run_timestamp: str,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_paths: Dict[str, str] = {}

    for bucket_name in BUCKET_NAMES:
        bucket_frame = frame[frame["bucket"] == bucket_name].copy()
        messages = _build_messages(bucket_frame, prompt_name)
        prompt_path = output_dir / f"{dataset_key}_{bucket_name}_llm_messages.json"
        with prompt_path.open("w", encoding="utf-8") as file_handle:
            json.dump(
                {
                    "metadata": {
                        "dataset_key": dataset_key,
                        "prompt_name": prompt_name,
                        "bucket": bucket_name,
                        "run_timestamp": run_timestamp,
                        "count": int(len(bucket_frame)),
                    },
                    "messages": messages,
                },
                file_handle,
                ensure_ascii=False,
                indent=2,
            )
        prompt_paths[bucket_name] = str(prompt_path)

    return prompt_paths


def _evaluate_predictions(frame: pd.DataFrame, predictions: List[str], results_path: Path) -> Dict[str, object]:
    if len(predictions) != len(frame):
        shared_len = min(len(predictions), len(frame))
        frame = frame.iloc[:shared_len].copy()
        predictions = predictions[:shared_len]

    predicted_labels = np.array([_parse_binary_decision(prediction) for prediction in predictions], dtype=int)
    gold_labels = frame["label"].astype(int).to_numpy()

    overall = {
        "accuracy": float(accuracy_score(gold_labels, predicted_labels)),
        "precision": float(precision_score(gold_labels, predicted_labels, zero_division=0)),
        "recall": float(recall_score(gold_labels, predicted_labels, zero_division=0)),
        "f1": float(f1_score(gold_labels, predicted_labels, zero_division=0)),
    }

    per_bucket: Dict[str, Dict[str, float]] = {}
    for bucket_name in BUCKET_NAMES:
        bucket_mask = frame["bucket"] == bucket_name
        bucket_frame = frame.loc[bucket_mask]
        if bucket_frame.empty:
            per_bucket[bucket_name] = {
                "count": 0,
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "positive_rate": 0.0,
            }
            continue

        bucket_predictions = predicted_labels[bucket_mask.to_numpy()]
        bucket_gold = gold_labels[bucket_mask.to_numpy()]
        per_bucket[bucket_name] = {
            "count": int(len(bucket_frame)),
            "accuracy": float(accuracy_score(bucket_gold, bucket_predictions)),
            "precision": float(precision_score(bucket_gold, bucket_predictions, zero_division=0)),
            "recall": float(recall_score(bucket_gold, bucket_predictions, zero_division=0)),
            "f1": float(f1_score(bucket_gold, bucket_predictions, zero_division=0)),
            "positive_rate": float(bucket_gold.mean()),
        }

    return {
        "results_path": str(results_path),
        "overall": overall,
        "per_bucket": per_bucket,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GPT-4o-mini LLM-only split bucket results with the enforced10 prompt.")
    parser.add_argument(
        "--dataset-key",
        default="amgo",
        choices=sorted(CONSTANTS.DATASET_PATHS.keys()),
        help="Dataset key to evaluate (default: amgo). Supported keys include waam.",
    )
    parser.add_argument("--partition", default="test", help="Dataset partition to use (default: test).")
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model used to rebuild the similarity buckets.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    parser.add_argument(
        "--prompt-name",
        default="enforced10",
        help="Prompt template used for the LLM-only messages (default: enforced10).",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model used to generate the LLM-only results.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5,
        help="OpenAI sampling temperature.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help="Number of requests to send per batch.",
    )
    parser.add_argument(
        "--max-processes",
        type=int,
        default=128,
        help="Maximum worker processes for OpenAI requests.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory where the evaluation report and generated results are written.",
    )
    parser.add_argument(
        "--split-report-path",
        default="",
        help="Path to the saved split report containing the similarity thresholds.",
    )
    parser.add_argument(
        "--bucket-csv-dir",
        default="",
        help="Directory containing {dataset}_easy_non_match.csv, {dataset}_ambiguous.csv, and {dataset}_easy_match.csv.",
    )
    parser.add_argument(
        "--save-prompts",
        action="store_true",
        help="Write prompt JSON files for each bucket using the LLM-only template.",
    )
    parser.add_argument(
        "--prompt-output-dir",
        default="output",
        help="Base directory for bucket prompt JSON files when --save-prompts is set.",
    )
    args = parser.parse_args()

    if args.dataset_key not in CONSTANTS.DATASET_PATHS:
        raise ValueError(
            f"Unsupported dataset key '{args.dataset_key}'. Available keys: {sorted(CONSTANTS.DATASET_PATHS.keys())}"
        )

    output_dir = Path(args.output_dir) / f"{args.dataset_key}_split_llm_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    split_report_path = Path(args.split_report_path) if args.split_report_path else (
        output_dir.parent / f"{args.dataset_key}_match_entity_records_split" / f"{args.dataset_key}_{args.partition}_match_entity_records_split.json"
    )
    bucket_csv_dir = Path(args.bucket_csv_dir) if args.bucket_csv_dir else output_dir.parent / f"{args.dataset_key}_match_entity_records_split"

    if bucket_csv_dir.exists():
        bucket_frames = _load_bucket_csvs(args.dataset_key, bucket_csv_dir)
        frame = pd.concat([bucket_frames[bucket_name] for bucket_name in BUCKET_NAMES], ignore_index=True)
    else:
        frame = _build_similarity_frame(
            args.dataset_key,
            args.partition,
            model_name=args.model_name,
            batch_size=args.batch_size,
            split_report_path=split_report_path,
        )

    bucket_summary = _bucket_summary(frame)
    bucket_frames = {bucket_name: frame[frame["bucket"] == bucket_name].copy() for bucket_name in BUCKET_NAMES}
    prompt_paths: Dict[str, str] = {}
    prompt_output_dir = Path(args.prompt_output_dir) / f"split_{args.dataset_key}_llm_prompts"
    if args.save_prompts:
        prompt_paths = _write_bucket_prompts(frame, prompt_output_dir, args.dataset_key, args.prompt_name, run_timestamp)

    bucket_messages: Dict[str, List[List[Dict[str, str]]]] = {
        bucket_name: _build_messages(bucket_frame, args.prompt_name)
        for bucket_name, bucket_frame in bucket_frames.items()
    }

    bucket_results: Dict[str, List[str]] = {}
    bucket_output_paths: Dict[str, str] = {}
    bucket_comparison: Dict[str, Dict[str, object]] = {}

    for bucket_name in BUCKET_NAMES:
        messages = bucket_messages[bucket_name]
        print(f"Generating {len(messages)} responses for {bucket_name} using {args.model}...")
        responses = process_requests(
            model=args.model,
            messages_list=messages,
            temperature=args.temperature,
            chunk_size=args.chunk_size,
            max_processes=args.max_processes,
        )
        bucket_results[bucket_name] = responses

        bucket_output_path = output_dir / (
            f"{args.dataset_key}_{args.partition}_{bucket_name}_{args.model}_llm_{run_timestamp}.json"
        )
        with bucket_output_path.open("w", encoding="utf-8") as file_handle:
            json.dump(
                {
                    "metadata": {
                        "dataset_key": args.dataset_key,
                        "partition": args.partition,
                        "bucket": bucket_name,
                        "prompt_name": args.prompt_name,
                        "model": args.model,
                        "temperature": args.temperature,
                        "chunk_size": args.chunk_size,
                        "max_processes": args.max_processes,
                        "run_timestamp": run_timestamp,
                        "count": int(len(messages)),
                    },
                    "results": responses,
                },
                file_handle,
                ensure_ascii=False,
                indent=2,
            )
        bucket_output_paths[bucket_name] = str(bucket_output_path)

        bucket_comparison[bucket_name] = _evaluate_predictions(bucket_frames[bucket_name], responses, bucket_output_path)

    report = {
        "metadata": {
            "dataset_key": args.dataset_key,
            "partition": args.partition,
            "prompt_name": args.prompt_name,
            "model": args.model,
            "temperature": args.temperature,
            "chunk_size": args.chunk_size,
            "max_processes": args.max_processes,
            "model_name": args.model_name,
            "batch_size": args.batch_size,
            "run_timestamp": run_timestamp,
            "total_pairs": int(len(frame)),
            "split_report_path": str(split_report_path),
            "bucket_csv_dir": str(bucket_csv_dir),
            "context_config": {"enabled": False},
        },
        "buckets": bucket_summary,
        "prompt_paths": prompt_paths,
        "bucket_output_paths": bucket_output_paths,
        "comparison": bucket_comparison,
    }

    report_path = output_dir / f"{args.dataset_key}_{args.partition}_split_llm_results_{run_timestamp}.json"
    with report_path.open("w", encoding="utf-8") as file_handle:
        json.dump(report, file_handle, ensure_ascii=False, indent=2)

    print(f"Saved evaluation report to {report_path}")
    print("Bucket summary:")
    for bucket_name in BUCKET_NAMES:
        stats = bucket_summary[bucket_name]
        print(
            f"- {bucket_name}: count={stats['count']}, pos_rate={stats['positive_rate']:.3f}, "
            f"mean_sim={stats['mean_similarity']:.3f}"
        )

    print("Per-bucket metrics:")
    for bucket_name in BUCKET_NAMES:
        metrics = bucket_comparison[bucket_name]
        print(
            f"- {bucket_name}: accuracy={metrics['overall']['accuracy']:.4f}, "
            f"precision={metrics['overall']['precision']:.4f}, "
            f"recall={metrics['overall']['recall']:.4f}, f1={metrics['overall']['f1']:.4f}"
        )


if __name__ == "__main__":
    main()
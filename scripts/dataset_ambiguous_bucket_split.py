#!/usr/bin/env python3
"""Split entity matching pairs into similarity buckets.

This script follows the Match Entity Records idea of using record embeddings to
measure pair similarity, then buckets record pairs into:

- easy_non_match: low similarity
- ambiguous: mid similarity
- easy_match: high similarity
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

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


def _encode_texts(
    model: SentenceTransformer,
    texts: Sequence[str],
    *,
    batch_size: int,
) -> np.ndarray:
    return model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )


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


def _build_similarity_frame(
    dataset_key: str,
    partition: str,
    *,
    model_name: str,
    batch_size: int,
) -> pd.DataFrame:
    text_df = _prepare_text_data(dataset_key, partition, with_semantic=True).copy()
    text_df["pair_id"] = text_df.apply(
        lambda row: f"{row.ltable_id}-{row.rtable_id}",
        axis=1,
    )

    model = SentenceTransformer(model_name)
    entity_a_embeddings = _encode_texts(model, text_df["entityA"].tolist(), batch_size=batch_size)
    entity_b_embeddings = _encode_texts(model, text_df["entityB"].tolist(), batch_size=batch_size)

    similarity_scores = np.sum(entity_a_embeddings * entity_b_embeddings, axis=1)
    text_df["similarity_score"] = similarity_scores.astype(float)
    return text_df


def _assign_buckets(
    frame: pd.DataFrame,
    *,
    low_quantile: float,
    high_quantile: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if not 0.0 <= low_quantile < high_quantile <= 1.0:
        raise ValueError("Require 0.0 <= low_quantile < high_quantile <= 1.0")

    low_threshold = float(frame["similarity_score"].quantile(low_quantile))
    high_threshold = float(frame["similarity_score"].quantile(high_quantile))

    def bucket_for_score(score: float) -> str:
        if score <= low_threshold:
            return "easy_non_match"
        if score >= high_threshold:
            return "easy_match"
        return "ambiguous"

    frame = frame.copy()
    frame["bucket"] = frame["similarity_score"].map(bucket_for_score)

    return frame, {
        "low_quantile": float(low_quantile),
        "high_quantile": float(high_quantile),
        "low_threshold": low_threshold,
        "high_threshold": high_threshold,
    }


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


def _write_bucket_csvs(frame: pd.DataFrame, output_dir: Path, dataset_key: str) -> Dict[str, str]:
    bucket_paths: Dict[str, str] = {}
    for bucket_name in BUCKET_NAMES:
        bucket_frame = frame[frame["bucket"] == bucket_name].copy()
        csv_path = output_dir / f"{dataset_key}_{bucket_name}.csv"
        bucket_frame.to_csv(csv_path, index=False)
        bucket_paths[bucket_name] = str(csv_path)
    return bucket_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Split entity matching pairs into Match Entity Records-style similarity buckets.")
    parser.add_argument(
        "--dataset-key",
        default="amgo",
        choices=sorted(CONSTANTS.DATASET_PATHS.keys()),
        help="Dataset key to split (default: amgo). Supported keys include waam.",
    )
    parser.add_argument("--partition", default="test", help="Dataset partition to use (default: test).")
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model used for similarity scoring.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    parser.add_argument("--low-quantile", type=float, default=0.33, help="Lower quantile for the easy non-match bucket.")
    parser.add_argument("--high-quantile", type=float, default=0.67, help="Upper quantile for the easy match bucket.")
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Base directory where split files and reports will be written.",
    )
    args = parser.parse_args()

    if args.dataset_key not in CONSTANTS.DATASET_PATHS:
        raise ValueError(
            f"Unsupported dataset key '{args.dataset_key}'. Available keys: {sorted(CONSTANTS.DATASET_PATHS.keys())}"
        )

    output_dir = Path(args.output_dir) / f"{args.dataset_key}_match_entity_records_split"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = _build_similarity_frame(
        args.dataset_key,
        args.partition,
        model_name=args.model_name,
        batch_size=args.batch_size,
    )
    frame, thresholds = _assign_buckets(
        frame,
        low_quantile=args.low_quantile,
        high_quantile=args.high_quantile,
    )

    bucket_paths = _write_bucket_csvs(frame, output_dir, args.dataset_key)
    bucket_summary = _bucket_summary(frame)

    report = {
        "metadata": {
            "dataset_key": args.dataset_key,
            "partition": args.partition,
            "model_name": args.model_name,
            "batch_size": args.batch_size,
            "low_quantile": args.low_quantile,
            "high_quantile": args.high_quantile,
            "thresholds": thresholds,
            "total_pairs": int(len(frame)),
        },
        "buckets": bucket_summary,
        "bucket_csv_paths": bucket_paths,
    }

    split_json_path = output_dir / f"{args.dataset_key}_{args.partition}_match_entity_records_split.json"
    with split_json_path.open("w", encoding="utf-8") as file_handle:
        json.dump(report, file_handle, ensure_ascii=False, indent=2)

    print(f"Saved bucket report to {split_json_path}")
    print("Bucket summary:")
    for bucket_name in BUCKET_NAMES:
        stats = bucket_summary[bucket_name]
        print(
            f"- {bucket_name}: count={stats['count']}, pos_rate={stats['positive_rate']:.3f}, "
            f"mean_sim={stats['mean_similarity']:.3f}"
        )


if __name__ == "__main__":
    main()
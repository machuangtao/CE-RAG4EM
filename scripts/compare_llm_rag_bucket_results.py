#!/usr/bin/env python3
"""Compare LLM-only and RAG bucket results against ground truth.

This script mirrors the decision parsing logic used by the bucket evaluation
scripts and reports how often RAG improves, harms, or leaves predictions
unchanged relative to the LLM-only baseline.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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


def _load_results_json(path: Path) -> Tuple[List[str], Dict[str, object]]:
    with path.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)

    results = payload.get("results", [])
    if not isinstance(results, list):
        raise ValueError(f"Expected 'results' to be a list in {path}")

    return [str(item) for item in results], payload.get("metadata", {})


def _resolve_ground_truth_csv(
    llm_results_path: Optional[Path] = None,
    rag_results_path: Optional[Path] = None,
    metadata: Optional[Dict[str, object]] = None,
    bucket_csv_path: Optional[Path] = None,
    dataset_key: Optional[str] = None,
    bucket_name: Optional[str] = None,
) -> Path:
    if bucket_csv_path is not None:
        return bucket_csv_path

    if dataset_key and bucket_name:
        candidate = REPO_ROOT / "output" / f"{dataset_key}_match_entity_records_split" / f"{dataset_key}_{bucket_name}.csv"
        if candidate.exists():
            return candidate

    if metadata and not dataset_key and not bucket_name:
        dataset_key = str(metadata.get("dataset_key", "") or "")
        bucket_name = str(metadata.get("bucket", "") or "")

    if dataset_key and bucket_name:
        candidate = REPO_ROOT / "output" / f"{dataset_key}_match_entity_records_split" / f"{dataset_key}_{bucket_name}.csv"
        if candidate.exists():
            return candidate

    if metadata:
        metadata_dataset_key = metadata.get("dataset_key")
        metadata_bucket = metadata.get("bucket")
        if metadata_dataset_key and metadata_bucket:
            candidate = REPO_ROOT / "output" / f"{metadata_dataset_key}_match_entity_records_split" / f"{metadata_dataset_key}_{metadata_bucket}.csv"
            if candidate.exists():
                return candidate

    default_candidates = [
        REPO_ROOT / "output" / "amgo_match_entity_records_split" / "amgo_ambiguous.csv",
        REPO_ROOT / "output" / "amgo_match_entity_records_split" / "amgo_test_ambiguous.csv",
    ]
    for candidate in default_candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate the ground-truth bucket CSV. Provide --bucket-csv explicitly or ensure the JSON metadata points to an existing bucket CSV."
    )


def _load_ground_truth_labels(csv_path: Path) -> pd.Series:
    frame = pd.read_csv(csv_path)
    if "label" not in frame.columns:
        raise ValueError(f"Ground-truth CSV is missing a 'label' column: {csv_path}")
    return frame["label"].astype(int)


def compare_results(
    llm_results_path: Path,
    rag_results_path: Path,
    bucket_csv_path: Optional[Path] = None,
    dataset_key: Optional[str] = None,
    bucket_name: Optional[str] = None,
) -> Dict[str, object]:
    llm_results, llm_metadata = _load_results_json(llm_results_path)
    rag_results, rag_metadata = _load_results_json(rag_results_path)

    metadata = llm_metadata or rag_metadata or {}
    ground_truth_csv = _resolve_ground_truth_csv(
        llm_results_path=llm_results_path,
        rag_results_path=rag_results_path,
        metadata=metadata,
        bucket_csv_path=bucket_csv_path,
        dataset_key=dataset_key,
        bucket_name=bucket_name,
    )
    ground_truth = _load_ground_truth_labels(ground_truth_csv)

    shared_len = min(len(llm_results), len(rag_results), len(ground_truth))
    llm_predictions = [
        _parse_binary_decision(prediction)
        for prediction in llm_results[:shared_len]
    ]
    rag_predictions = [
        _parse_binary_decision(prediction)
        for prediction in rag_results[:shared_len]
    ]
    gold_labels = ground_truth.iloc[:shared_len].to_numpy()

    llm_correct = [pred == gold for pred, gold in zip(llm_predictions, gold_labels)]
    rag_correct = [pred == gold for pred, gold in zip(rag_predictions, gold_labels)]

    total = int(shared_len)
    both_correct = int(sum(1 for a, b in zip(llm_correct, rag_correct) if a and b))
    rag_success = int(sum(1 for a, b in zip(llm_correct, rag_correct) if (not a) and b))
    rag_failure = int(sum(1 for a, b in zip(llm_correct, rag_correct) if a and (not b)))
    both_wrong = int(sum(1 for a, b in zip(llm_correct, rag_correct) if (not a) and (not b)))

    success_rate = rag_success / total * 100.0 if total else 0.0
    failure_rate = rag_failure / total * 100.0 if total else 0.0
    net_gain = (rag_success - rag_failure) / total * 100.0 if total else 0.0

    return {
        "metadata": {
            "llm_results_path": str(llm_results_path),
            "rag_results_path": str(rag_results_path),
            "ground_truth_csv": str(ground_truth_csv),
            "total_pairs": total,
            "dataset_key": dataset_key or metadata.get("dataset_key", ""),
            "bucket": bucket_name or metadata.get("bucket", ""),
        },
        "counts": {
            "both_correct": both_correct,
            "rag_success": rag_success,
            "rag_failure": rag_failure,
            "both_wrong": both_wrong,
        },
        "rates": {
            "success_rate": success_rate,
            "failure_rate": failure_rate,
            "net_gain": net_gain,
        },
    }


def _extract_timestamp_from_path(path: Path) -> str:
    match = re.search(r"(\d{8}_\d{6})(?=\.json$)", path.name)
    return match.group(1) if match else ""


def _resolve_llm_result_file(llm_dir: Path, dataset_key: str, partition: str, bucket_name: str) -> Path:
    if not llm_dir.exists():
        raise FileNotFoundError(f"LLM results directory not found: {llm_dir}")

    candidates = []
    for path in llm_dir.glob("*.json"):
        name = path.name
        if not name.startswith(f"{dataset_key}_{partition}_{bucket_name}_"):
            continue
        if "_llm" not in name:
            continue
        candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"No LLM result file found for bucket '{bucket_name}' in {llm_dir}. "
            f"Expected pattern like {dataset_key}_{partition}_{bucket_name}_*_llm*.json"
        )

    candidates.sort(key=lambda p: (_extract_timestamp_from_path(p), p.stat().st_mtime), reverse=True)
    return candidates[0]


def _resolve_rag_result_files(
    rag_dir: Path,
    dataset_key: str,
    partition: str,
    bucket_name: str,
    num_runs: int,
) -> List[Path]:
    if not rag_dir.exists():
        raise FileNotFoundError(f"RAG results directory not found: {rag_dir}")

    candidates = []
    for path in rag_dir.glob("*.json"):
        name = path.name
        if not name.startswith(f"{dataset_key}_{partition}_{bucket_name}_"):
            continue
        if "_blocking_rag_" not in name:
            continue
        candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"No blocking-RAG result files found for bucket '{bucket_name}' in {rag_dir}. "
            f"Expected pattern like {dataset_key}_{partition}_{bucket_name}_*_blocking_rag_*.json"
        )

    candidates.sort(key=lambda p: (_extract_timestamp_from_path(p), p.stat().st_mtime), reverse=True)
    return candidates[:num_runs]


def compare_latest_runs_by_bucket(
    dataset_key: str,
    partition: str,
    llm_dir: Path,
    rag_dir: Path,
    num_runs: int,
    output_json: Optional[Path] = None,
) -> Dict[str, object]:
    if num_runs <= 0:
        raise ValueError("--num-runs must be greater than 0")

    bucket_reports: Dict[str, Dict[str, object]] = {}
    for bucket_name in BUCKET_NAMES:
        llm_file = _resolve_llm_result_file(llm_dir, dataset_key, partition, bucket_name)
        rag_files = _resolve_rag_result_files(rag_dir, dataset_key, partition, bucket_name, num_runs)

        run_summaries = [
            compare_results(
                llm_results_path=llm_file,
                rag_results_path=rag_file,
                dataset_key=dataset_key,
                bucket_name=bucket_name,
            )
            for rag_file in rag_files
        ]

        run_count = len(run_summaries)
        avg_success_rate = sum(item["rates"]["success_rate"] for item in run_summaries) / run_count if run_count else 0.0
        avg_failure_rate = sum(item["rates"]["failure_rate"] for item in run_summaries) / run_count if run_count else 0.0
        avg_net_gain = sum(item["rates"]["net_gain"] for item in run_summaries) / run_count if run_count else 0.0
        avg_rag_success = sum(item["counts"]["rag_success"] for item in run_summaries) / run_count if run_count else 0.0
        avg_rag_failure = sum(item["counts"]["rag_failure"] for item in run_summaries) / run_count if run_count else 0.0

        bucket_reports[bucket_name] = {
            "llm_results_path": str(llm_file),
            "rag_results_paths": [str(path) for path in rag_files],
            "runs_compared": run_count,
            "averages": {
                "rag_success_count": avg_rag_success,
                "rag_failure_count": avg_rag_failure,
                "rag_success_rate": avg_success_rate,
                "rag_failure_rate": avg_failure_rate,
                "net_gain": avg_net_gain,
            },
            "runs": run_summaries,
        }

    overall_avg_success = sum(report["averages"]["rag_success_rate"] for report in bucket_reports.values()) / len(BUCKET_NAMES)
    overall_avg_failure = sum(report["averages"]["rag_failure_rate"] for report in bucket_reports.values()) / len(BUCKET_NAMES)
    overall_avg_net = sum(report["averages"]["net_gain"] for report in bucket_reports.values()) / len(BUCKET_NAMES)

    summary = {
        "metadata": {
            "dataset_key": dataset_key,
            "partition": partition,
            "num_runs_requested": num_runs,
            "llm_dir": str(llm_dir),
            "rag_dir": str(rag_dir),
        },
        "per_bucket": bucket_reports,
        "overall_bucket_average_rates": {
            "rag_success_rate": overall_avg_success,
            "rag_failure_rate": overall_avg_failure,
            "net_gain": overall_avg_net,
        },
    }

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with output_json.open("w", encoding="utf-8") as file_handle:
            json.dump(summary, file_handle, ensure_ascii=False, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LLM-only and RAG bucket results against the bucket ground truth.")
    parser.add_argument("--llm-results", default="", help="Path to one LLM results JSON file (single comparison mode).")
    parser.add_argument("--rag-results", default="", help="Path to one RAG results JSON file (single comparison mode).")
    parser.add_argument(
        "--bucket-csv",
        default="",
        help="Optional path to the bucket CSV containing the ground-truth labels.",
    )
    parser.add_argument("--dataset-key", default="amgo", help="Dataset key for automatic comparison mode.")
    parser.add_argument("--partition", default="test", help="Dataset partition for automatic comparison mode.")
    parser.add_argument("--num-runs", type=int, default=3, help="How many latest RAG runs to compare per bucket in automatic mode.")
    parser.add_argument(
        "--llm-dir",
        default="",
        help="Directory for LLM bucket result files. Defaults to output/{dataset_key}_split_llm_results.",
    )
    parser.add_argument(
        "--rag-dir",
        default="",
        help="Directory for blocking-RAG bucket result files. Defaults to output/split_{dataset_key}_blocking_rag_results.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to save the comparison summary as JSON.",
    )
    args = parser.parse_args()

    bucket_csv_path = Path(args.bucket_csv).resolve() if args.bucket_csv else None
    output_json_path = Path(args.output_json).resolve() if args.output_json else None

    if args.llm_results and args.rag_results:
        llm_results_path = Path(args.llm_results).resolve()
        rag_results_path = Path(args.rag_results).resolve()

        summary = compare_results(
            llm_results_path,
            rag_results_path,
            bucket_csv_path=bucket_csv_path,
            dataset_key=args.dataset_key,
        )

        print("LLM vs RAG bucket comparison")
        print(f"- total pairs: {summary['metadata']['total_pairs']}")
        print(f"- both correct: {summary['counts']['both_correct']}")
        print(f"- RAG success (LLM wrong, RAG right): {summary['counts']['rag_success']} ({summary['rates']['success_rate']:.2f}%)")
        print(f"- RAG failure (LLM right, RAG wrong): {summary['counts']['rag_failure']} ({summary['rates']['failure_rate']:.2f}%)")
        print(f"- both wrong: {summary['counts']['both_wrong']}")
        print(f"- net gain: {summary['rates']['net_gain']:.2f}%")

        if output_json_path is not None:
            output_json_path.parent.mkdir(parents=True, exist_ok=True)
            with output_json_path.open("w", encoding="utf-8") as file_handle:
                json.dump(summary, file_handle, ensure_ascii=False, indent=2)
            print(f"Saved summary to {output_json_path}")
        return

    llm_dir = Path(args.llm_dir).resolve() if args.llm_dir else (REPO_ROOT / "output" / f"{args.dataset_key}_split_llm_results")
    rag_dir = Path(args.rag_dir).resolve() if args.rag_dir else (REPO_ROOT / "output" / f"split_{args.dataset_key}_blocking_rag_results")

    summary = compare_latest_runs_by_bucket(
        dataset_key=args.dataset_key,
        partition=args.partition,
        llm_dir=llm_dir,
        rag_dir=rag_dir,
        num_runs=args.num_runs,
        output_json=output_json_path,
    )

    print(f"Automatic comparison for dataset={args.dataset_key}, partition={args.partition}")
    for bucket_name in BUCKET_NAMES:
        bucket_summary = summary["per_bucket"][bucket_name]
        averages = bucket_summary["averages"]
        print(
            f"- {bucket_name}: runs={bucket_summary['runs_compared']}, "
            f"avg_rag_success={averages['rag_success_count']:.2f} ({averages['rag_success_rate']:.2f}%), "
            f"avg_rag_failure={averages['rag_failure_count']:.2f} ({averages['rag_failure_rate']:.2f}%), "
            f"avg_net_gain={averages['net_gain']:.2f}%"
        )

    overall = summary["overall_bucket_average_rates"]
    print(
        f"Overall bucket-average rates: success={overall['rag_success_rate']:.2f}%, "
        f"failure={overall['rag_failure_rate']:.2f}%, net_gain={overall['net_gain']:.2f}%"
    )

    if output_json_path is not None:
        print(f"Saved summary to {output_json_path}")


if __name__ == "__main__":
    main()

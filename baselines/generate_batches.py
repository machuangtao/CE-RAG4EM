import argparse
from pathlib import Path

import pandas as pd

from data_utils.data_handler import _load_raw_data, entity_to_text

BLOCKING_ATTRIBUTES = {
    "abt": ["name", "description"],
    "amgo": ["title", "manufacturer"],
    "beer": ["Beer_Name", "Brew_Factory_Name", "Style"],
    "dbac": ["title", "authors", "venue"],
    "dbgo": ["title", "authors", "venue"],
    "foza": ["name", "addr", "city", "type"],
    "itam": ["Song_Name", "Artist_Name", "Album_Name", "CopyRight"],
    "waam": ["title", "category", "brand", "modelino"],
    "wdc": ["brand", "title", "description"],
}

Q = 6


def qgrams(text: str, q: int = Q) -> set:
    """Character q-grams of every whitespace token in text (short tokens kept whole)."""
    grams = set()
    for token in text.lower().split():
        if len(token) < q:
            grams.add(token)
        else:
            grams.update(token[i:i + q] for i in range(len(token) - q + 1))
    return grams


def entity_qgrams(table: pd.DataFrame, attributes: list) -> dict:
    result = {}
    for _, row in table.iterrows():
        text = " ".join(str(row[a]) for a in attributes if pd.notnull(row.get(a)))
        result[int(row["id"])] = qgrams(text)
    return result


def assign_block_ids(table_a: pd.DataFrame, table_b: pd.DataFrame, ground_truth_df: pd.DataFrame, attributes: list) -> dict:
    """Map each ground-truth pair to a block_id: the smallest q-gram it shares between its two entities."""
    qgrams_a = entity_qgrams(table_a, attributes)
    qgrams_b = entity_qgrams(table_b, attributes)

    pair_to_block = {}
    for row in ground_truth_df.itertuples():
        ltable_id, rtable_id = int(row.ltable_id), int(row.rtable_id)
        shared = qgrams_a.get(ltable_id, set()) & qgrams_b.get(rtable_id, set())
        block_id = min(shared) if shared else f"solo_{ltable_id}_{rtable_id}"
        pair_to_block[(ltable_id, rtable_id)] = block_id

    return pair_to_block


def create_batches(pair_to_block: dict, batch_size: int) -> list:
    """Group pairs by block_id, splitting blocks bigger than batch_size sequentially (blocks are never merged)."""
    pairs_by_block = {}
    for pair, block_id in pair_to_block.items():
        pairs_by_block.setdefault(block_id, []).append(pair)

    batches = []
    for pairs in pairs_by_block.values():
        for start in range(0, len(pairs), batch_size):
            batches.append(pairs[start:start + batch_size])
    return batches


def main(dataset: str, batch_size: int, partition: str = "test"):
    attributes = BLOCKING_ATTRIBUTES[dataset]

    table_a, table_b, ground_truth_df = _load_raw_data(dataset, partition)
    table_a_by_id = table_a.set_index("id")
    table_b_by_id = table_b.set_index("id")
    ground_truth = {(int(r.ltable_id), int(r.rtable_id)): int(r.label) for r in ground_truth_df.itertuples()}

    pair_to_block = assign_block_ids(table_a, table_b, ground_truth_df, attributes)
    batches = create_batches(pair_to_block, batch_size)

    rows = []
    for batch_id, pairs in enumerate(batches):
        for ltable_id, rtable_id in pairs:
            rows.append({
                "dataset": dataset,
                "batch_size": batch_size,
                "batch_id": batch_id,
                "pair_id": f"{ltable_id}_{rtable_id}",
                "record_1": entity_to_text(table_a_by_id.loc[ltable_id]),
                "record_2": entity_to_text(table_b_by_id.loc[rtable_id]),
                "ground_truth": ground_truth[(ltable_id, rtable_id)],
            })

    output_dir = Path(f"batch_outputs/{dataset}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset}_{partition}_QG_{batch_size}_batches.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)

    print(f"{len(batches)} batches, {len(rows)} pairs -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate q-gram-blocked batches for a dataset's test set")
    parser.add_argument("--dataset", "-d", required=True, help='Dataset name (e.g. "abt", "amgo", "beer", "dbac", "dbgo", "foza", "itam", "waam", "wdc")')
    parser.add_argument("--batch_size", "-b", type=int, required=True, help="Number of pairs per batch")
    parser.add_argument("--partition", "-p", default="test", choices=["train", "test", "valid"], help='Data partition (default: "test")')
    args = parser.parse_args()

    main(dataset=args.dataset, batch_size=args.batch_size, partition=args.partition)

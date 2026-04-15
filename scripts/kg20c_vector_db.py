import argparse
import json
import os
import sys
import pandas as pd


from data_utils.kg20c_astra_vector import (
    fetch_and_save_relevant_kg20c_entities,
    index_kg20c_entities_to_astra,
    retrieve_relevant_kg20c_entities,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="KG20C Astra vector DB indexing and retrieval")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="Embed KG20C entities and upsert into Astra DB")
    p_index.add_argument("--entity-info-path", default="data/raw/KG20C/all_entity_info.txt")
    p_index.add_argument("--collection", default="kg20c_entities_v1")
    p_index.add_argument("--embedding-dim", type=int, default=1024)
    p_index.add_argument("--model", default="jinaai/jina-embeddings-v3")
    p_index.add_argument("--batch-size", type=int, default=64)
    p_index.add_argument(
        "--recreate-collection",
        action="store_true",
        help="Delete and recreate collection as vector-enabled if it already exists but is non-vector.",
    )

    p_query = sub.add_parser("query", help="Retrieve top-k KG20C entities for one textual query")
    p_query.add_argument("--text", required=True)
    p_query.add_argument("--top-k", type=int, default=10)
    p_query.add_argument("--collection", default="kg20c_entities_v1")
    p_query.add_argument("--embedding-dim", type=int, default=1024)
    p_query.add_argument("--model", default="jinaai/jina-embeddings-v3")

    p_batch = sub.add_parser("batch-query", help="Batch query from a CSV/TSV file")
    p_batch.add_argument("--input", required=True, help="CSV/TSV with id and query columns")
    p_batch.add_argument("--id-column", default="id")
    p_batch.add_argument("--query-column", default="query")
    p_batch.add_argument("--output", required=True)
    p_batch.add_argument("--top-k", type=int, default=10)
    p_batch.add_argument("--collection", default="kg20c_entities_v1")
    p_batch.add_argument("--embedding-dim", type=int, default=1024)
    p_batch.add_argument("--model", default="jinaai/jina-embeddings-v3")

    args = parser.parse_args()

    if args.cmd == "index":
        index_kg20c_entities_to_astra(
            entity_info_path=args.entity_info_path,
            collection_name=args.collection,
            embedding_dim=args.embedding_dim,
            model_name=args.model,
            batch_size=args.batch_size,
            recreate_collection=args.recreate_collection,
        )
        return

    if args.cmd == "query":
        results = retrieve_relevant_kg20c_entities(
            query_text=args.text,
            top_k=args.top_k,
            collection_name=args.collection,
            embedding_dim=args.embedding_dim,
            model_name=args.model,
        )
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return

    sep = "\t" if args.input.endswith(".tsv") else ","
    query_df = pd.read_csv(args.input, sep=sep)
    fetch_and_save_relevant_kg20c_entities(
        query_df=query_df,
        id_column=args.id_column,
        query_column=args.query_column,
        output_path=args.output,
        top_k=args.top_k,
        collection_name=args.collection,
        embedding_dim=args.embedding_dim,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()

import json
import os
from typing import Optional, Literal, List, Dict, Any

import pandas as pd
from .data_handler import (prepare_structured_data,
                          structured_to_text_data,
                          prepare_text_data)
from .wiki_query import (fetch_and_save_relevant_ids,
                        fetch_and_save_wikidata,
                        fetch_and_save_triplets)


def query_generation(
        dataset_key: str,
        partition: str,
        query_type: Literal["entity", "pair"],
        drop_columns: Optional[Literal[list, str]] = None,
):
    """
    Generate queries for entity matching based on the given dataset and parameters.

    This function processes the dataset and creates queries for either individual entities
    or entity pairs, depending on the specified query type.

    Args:
        dataset_key (str): The identifier for the dataset to be used.
        partition (str): The partition of the dataset to be used (e.g., "train", "test", "valid").
        query_type (Literal["entity", "pair"]): The type of query to generate.
            "entity" generates separate queries for each entity, while "pair" generates
            a single query for each entity pair.
        drop_columns (Optional[Literal[list, str]], optional): Columns to be dropped from the dataset.
            Can be a list of column names or a string for a single column. Defaults to None.

    Returns:
        pd.DataFrame: A DataFrame containing the generated queries.
            For "entity" query_type: Columns: ["queryA", "queryB", "ltable_id", "rtable_id"]
            For "pair" query_type: Columns: ["query", "pair_id"]
    """
    if drop_columns:
        data_df = prepare_structured_data(dataset_key, partition)
        kept_columns = data_df.columns.tolist()
        kept_columns = [
            col for col in kept_columns
            if col.rsplit("_", 1)[0] not in drop_columns
        ]
        data_df = structured_to_text_data(data_df[kept_columns], with_semantic=False)
    else:
        data_df = prepare_text_data(dataset_key, partition, with_semantic=False)

    def _query_form(row, qt):
        if qt == "entity":
            row["queryA"] = f"What is {row['entityA']}"
            row["queryB"] = f"What is {row['entityB']}"
            return row[["queryA", "queryB", "ltable_id", "rtable_id"]]
        else:
            row["query"] = f"What are {row['entityA']}, and {row['entityB']}?"
            row["pair_id"] = f"{row['ltable_id']}-{row['rtable_id']}"
            return row[["query", "pair_id"]]

    query_df = data_df.apply(lambda row: _query_form(row, query_type), axis=1)
    return query_df

def query_generation_from_block(
        dataset_key: str,
        partition: str,
        query_type: Literal["entity", "pair"],
        entity_a_text: str,
        entity_b_text: str,
):
    """ #TODO: fine during experimental phase, but should be overwritten before releasing
    Generate queries for entity matching based on the given dataset and parameters.

    This function processes the dataset and creates queries for either individual entities
    or entity pairs, depending on the specified query type.

    Args:
        dataset_key (str): The identifier for the dataset to be used.
        partition (str): The partition of the dataset to be used (e.g., "train", "test", "valid").
        query_type (Literal["entity", "pair"]): The type of query to generate.
            "entity" generates separate queries for each entity, while "pair" generates
            a single query for each entity pair.
        entity_a_text (str): The text representation of entity A.
        entity_b_text (str): The text representation of entity B.

    Returns:
        str: The generated query string.
            For "entity" query_type: Returns a tuple of two strings (queryA, queryB).
            For "pair" query_type: Returns a single string (query).
    """

    if query_type == "entity":
        queryA = f"What is {entity_a_text}"
        queryB = f"What is {entity_b_text}"
        return queryA, queryB
    else:
        query = f"What are {entity_a_text}, and {entity_b_text}?"
        return query


def _merge_relevant_ids_with_info(
    relevant_ids_path: str,
    pids_path: str,
    qids_path: str,
    triplets_path: str,
) -> None:
    """
    Merge relevant IDs with label, description, and concatenated pretty_string.
    Updates the original relevant_ids_path file in place.
    """
    # Load all files
    with open(relevant_ids_path, encoding="utf-8") as f:
        relevant_ids = json.load(f)
    with open(pids_path, encoding="utf-8") as f:
        pids = json.load(f)
    with open(qids_path, encoding="utf-8") as f:
        qids = json.load(f)
    with open(triplets_path, encoding="utf-8") as f:
        triplets = json.load(f)

    def enrich_item(item: Dict[str, Any]) -> Dict[str, Any]:
        key_type = "QID" if "QID" in item else "PID"
        id_value = item[key_type]

        # Look up label and description
        info = qids.get(id_value, {}) if key_type == "QID" else pids.get(id_value, {})
        label = info.get("label", "")
        description = info.get("description", "")

        # Gather pretty strings from triplets.json
        pretty_list = triplets.get(id_value, [])
        pretty_str = "\n".join(p.get("pretty_string", "") for p in pretty_list)

        return {
            **item,  # keep similarity_score and other keys
            "label": label,
            "description": description,
            "pretty_string": pretty_str,
        }

    # Enrich relevant_qids
    for main_id, items in relevant_ids.get("relevant_qids", {}).items():
        relevant_ids["relevant_qids"][main_id] = [enrich_item(it) for it in items] if items is not None else []

    # Enrich relevant_pids
    for main_id, items in relevant_ids.get("relevant_pids", {}).items():
        relevant_ids["relevant_pids"][main_id] = [enrich_item(it) for it in items] if items is not None else []

    # Save back to the same file
    with open(relevant_ids_path, "w", encoding="utf-8") as f:
        json.dump(relevant_ids, f, ensure_ascii=False, indent=2)

    print(f"Updated {relevant_ids_path} with labels, descriptions, and pretty strings.")


def wiki_query_execution(
        dataset_key: str,
        partition: str,
        drop_columns: Optional[Literal[List, str]] = None,
        save_path_dir: str = ".",
):
    """
        Execute Wikidata queries and save relevant entity information.

        Args:
            dataset_key (str): The identifier for the dataset to be used.
            partition (str): The partition of the dataset ("train", "valid", "test").
            drop_columns (Optional[Literal[list, str]]): Columns to exclude from processing.
                Can be either a list of column names or a single column name.
            save_path_dir (str, optional): Directory path to save the output files.
                If None, uses default path. Defaults to None.

        Saves:
            - entity_a_relevant_ids_{partition}.json: Relevance info for first entities
            - entity_b_relevant_ids_{partition}.json: Relevance info for second entities
            - pair_relevant_ids_{partition}.json: Relevance info for entity pairs

            - qids_{partition}.json: Information about relevant QIDs
            - pids_{partition}.json: Information about relevant PIDs
            - triplets_{partition}.json: Triplet information for relevant PIDs/QIDs

        Notes:
            - Creates necessary directories if they don't exist
            - Skips fetching if output files already exist
            - Uses query_generation() to create initial queries

        Example:
            >>> wiki_query_execution(
            ...     dataset_key="AMGO",
            ...     partition="test",
            ...     query_type="entity",
            ...     drop_columns=["price"],
            ...     save_path_dir="data/query/amgo"
            ... )
        """
    def _query_execution_for_relevance(q_df, cols):
        q_df = q_df[cols]
        q_df = q_df.drop_duplicates()
        if "queryA" in cols:
            save_path = save_path_dir + f"/entity_a_relevant_ids_{partition}.json"
        elif "queryB" in cols:
            save_path = save_path_dir + f"/entity_b_relevant_ids_{partition}.json"
        else:
            save_path = save_path_dir + f"/pair_relevant_ids_{partition}.json"

        if not os.path.exists(save_path):
            r_dict = fetch_and_save_relevant_ids(q_df, cols[1], cols[0], save_path)
        else:
            r_dict = json.load(open(save_path))

        return r_dict, save_path

    def _extract_ids(r_dict):
        all_qids = [
            item["QID"]
            for items in r_dict.get("relevant_qids", {}).values()
            if items is not None
            for item in items
        ]
        all_pids = [
            item["PID"]
            for items in r_dict.get("relevant_pids", {}).values()
            if items is not None
            for item in items
        ]
        return all_pids, all_qids

    os.makedirs(save_path_dir, exist_ok=True)

    ## the pids, qids file generation
    entity_query_df = query_generation(dataset_key, partition, "entity", drop_columns)
    pair_query_df = query_generation(dataset_key, partition, "pair", drop_columns)
    print(f"wiki query execution for dataset {dataset_key} {partition}:", flush=True)
    print(f"step 1: query for entity A", flush=True)
    relevance_a, relevance_a_save_path = _query_execution_for_relevance(entity_query_df, ["queryA", "ltable_id"])
    print(f"step 2: query for entity B", flush=True)
    relevance_b, relevance_b_save_path = _query_execution_for_relevance(entity_query_df, ["queryB", "rtable_id"])
    print(f"step 3: query for entity pair (A + B)", flush=True)
    relevance_pair, relevance_pair_save_path = _query_execution_for_relevance(pair_query_df, ["query", "pair_id"])

    pids_a, qids_a = _extract_ids(relevance_a)
    pids_b, qids_b = _extract_ids(relevance_b)
    pids_pair, qids_pair = _extract_ids(relevance_pair)

    pids = list(set(pids_a + pids_b + pids_pair))
    qids = list(set(qids_a + qids_b + qids_pair))

    qids_save_path = save_path_dir + f"/qids_{partition}.json"
    pids_save_path = save_path_dir + f"/pids_{partition}.json"
    print(f"step 4: fetch label and description for all qids", flush=True)
    if not os.path.exists(qids_save_path):
        fetch_and_save_wikidata(qids, qids_save_path)
    print(f"step 5: fetch label and description for all pids", flush=True)
    if not os.path.exists(pids_save_path):
        fetch_and_save_wikidata(pids, pids_save_path)

    triplets_save_path = save_path_dir + f"/triplets_{partition}.json"
    ids = pids + qids
    print(f"step 6: fetch triplets for all qids and pids", flush=True)
    if not os.path.exists(triplets_save_path):
        fetch_and_save_triplets(ids, triplets_save_path)

    print(f"step 7: merge results within the same file for further usage", flush=True)
    _merge_relevant_ids_with_info(relevance_a_save_path, pids_save_path, qids_save_path, triplets_save_path)
    _merge_relevant_ids_with_info(relevance_b_save_path, pids_save_path, qids_save_path, triplets_save_path)
    _merge_relevant_ids_with_info(relevance_pair_save_path, pids_save_path, qids_save_path, triplets_save_path)

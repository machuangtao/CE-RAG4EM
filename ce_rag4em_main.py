import argparse
import json
from typing import Optional, List, Dict, Union, Any, Tuple
import logging
from datetime import datetime
import time
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from pathlib import Path
from data_utils import PROMPT_TEMPLATES
from data_utils.wiki_query import fetch_and_save_wikidata, fetch_and_save_triplets
from data_utils.bfs_query import bfs_search_with_entity
from model_utils import (EntityMatchPrompt, process_openai_requests, process_local_request, process_gemini_requests)
from sklearn.metrics import f1_score, confusion_matrix, precision_score, recall_score



logger = logging.getLogger(__name__)

def setup_logging(dataset: str, partition: str, context_config: Dict[str, Union[str, int]], model: str, triple_generation_type: Optional[str], timestamp: str) -> str:
    """Set up logging to file and console. Returns the log file path."""
    global logger
    
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # Determine log file path based on context config
    if context_config and context_config.get("enabled"):
        context_type = context_config.get("context_type")
        top_k = context_config.get("top_k")
        if context_type == "triple":
            log_file_path = os.path.join(log_dir, f"{dataset}/{dataset}_{partition}_{top_k}_{triple_generation_type}_{model}_rag_generation_{timestamp}.log")
        else:
            log_file_path = os.path.join(log_dir, f"{dataset}/{dataset}_{partition}_{top_k}_{context_type}_{model}_rag_generation_{timestamp}.log")
    elif context_config and not context_config.get("enabled"):
        log_file_path = os.path.join(log_dir, f"{dataset}/{dataset}_{partition}_llm_generation_{timestamp}.log")
    else:
        log_file_path = os.path.join(log_dir, f"{dataset}/{dataset}_{partition}_generation_{timestamp}.log")

    # Ensure the log directory (including dataset subfolder) exists
    log_file_parent = Path(log_file_path).parent
    log_file_parent.mkdir(parents=True, exist_ok=True)

    # Configure the root logger so both module-level and root-level calls are captured
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Rebind the module logger to the configured root handlers
    logger = logging.getLogger(__name__)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    logger.propagate = True

    logger.info(f"Logging initialized. Log file: {log_file_path}")
    return log_file_path

def load_group_retrieval_results(file_path: str) -> Dict:
    """Load the group retrieval results JSON file"""
    logger.info(f"Loading group retrieval results from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"Loaded data with {len(data.get('blocks', {}))} blocks from group retrieval results")
        return data
    except Exception as e:
        logger.error(f"Error loading group retrieval results: {e}")
        raise

def extract_ids_from_block(block_data: Dict, top_k: int = 3) -> Tuple[List[str], List[str]]:
    """Extract QID and PID strings from block data with similarity scores"""
    qids = []
    pids = []
    
    # Extract QIDs
    relevant_qids = block_data.get('relevant_qids')
    if relevant_qids and isinstance(relevant_qids, list):
        qids = [item['QID'] for item in relevant_qids[:top_k] if isinstance(item, dict) and 'QID' in item]
    
    # Extract PIDs
    relevant_pids = block_data.get('relevant_pids')
    if relevant_pids and isinstance(relevant_pids, list):
        pids = [item['PID'] for item in relevant_pids[:top_k] if isinstance(item, dict) and 'PID' in item]
    
    logger.debug(f"Extracted {len(qids)} QIDs and {len(pids)} PIDs from block")
    return qids, pids



def generate_triples_for_block(block_id: str, block_data: Dict, dataset: str, top_k: int, triple_id_type: str = "QID", triple_generation_type: Optional[str] = None, top_k_entities: int = 3, blocking_method: str = "QG") -> List[Dict[str, Any]]:
    """Generate triples for a block using async functions"""
    
    # Extract IDs from block data
    qids, pids = extract_ids_from_block(block_data, top_k_entities)
    
    # Only process QIDs for triples - PIDs are properties, not entities
    if triple_id_type == "QID":
        entities_to_process = qids[:top_k_entities]
    elif triple_id_type == "PID":
        entities_to_process = pids[:top_k_entities]
    else:
        logger.error(f"Invalid triple_id_type: {triple_id_type}")
        return []

    logger.info(f"Getting triples for {len(entities_to_process)} entities in block {block_id}: {entities_to_process}")

    # Fetch and save triples
    output_path = f"retrieval_outputs/{dataset}/temp/triples_{triple_id_type}_{triple_generation_type}_{top_k_entities}_{block_id}_{blocking_method}.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if triple_generation_type == "EXP":
        fetch_and_save_triplets(entities_to_process, output_path)
        # Load the fetched triples
        with open(output_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
    elif triple_generation_type == "BFS":
        # BFS search returns results directly, no need to save/load from file
        bfs_results = bfs_search_with_entity(entities_to_process, max_depth=5)
        
        # Collect all unique IDs for enrichment
        all_ids = set()
        for entity_id, paths in bfs_results.get('single_entity_paths', {}).items():
            for path in paths:
                for step in path:
                    if isinstance(step, tuple) and len(step) == 3:
                        subj, pred, obj = step
                        all_ids.add(subj)
                        all_ids.add(pred)
                        if isinstance(obj, str) and (obj.startswith('Q') or obj.startswith('P')):
                            all_ids.add(obj)
        for pair_key, paths in bfs_results.get('entity_pair_paths', {}).items():
            for path in paths:
                for step in path:
                    if isinstance(step, tuple) and len(step) == 3:
                        subj, pred, obj = step
                        all_ids.add(subj)
                        all_ids.add(pred)
                        if isinstance(obj, str) and (obj.startswith('Q') or obj.startswith('P')):
                            all_ids.add(obj)
        
        # Fetch enrichment data for all IDs
        enrichment_path = f"retrieval_outputs/{dataset}/temp/bfs_enrichment_{block_id}_{blocking_method}.json"
        id_info = {}
        if all_ids:
            try:
                fetch_and_save_wikidata(list(all_ids), enrichment_path)
                with open(enrichment_path, 'r', encoding='utf-8') as f:
                    id_info = json.load(f)
                logger.info(f"Enriched {len(id_info)} IDs for BFS triples in block {block_id}")
            except Exception as e:
                logger.warning(f"Failed to enrich BFS triples for block {block_id}: {e}")
        
        # Convert BFS results to enriched format, preserving path structure
        results = {}
        # Combine single entity paths and pair paths - store enriched paths
        for entity_id, paths in bfs_results.get('single_entity_paths', {}).items():
            enriched_paths = []
            for path in paths:
                enriched_path = []
                for step in path:
                    if isinstance(step, tuple) and len(step) == 3:
                        subj, pred, obj = step
                        subj_info = id_info.get(subj, {})
                        pred_info = id_info.get(pred, {})
                        obj_is_id = isinstance(obj, str) and (obj.startswith('Q') or obj.startswith('P'))
                        obj_info = id_info.get(obj, {}) if obj_is_id else {}
                        
                        enriched_step = {
                            "subject": subj,
                            "predicate": pred,
                            "object": obj,
                            "subject_label": subj_info.get('label', subj),
                            "predicate_label": pred_info.get('label', pred),
                            "object_label": obj_info.get('label', obj) if obj_is_id else str(obj)
                        }
                        enriched_path.append(enriched_step)
                enriched_paths.append(enriched_path)
            results[entity_id] = {"triples": enriched_paths, "type": "single"}
        
        for pair_key, paths in bfs_results.get('entity_pair_paths', {}).items():
            enriched_paths = []
            for path in paths:
                enriched_path = []
                for step in path:
                    if isinstance(step, tuple) and len(step) == 3:
                        subj, pred, obj = step
                        subj_info = id_info.get(subj, {})
                        pred_info = id_info.get(pred, {})
                        obj_is_id = isinstance(obj, str) and (obj.startswith('Q') or obj.startswith('P'))
                        obj_info = id_info.get(obj, {}) if obj_is_id else {}
                        
                        enriched_step = {
                            "subject": subj,
                            "predicate": pred,
                            "object": obj,
                            "subject_label": subj_info.get('label', subj),
                            "predicate_label": pred_info.get('label', pred),
                            "object_label": obj_info.get('label', obj) if obj_is_id else str(obj)
                        }
                        enriched_path.append(enriched_step)
                enriched_paths.append(enriched_path)
            results[pair_key] = {"triples": enriched_paths, "type": "pair"}
        
        # Save enriched BFS results to file for consistency
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    else:
        logger.error(f"Invalid triple_generation_type type: {triple_generation_type}")
        return []
    
    logger.info(f"Loaded {len(results)} entities/pairs with triples from {output_path}")

    # Collect all triples
    all_triples = []
    for entity_id, data in results.items():
        if isinstance(data, Exception):
            logger.error(f"Error processing entity {entity_id}: {data}")
            continue
        
        # Handle BFS format with paths structure
        if isinstance(data, dict) and "triples" in data:
            paths = data["triples"]
            # Process each path (list of enriched steps)
            for path in paths:
                if isinstance(path, list) and len(path) > 0:
                    # Format multi-hop path with arrows using labels
                    path_triples = []
                    for step in path:
                        if isinstance(step, dict):
                            # Enriched format
                            subj_label = step.get('subject_label', step.get('subject', ''))
                            pred_label = step.get('predicate_label', step.get('predicate', ''))
                            obj_label = step.get('object_label', step.get('object', ''))
                            path_triples.append(f"({subj_label}, {pred_label}, {obj_label})")
                        elif isinstance(step, tuple) and len(step) == 3:
                            # Fallback for unenriched tuples (correct order)
                            subj, pred, obj = step
                            path_triples.append(f"({subj}, {pred}, {obj})")
                    
                    if path_triples:
                        # Join path steps with arrows to show multi-hop connection
                        pretty_string = "->".join(path_triples)
                        all_triples.append({
                            "path": path,
                            "pretty_string": pretty_string
                        })
        # Handle EXP format or old format
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, tuple) and len(item) == 3:
                    # Old BFS format: single tuple
                    all_triples.append({
                        "subject": item[0],
                        "predicate": item[1],
                        "object": item[2],
                        "pretty_string": f"({item[0]}, {item[1]}, {item[2]})"
                    })
                elif isinstance(item, dict):
                    # EXP format: already a dictionary
                    all_triples.append(item)
        elif isinstance(data, dict):
            all_triples.append(data)
    
    if not all_triples:
        logger.warning(f"No triples generated for block {block_id} from {len(entities_to_process)} entities")
    else:
        logger.info(f"Generated {len(all_triples)} triples for block {block_id}")
    return all_triples


def generate_save_triples(group_retrieval_results: Dict, dataset: str, top_k: int = 3, max_blocks: Optional[int] = None, id_type: str = "QID", triple_generation_type: Optional[str] = None, top_k_entities: int = 3, blocking_method: str = "QG") -> Dict:
    """Generate triples for all blocks and add to the results"""
    logger.info("=== Generating triples for all blocks ===")
    
    # Copy the original structure
    updated_results = {
        "metadata": group_retrieval_results.get("metadata", {}),
        "blocks": {}
    }

    blocks = group_retrieval_results.get("blocks", {})
    total_blocks = len(blocks)
    blocks_with_qids = 0
    blocks_with_pids = 0
    if id_type == "QID":
        logger.warning("Proceeding with QIDs to generate triples.")
        for block_id, block_data in blocks.items():
            qids, _ = extract_ids_from_block(block_data, top_k)
            if qids:
                blocks_with_qids += 1
        logger.info(f"Found {blocks_with_qids} blocks with QIDs out of {total_blocks} total blocks")
    elif id_type == "PID":
        logger.warning("Proceeding with PIDs to generate triples.")
        for block_id, block_data in blocks.items():
            _, pids = extract_ids_from_block(block_data, top_k)
            if pids:
                blocks_with_pids += 1
        logger.info(f"Found {blocks_with_pids} blocks with PIDs out of {total_blocks} total blocks")
    else:
        logger.error(f"Invalid id_type specified: {id_type}. Must be 'QID' or 'PID'.")
    
    
    # Process blocks 
    block_items = list(blocks.items())
    
    for block_id, block_data in tqdm(block_items, desc="Processing blocks"):
        # Copy existing block data
        updated_block = block_data.copy()
        
        # Generate triples (function handles both EXP and BFS)
        relevant_triples = generate_triples_for_block(block_id, block_data, dataset, top_k, id_type, triple_generation_type, top_k_entities, blocking_method)
        
        # Add triples to the block data
        updated_block['relevant_triples'] = relevant_triples
        
        updated_results["blocks"][block_id] = updated_block
        
        if relevant_triples:
            logger.info(f"Processed block {block_id}: {len(relevant_triples)} triples added")
        
    
    return updated_results


def enrich_retrieval_results(context_type: str, top_k: int, dataset_key: str, partition: str, triple_id_type: str, triple_generation_type: Optional[str] = None, top_k_entities: int = 3, blocking_method: str = "QG", max_blocking_size: int = 6) -> Dict:
    """
    Enrich retrieval results with labels and descriptions for top-k items.
    Removes similarity_score and pretty_string fields.
    
    Args:
        retrieval_results: The loaded group_retrieval_results.json
        context_type: "qid", "pid", or "triple" to specify which type of context to enrich
        top_k: Number of top results to enrich (1 or 2)
        dataset_key: Identifier for the target dataset (used for file paths)
        partition: Dataset partition ("train", "valid", or "test")
        triple_id_type: "QID" or "PID", only relevant if context_type is "triple"
        triple_generation_type: Optional[str] = None  # "BFS" or "EXP (expansion)" Whether to use BFS search for triple generation
        top_k_entities: Number of top entities/properties to consider for triple generation
        blocking_method: str = "QG", Blocking method used (for future use)
        max_blocking_size: int = 6, Maximum blocking size used (for future use)
    
    Returns:
        Enriched retrieval results with cleaned up fields
    """
    
    # Define output file path

    if context_type=="triple":
        output_file = f"output/{dataset_key}/enriched_retrieval_results_{dataset_key}_{context_type}_{triple_generation_type}_{blocking_method}_{max_blocking_size}.json"
    else:
        output_file = f"output/{dataset_key}/enriched_retrieval_results_{dataset_key}_{context_type}_{top_k}_{blocking_method}_{max_blocking_size}.json"
    
    # Check if enriched file already exists
    if os.path.exists(output_file):
        logger.info(f"Loading existing enriched results from: {output_file}")
        with open(output_file, 'r', encoding='utf-8') as f:
            enriched_results = json.load(f)
        return enriched_results
    
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    logger.info(f"Enriched file not found. Generating enriched results for context_type: {context_type}...")
    
    # Load input retrieval results
    input_file = f"retrieval_outputs/{dataset_key}/{dataset_key}_{partition}_{blocking_method}_{max_blocking_size}_group_retrieval_results.json"
    group_retrieval_results = load_group_retrieval_results(input_file)
    
    # Create a copy of the structure
    enriched_results = {
        "metadata": group_retrieval_results.get("metadata", {}),
        "blocks": {}
    }
    
    blocks = group_retrieval_results.get("blocks", {})
    
        
    if context_type == "qid":
        # Process QID enrichment
        logger.info(f"Enriching QIDs with top_k={top_k}")
        
        blocks_with_qids = 0
        total_qids_enriched = 0
        
        for block_id, block_data in tqdm(blocks.items(), desc="Enriching QIDs"):
            # Create new enriched block with only necessary fields
            enriched_block = {
                "pairs": block_data.get("pairs", [])
            }
            
            if "relevant_qids" in block_data and isinstance(block_data["relevant_qids"], list):
                # Sort by similarity_score and take top_k
                sorted_qids = sorted(
                    block_data["relevant_qids"],
                    key=lambda x: x.get("similarity_score", 0),
                    reverse=True
                )
                top_qids_list = [item.get("QID") for item in sorted_qids[:top_k] if isinstance(item, dict) and "QID" in item]
                
                if top_qids_list:
                    blocks_with_qids += 1
                    enriched_qids = []
                    
                    # Create temp directory for intermediate files
                    output_path = f"retrieval_outputs/{dataset_key}/temp/{blocking_method}/{max_blocking_size}/enriched_{context_type}_{top_k}_{partition}_{block_id}.json"
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    # Fetch and save enriched Wikidata info
                    fetch_and_save_wikidata(top_qids_list, output_path)
                    
                    # Load the enriched info from saved file
                    with open(output_path, "r", encoding="utf-8") as f:
                        updated_group_qids_results = json.load(f)
                    
                    # Restructure to enriched format
                    for qid in top_qids_list:
                        qid_str = qid if isinstance(qid, str) else qid.get("QID")
                        qid_info = updated_group_qids_results.get(qid_str, {})
                        if qid_info:
                            enriched_qids.append({
                                "QID": qid_str,
                                "label": qid_info.get("label", qid_str),
                                "description": qid_info.get("description", "")
                            })
                            total_qids_enriched += 1
                    
                    # Only add enriched QIDs if we have them
                    if enriched_qids:
                        enriched_block["relevant_qids"] = enriched_qids
            
            enriched_results["blocks"][block_id] = enriched_block
        
        logger.info(f"QID enrichment completed. Processed {len(blocks)} blocks.")
        logger.info(f"Blocks with QIDs: {blocks_with_qids}")
        logger.info(f"Total QIDs enriched: {total_qids_enriched}")
        
    elif context_type == "pid":
        # Process PID enrichment
        logger.info(f"Enriching PIDs with top_k={top_k}")
        
        blocks_with_pids = 0
        total_pids_enriched = 0

        for block_id, block_data in tqdm(blocks.items(), desc="Enriching PIDs"):
            # Create new enriched block with only necessary fields
            enriched_block = {
                "pairs": block_data.get("pairs", [])
            }
            
            if "relevant_pids" in block_data and isinstance(block_data["relevant_pids"], list):
                # Sort by similarity_score and take top_k
                sorted_pids = sorted(
                    block_data["relevant_pids"],
                    key=lambda x: x.get("similarity_score", 0),
                    reverse=True
                )
                top_pids_list = [item.get("PID") for item in sorted_pids[:top_k] if isinstance(item, dict) and "PID" in item]

                if top_pids_list:
                    blocks_with_pids += 1
                    enriched_pids = []

                    # Create temp directory for intermediate files
                    output_path = f"retrieval_outputs/{dataset_key}/temp/enriched_{context_type}_{top_k}_{partition}_{block_id}_{blocking_method}.json"
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    # Fetch and save enriched Wikidata info
                    fetch_and_save_wikidata(top_pids_list, output_path)
                    
                    # Load the enriched info from saved file
                    with open(output_path, "r", encoding="utf-8") as f:
                        updated_group_pids_results = json.load(f)

                    # Restructure to enriched format
                    for pid in top_pids_list:
                        pid_str = pid if isinstance(pid, str) else pid.get("PID")
                        pid_info = updated_group_pids_results.get(pid_str, {})
                        if pid_info:
                            enriched_pids.append({
                                "PID": pid_str,
                                "label": pid_info.get("label", pid_str),
                                "description": pid_info.get("description", "")
                            })
                            total_pids_enriched += 1

                    # Only add enriched PIDs if we have them
                    if enriched_pids:
                        enriched_block["relevant_pids"] = enriched_pids

            enriched_results["blocks"][block_id] = enriched_block
        
        logger.info(f"PID enrichment completed. Processed {len(blocks)} blocks.")
        logger.info(f"Blocks with PIDs: {blocks_with_pids}")
        logger.info(f"Total PIDs enriched: {total_pids_enriched}")
    
    elif context_type == "triple":
        # Generate triples for all blocks
        logger.info(f"Generating triples with id_type: {triple_id_type} and generation method: {triple_generation_type}")
        
        updated_group_results = generate_save_triples(group_retrieval_results, dataset_key, top_k, max_blocks, triple_id_type, triple_generation_type, top_k_entities, blocking_method)
        
        # Enrich BFS triples if using BFS generation method
        if triple_generation_type == "BFS":
            logger.info("Enriching BFS triples with Wikidata labels and descriptions...")
            
            for block_id, block_data in tqdm(updated_group_results.get("blocks", {}).items(), desc="Enriching BFS triples"):
                relevant_triples = block_data.get('relevant_triples', [])
                
                if not relevant_triples:
                    continue
                
                # Collect all IDs that need enrichment
                all_ids = set()
                for triple in relevant_triples:
                    if isinstance(triple, dict):
                        subj = triple.get("subject", "")
                        pred = triple.get("predicate", "")
                        obj = triple.get("object", "")
                        
                        if subj:
                            all_ids.add(subj)
                        if pred:
                            all_ids.add(pred)
                        # Only add object if it's a Wikidata ID (starts with Q or P)
                        if obj and isinstance(obj, str) and (obj.startswith('Q') or obj.startswith('P')):
                            all_ids.add(obj)
                
                # Fetch labels and descriptions for all IDs in this block
                if all_ids:
                    enrichment_path = f"retrieval_outputs/{dataset_key}/temp/bfs_enrichment_{block_id}.json"
                    os.makedirs(os.path.dirname(enrichment_path), exist_ok=True)
                    
                    fetch_and_save_wikidata(list(all_ids), enrichment_path)
                    
                    with open(enrichment_path, 'r', encoding='utf-8') as f:
                        id_info = json.load(f)
                    
                    # Enrich each triple with labels
                    enriched_triples = []
                    for triple in relevant_triples:
                        if isinstance(triple, dict):
                            subj = triple.get("subject", "")
                            pred = triple.get("predicate", "")
                            obj = triple.get("object", "")
                            
                            subj_info = id_info.get(subj, {})
                            pred_info = id_info.get(pred, {})
                            
                            # Check if object is a Wikidata ID
                            obj_is_id = isinstance(obj, str) and (obj.startswith('Q') or obj.startswith('P'))
                            obj_info = id_info.get(obj, {}) if obj_is_id else {}
                            
                            subj_label = subj_info.get('label', subj)
                            pred_label = pred_info.get('label', pred)
                            obj_label = obj_info.get('label', obj) if obj_is_id else str(obj)
                            
                            # Create enriched triple with labels and pretty_string
                            enriched_triple = {
                                "subject": subj,
                                "predicate": pred,
                                "object": obj,
                                "subject_label": subj_label,
                                "predicate_label": pred_label,
                                "object_label": obj_label,
                                "pretty_string": f"({subj_label}, {pred_label}, {obj_label})"
                            }
                            enriched_triples.append(enriched_triple)
                    
                    # Update block with enriched triples
                    block_data['relevant_triples'] = enriched_triples
            
            logger.info("BFS triple enrichment completed.")
        
        # Generate statistics
        blocks_with_triples = 0
        total_triples = 0
        
        for block_id, block_data in updated_group_results.get("blocks", {}).items():
            relevant_triples = block_data.get('relevant_triples', [])
            if relevant_triples:
                blocks_with_triples += 1
                total_triples += len(relevant_triples)
        
        logger.info(f"Triple generation completed. Processed {len(updated_group_results.get('blocks', {}))} blocks.")
        logger.info(f"Blocks with triples: {blocks_with_triples}")
        logger.info(f"Total triples retrieved: {total_triples}")
        
        enriched_results = updated_group_results
    
    else:
        logger.error(f"Invalid context_type: {context_type}. Must be 'qid', 'pid', or 'triple'.")
        return enriched_results
    
    # Save enriched results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(enriched_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Enriched results saved to: {output_file}")
    
    return enriched_results

def build_context_from_retrieval(retrieval_info: List[Dict], context_type: str, top_k: Optional[int] = None) -> str:
    """
    Build context text from retrieval results.
    Updated to use label and description instead of pretty_string and similarity_score.
    
    Args:
        retrieval_info: List of enriched QID or PID information
        context_type: "qid" or "pid" or "triple"
        top_k: Number of top triples to include for context
        
    
    Returns:
        Formatted context string
    """
    if not retrieval_info:
        return ""
    
    # Limit results to top_k if specified (applies to all context types)
    if top_k is not None:
        retrieval_info = retrieval_info[:top_k]
    
    context_parts = []
    for item in retrieval_info:
        if context_type == "pid":
            pid = item.get("PID", "")
            label = item.get("label", pid)
            description = item.get("description", "")
            # Only include description if it exists and is different from label
            if description and description != label and not description.startswith(f"Wikidata property {pid}"):
                context_parts.append(f"{pid} ({label}: {description})")
            else:
                context_parts.append(f"{pid} ({label})")
        elif context_type == "qid":
            qid = item.get("QID", "")
            label = item.get("label", qid)
            description = item.get("description", "")
            # Only include description if it exists and is different from label
            if description and description != label and not description.startswith(f"Wikidata entity {qid}"):
                context_parts.append(f"{qid} ({label}: {description})")
            else:
                context_parts.append(f"{qid} ({label})")
        elif context_type == "triple":
            triple = item.get("pretty_string", "")
            if triple:
                context_parts.append(triple)

    context_text = "; ".join(context_parts)
    return context_text



def load_group_data(dataset_key: str, partition: str, blocking_method: str, max_blocking_size: int) -> Dict:
    """
    Load group retrieval results and subblocks with pairs.
    
    Returns: subblocks with pairs (as dictionary)
    """
    subblocks_path = f"blocking_outputs/{dataset_key}/{dataset_key}_{partition}_{blocking_method}_{max_blocking_size}_subblocks_with_pairs.json"
    
    with open(subblocks_path, "r", encoding="utf-8") as f:
        subblocks_with_pairs = json.load(f)

    return subblocks_with_pairs

def extract_pairs_from_subblocks(subblocks_data: Dict) -> pd.DataFrame:
    """
    Extract all pairs from subblocks data and convert to DataFrame.
    
    Args:
        subblocks_data: The loaded subblocks_with_pairs.json
    
    Returns:
        pd.DataFrame: DataFrame with columns [ltable_id, rtable_id, label, entityA, entityB, block_id]
    """
    all_pairs = []
    blocks = subblocks_data.get("blocks", {})
    
    for block_id, block_data in blocks.items():
        pairs = block_data.get("pairs", [])
        for pair in pairs:
            pair_data = {
                "ltable_id": pair.get("ltable_id"),
                "rtable_id": pair.get("rtable_id"), 
                "label": pair.get("label"),
                "entityA": pair.get("entity_a", ""),
                "entityB": pair.get("entity_b", ""),
                "block_id": block_id
            }
            all_pairs.append(pair_data)
    
    return pd.DataFrame(all_pairs)

def create_pair_to_block_mapping(subblocks_data: Dict) -> Dict[str, str]:
    """
    Create a mapping from pair (ltable_id-rtable_id) to block_id.
    
    Args:
        subblocks_data: The loaded subblocks_with_pairs.json
    
    Returns:
        Dict mapping "ltable_id-rtable_id" to block_id
    """
    pair_to_block = {}
    blocks = subblocks_data.get("blocks", {})
    
    for block_id, block_data in blocks.items():
        pairs = block_data.get("pairs", [])
        for pair in pairs:
            ltable_id = pair.get("ltable_id")
            rtable_id = pair.get("rtable_id")
            if ltable_id is not None and rtable_id is not None:
                pair_key = f"{ltable_id}-{rtable_id}"
                pair_to_block[pair_key] = block_id
    
    return pair_to_block

def prompt_generation(
    dataset_key: str,
    *,
    prompt_name: str,
    context_arg: Optional[Dict[str, Union[bool, str, int]]] = None,
    top_k: int,
    output_dir: str,
    partition: str = "test",
    triple_id_type: str = "QID",
    top_k_entities: int = 3,
    triple_generation_type: Optional[str] = None,
    blocking_method: str = "QG",
    max_blocking_size: int = 6
) -> List[List[Dict[str, str]]]:
    """
    Generates conversation messages for dataset entries using group retrieval results.
    
    Args:
        dataset_key: Identifier for the target dataset
        prompt_name: Name of the prompt template to use from PROMPT_TEMPLATES.".
        context_arg: Configuration for RAG context generation with fields:
            {
                "enabled": bool,
                "context_type": "pid" | "qid" | "both"  # which type of context to use
            }
        top_k: Number of top retrieval results to use (1 or 2)
        output_dir: Directory to save intermediate files if needed
        partition: Dataset partition to use ("train", "valid", or "test")
        triple_id_type: "QID" or "PID", only relevant if context_type is "triple"
        top_k_entities: int = 3, Number of top entities to consider for triple generation
        triple_generation_type: Optional[str] = None  # "BFS" or "EXP (expansion)" Whether to use BFS search for triple generation
        blocking_method: str = "QG", Blocking method used (for future use)

    Returns:
        List: each item is a message list that will be used by OpenAI
    """
    # Get the prompt template
    if prompt_name not in PROMPT_TEMPLATES:
        raise ValueError(f"Prompt '{prompt_name}' not found. Available prompts: {list(PROMPT_TEMPLATES.keys())}")
    
    prompt_template = PROMPT_TEMPLATES[prompt_name]['user']
    
    # Load group data
    print("Loading group data...")
    subblocks_with_pairs = load_group_data(dataset_key, partition, blocking_method, max_blocking_size)

    # Extract pairs from subblocks instead of using prepare_text_data
    print("Extracting pairs from subblocks...")
    pairs_df = extract_pairs_from_subblocks(subblocks_with_pairs)
    print(f"Found {len(pairs_df)} pairs in subblocks")
    
    # Only set context_type if context is enabled
    if context_arg and context_arg.get("enabled"):
        # Check if enriched file exists, otherwise enrich and save
        if context_arg['context_type'] == "triple":
            enriched_output_path = f"{output_dir}/{dataset_key}/enriched_retrieval_results_{dataset_key}_{context_arg['context_type']}_{triple_generation_type}_{blocking_method}_{max_blocking_size}.json"
        else:
            enriched_output_path = f"{output_dir}/{dataset_key}/enriched_retrieval_results_{dataset_key}_{context_arg['context_type']}_{top_k}_{blocking_method}_{max_blocking_size}.json"

        if os.path.exists(enriched_output_path):
            print(f"Loading existing enriched retrieval results from: {enriched_output_path}")
            with open(enriched_output_path, "r", encoding="utf-8") as f:
                enriched_results = json.load(f)
        else:
            print("Enriching retrieval results with Wikidata information...")
            enriched_results = enrich_retrieval_results(context_arg['context_type'], top_k, dataset_key, partition, triple_id_type, triple_generation_type, top_k_entities, blocking_method, max_blocking_size)

            # Save enriched results
            with open(enriched_output_path, "w", encoding="utf-8") as f:
                json.dump(enriched_results, f, ensure_ascii=False, indent=2)
            print(f"Enriched retrieval results saved to: {enriched_output_path}")
    

        # Create pair to block mapping
        pair_to_block = create_pair_to_block_mapping(subblocks_with_pairs)

        messages: List[List[Dict[str, str]]] = []

        
        # ---- Build per-row messages ----
        for row in pairs_df.itertuples():
            entity_a = row.entityA
            entity_b = row.entityB
            id_a = str(row.ltable_id)
            id_b = str(row.rtable_id)

            # Build context from retrieval results
            context_text = ""
            if context_arg and context_arg.get("enabled"):
                pair_key = f"{id_a}-{id_b}"
                block_id = pair_to_block.get(pair_key)
                
                if block_id:
                    blocks = enriched_results.get("blocks", {})
                    block_data = blocks.get(block_id, {})
                    
                    context_type = context_arg.get("context_type", "pid")
                    context_parts = []
                    
                    if context_type == "pid":
                        relevant_pids = block_data.get("relevant_pids", [])
                        if relevant_pids:
                            pid_context = build_context_from_retrieval(relevant_pids, "pid", context_arg.get("top_k"))
                            if pid_context:
                                context_parts.append(f"Properties: {pid_context}")
                    
                    elif context_type == "qid":
                        relevant_qids = block_data.get("relevant_qids", [])
                        if relevant_qids:
                            qid_context = build_context_from_retrieval(relevant_qids, "qid", context_arg.get("top_k"))
                            if qid_context:
                                context_parts.append(f"Entities: {qid_context}")

                    elif context_type == "triple":
                        relevant_triples = block_data.get("relevant_triples", [])
                        if relevant_triples:
                            triple_context = build_context_from_retrieval(relevant_triples, "triple", top_k)
                            if triple_context:
                                context_parts.append(f"Triples: {triple_context}")
                    
                    if context_parts:
                        context_text = ". ".join(context_parts)
                        
            
            # If no context available, use appropriate fallback message
            if not context_text:
                if context_arg and context_arg.get("enabled"):
                    context_text = "No avaliable relevant knowledge, please make the decision on your own."
                else:
                    context_text = " "
            
            # Format the prompt using the template
            formatted_prompt = prompt_template.format(entity_a, entity_b, context_text)
            
            # Create message list
            message = [{"role": "user", "content": formatted_prompt}]
            messages.append(message)

        return messages
    else:
        print("Context generation is disabled. Generating prompts without context...")
        messages: List[List[Dict[str, str]]] = []

        # ---- Build per-row messages without context ----
        for row in pairs_df.itertuples():
            entity_a = row.entityA
            entity_b = row.entityB
            id_a = str(row.ltable_id)
            id_b = str(row.rtable_id)

            # Format the prompt using the template without context
            formatted_prompt = prompt_template.format(entity_a, entity_b, " ")
            
            # Create message list
            message = [{"role": "user", "content": formatted_prompt}]
            messages.append(message)

        return messages


def evaluate_predictions(dataset_key: str, partition: str, results_path: str, blocking_method: str = "QG", max_blocking_size: int = 6) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score based on LLM predictions and ground truth labels.
    
    Args:
        dataset_key: Identifier for the target dataset
        results_path: Path to the results file
        partition: Dataset partition ("train", "valid", or "test")
        blocking_method: str = "QG", Blocking method used (for future use)
        max_blocking_size: Maximum size of blocks used during blocking (for future use)
    
    Returns:
        Dict containing precision, recall, and F1 scores
    """
    # Load results and ground truth  
    
    with open(results_path, "r", encoding="utf-8") as f:
        results_data = json.load(f)

    
    
    # Load ground truth from subblocks
    subblocks_with_pairs = load_group_data(dataset_key, partition, blocking_method, max_blocking_size)
    pairs_df = extract_pairs_from_subblocks(subblocks_with_pairs)
    
    
    # Extract predictions from results
    predictions_labels = []
    results = results_data.get("results", [])
    
    for result in results:
        # Handle both string and dict formats
        if isinstance(result, dict):
            response = result.get("response", "").strip().lower()
        elif isinstance(result, str):
            response = result.strip().lower()
        else:
            response = str(result).strip().lower()
        
        # Robust parsing logic to handle different response formats
        response_lower = response.lower().strip()
        
        decision_found = False
        
        if "match decision" in response_lower:
            # Find the position of "match decision"
            decision_start = response_lower.find("match decision")
            # Extract the part after "match decision"
            decision_part = response_lower[decision_start + len("match decision"):]
            
            # Remove common separators: colons, asterisks, quotes, newlines, etc.
            decision_part_cleaned = decision_part.replace(':', '').replace('*', '').replace('"', '').replace("'", '').replace('\\n', ' ').strip()
            
            # Get first word after cleaning
            words = decision_part_cleaned.split()
            first_word = words[0] if words else ""
            
            # Check for Yes/No
            if first_word == "yes":
                predictions_labels.append(1)
                decision_found = True
            elif first_word == "no":
                predictions_labels.append(0)
                decision_found = True
        
        # Fallback: look anywhere in the response for yes/no
        if not decision_found:
            if "yes" in response_lower or "matched" in response_lower:
                predictions_labels.append(1)
            elif "no" in response_lower or "not matched" in response_lower:
                predictions_labels.append(0)
            else:
                # Handle ambiguous cases - default to no match to be conservative
                print(f"Warning: Ambiguous response, defaulting to 0: {response[:50]}...")
                predictions_labels.append(0)
    
    # Get ground truth labels
    ground_truth = pairs_df['label'].tolist()
    
    # Ensure same length
    min_len = min(len(predictions_labels), len(ground_truth))
    predictions_labels = predictions_labels[:min_len]
    ground_truth = ground_truth[:min_len]

    f1 = f1_score(ground_truth, predictions_labels)
    conf_matrix = confusion_matrix(ground_truth, predictions_labels)
    precision = precision_score(ground_truth, predictions_labels)
    recall = recall_score(ground_truth, predictions_labels)
    accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / np.sum(conf_matrix) if np.sum(conf_matrix) > 0 else 0.0

    
    metrics_dict ={
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "total_pairs": len(ground_truth),
        "true_positives": int(conf_matrix[1, 1]),
        "true_negatives": int(conf_matrix[0, 0]),
        "false_positives": int(conf_matrix[0, 1]),
        "false_negatives": int(conf_matrix[1, 0])
    }
    return metrics_dict
    
    
def main(
        dataset: str,
        partition: str = "test",
        model = "gpt-4o-mini",
        blocking_method: str = "QG",
        max_blocking_size: int = 6
):
    output_dir = Path(f"./output")
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists
    # group_generation = True  # Whether to use group retrieval context

     # Configuration for context retrieval
    context_config = {
        "enabled": True,  # Set to False to disable context retrieval
        "context_type": "qid",  # "pid", "qid", or "triple"
        "top_k": 2   # Number of top retrieval results to use (1 or 2)
    }

 
    retrieval_outputs_dir = Path(f"retrieval_outputs/{dataset}")
    retrieval_outputs_dir.mkdir(exist_ok=True, parents=True)
    triple_generation_type = None
    
    if context_config["context_type"] == "triple":
        print("Enabling triple context generation...")
        # Triple ID type for generation if using triple context
        triple_id_type = "QID"  # "QID" or "PID", only relevant if context_type is "triple"
        triple_generation_type = "BFS"  # "BFS" or "EXP (expansion)" Whether to use BFS search for triple generation
        top_k_entities = 3  # Number of top entities/properties to use for triple generation

    if context_config["enabled"]:
        prompt_name = "rag4em"  # Use a prompt suitable for context
    else:
        prompt_name = "llm4em"  # Use a prompt without context

    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = setup_logging(dataset, partition, context_config, model, triple_generation_type, timestamp)
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    print(f"Processing dataset: {dataset}")
    print(f"Using prompt template: {prompt_name}")
    
    # Define messages output path early to check if it already exists
    if context_config["enabled"]:
        if context_config["context_type"] == "triple":
            messages_output_path = f"{output_dir}/{dataset}/messages_{partition}_{context_config['context_type']}_{triple_generation_type}_{context_config['top_k']}_{prompt_name}_{blocking_method}_{max_blocking_size}_{model}_group_rag.json"
        elif context_config["context_type"] in ["qid", "pid"]:
            messages_output_path = f"{output_dir}/{dataset}/messages_{partition}_{context_config['context_type']}_{context_config['top_k']}_{prompt_name}_{blocking_method}_{max_blocking_size}_{model}_group_rag.json"
    else:
        messages_output_path = f"{output_dir}/{dataset}/messages_{partition}_{prompt_name}_{model}_llm.json"

    print(f"Messages output path: {messages_output_path}")
    
    # Check if messages file already exists to skip triple generation
    if os.path.exists(messages_output_path):
        print(f"Messages file already exists: {messages_output_path}")
        print("Loading existing messages...")
        logger.info(f"Loading existing messages from: {messages_output_path}")
        with open(messages_output_path, "r", encoding="utf-8") as f:
            messages_data = json.load(f)
        messages_list = messages_data.get("messages", [])
        print(f"Loaded {len(messages_list)} messages from existing file")
        logger.info(f"Loaded {len(messages_list)} messages from existing file")
    else:
        print("Messages file not found. Generating new messages...")
        logger.info("Starting message generation...")
        msg_gen_start = time.time()
        
        # Generate messages with group retrieval context (this calls enrich_retrieval_results and BFS)
        messages_list = prompt_generation(
            dataset_key=dataset,
            prompt_name=prompt_name,
            context_arg=context_config,
            top_k=context_config["top_k"],
            output_dir=output_dir,
            partition=partition,
            triple_id_type=triple_id_type if context_config["context_type"] == "triple" else None,
            top_k_entities=top_k_entities if context_config["context_type"] == "triple" else None,
            triple_generation_type=triple_generation_type if context_config["context_type"] == "triple" else None,
            blocking_method=blocking_method,
            max_blocking_size=max_blocking_size
        )

        msg_gen_time = time.time() - msg_gen_start
        print(f"Generated {len(messages_list)} messages for processing")
        logger.info(f"Message generation completed in {msg_gen_time:.2f}s: generated {len(messages_list)} messages")
        
        # Save messages
        print("Saving generated messages...")
        os.makedirs(os.path.dirname(messages_output_path), exist_ok=True)
        with open(messages_output_path, "w", encoding="utf-8") as f:
            json.dump({
                "metadata": {
                    "dataset": dataset,
                    "model": model,
                    "prompt_name": prompt_name,
                    "total_messages": len(messages_list),
                    "context_config": context_config
                },
                "messages": messages_list
            }, f, ensure_ascii=False, indent=2)
        print(f"Messages saved to: {messages_output_path}")
        logger.info(f"Messages saved to: {messages_output_path}")


    if context_config["enabled"]:
        if context_config["context_type"] == "triple":          
            results_output_path = f"{output_dir}/{dataset}/results_{partition}_{context_config['context_type']}_{triple_generation_type}_{context_config['top_k']}_{prompt_name}_{blocking_method}_{max_blocking_size}_{model}-group_rag_{timestamp}.json"
        elif context_config["context_type"] in ["qid", "pid"]:
            results_output_path = f"{output_dir}/{dataset}/results_{partition}_{context_config['context_type']}_{context_config['top_k']}_{prompt_name}_{blocking_method}_{max_blocking_size}_{model}-group_rag_{timestamp}.json"
    else:
        results_output_path = f"{output_dir}/{dataset}/results_{partition}_{prompt_name}_llm_{model}_{timestamp}.json"
    
    # Check if results file already exists
    if os.path.exists(results_output_path):
        print(f"Results file found: {results_output_path}")
        print("Loading existing results...")
        with open(results_output_path, "r", encoding="utf-8") as f:
            results_data = json.load(f)
        results = results_data.get("results", [])
        print(f"Loaded {len(results)} existing results")
    else:
        print("Results file not found. Generating new results...")
        # Process requests with LLM
        print("Sending requests to LLM...")
        print(f"Total messages to process: {len(messages_list)}")
        logger.info(f"Starting LLM generation with {len(messages_list)} prompts")
        
        llm_start_time = time.time()

        if model in ["jellyfish-8b", "jellyfish-7b", "mistral-7b", "qwen-7b", "qwen3-4b", "qwen3-8b"]:
            print("Using local model processing...")
            logger.info(f"Using local model: {model}")
            results = process_local_request(model, messages_list)
        elif model in ["gpt-4o-mini"]:
            print("Using OpenAI API processing...")
            logger.info(f"Using OpenAI model: {model}")
            results = process_openai_requests(model, messages_list)
        elif model in ["gemini-2.0-flash-lite", "gemini-1.5-flash-8b"]:
            print("Using Google Gemini API processing...")
            logger.info(f"Using Gemini model: {model}")
            results = process_gemini_requests(model, messages_list)
        else:
            raise ValueError(f"Unsupported model: {model}")
        
        llm_time = time.time() - llm_start_time
        print(f"LLM generation took {llm_time:.2f}s for {len(messages_list)} prompts")
        logger.info(f"LLM generation completed in {llm_time:.2f}s")
        
        if messages_list:
            avg_time = llm_time / len(messages_list)
            print(f"Average time per prompt: {avg_time:.2f}s")
            logger.info(f"Average time per prompt: {avg_time:.2f}s ({len(messages_list)} prompts)")
        
        print(f"Received {len(results)} responses")
        logger.info(f"Received {len(results)} responses from model")
        
        # Save results
        results_save_start = time.time()
        with open(results_output_path, "w", encoding="utf-8") as f:
            json.dump({
                "metadata": {
                    "dataset": dataset,
                    "model": model,
                    "prompt_name": prompt_name,
                    "total_pairs": len(results),
                    "context_config": context_config
                },
                "results": results
            }, f, ensure_ascii=False, indent=2)
        results_save_time = time.time() - results_save_start
        
        print(f"Results saved to: {results_output_path}")
        logger.info(f"Results saved to {results_output_path} in {results_save_time:.2f}s")
    
    # Evaluation
    print("Evaluating predictions...")
    metrics_dict = evaluate_predictions(dataset, partition, results_output_path, blocking_method=blocking_method, max_blocking_size=max_blocking_size)
    
    print("\n=== Evaluation Results ===")
    print(f"Precision: {metrics_dict['precision']:.4f}")
    print(f"Recall: {metrics_dict['recall']:.4f}")
    print(f"F1 Score: {metrics_dict['f1']:.4f}")
    print(f"Accuracy: {metrics_dict['accuracy']:.4f}")
    print(f"Total Pairs: {metrics_dict['total_pairs']}")
    print(f"True Positives: {metrics_dict['true_positives']}")
    print(f"True Negatives: {metrics_dict['true_negatives']}")
    print(f"False Positives: {metrics_dict['false_positives']}")
    print(f"False Negatives: {metrics_dict['false_negatives']}")
    
    # Save evaluation results
    if context_config["enabled"]:
        if context_config["context_type"] == "triple":          
            eval_output_path = f"{output_dir}/{dataset}/evaluation_{partition}_{context_config['context_type']}_{triple_generation_type}_{context_config['top_k']}_{prompt_name}_{blocking_method}_{max_blocking_size}_{model}_group_rag_{timestamp}.json"
        elif context_config["context_type"] in ["qid", "pid"]:
            eval_output_path = f"{output_dir}/{dataset}/evaluation_{partition}_{context_config['context_type']}_{context_config['top_k']}_{prompt_name}_{blocking_method}_{max_blocking_size}_{model}_group_rag_{timestamp}.json"
    else:
        eval_output_path = f"{output_dir}/{dataset}/evaluation_{partition}_{prompt_name}_{model}_llm_{timestamp}.json"
    with open(eval_output_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "dataset": dataset,
                "model": model,
                "prompt_name": prompt_name,
                "context_config": context_config
            },
            "metrics": metrics_dict
        }, f, ensure_ascii=False, indent=2)
    
    print(f"Evaluation results saved to: {eval_output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RAG4EM Pipeline"
    )
    parser.add_argument(
        "--dataset",
        "-d",
        required=True,
        help='Dataset name (e.g. "itam", "abt", "amgo", "beer", "dbac", "dbgo", "foza", "waam", "wdc")',
    )
    parser.add_argument(
        "--partition",
        "-p",
        default="test",
        choices=["train", "test", "valid"],
        help='Data partition to use (default: "test")',
    )
    parser.add_argument(
        "--model",
        "-m",
        default="gpt-4o-mini",
        help='Dataset name (e.g. "gpt-4o-mini", "jellyfish-8b", "jellyfish-7b", "mistral-7b", "qwen3-4b", "qwen3-8b", "gemini-2.0-flash-lite")',
    )
    parser.add_argument(
        "--blocking_method",
        "-b",
        default="QG",
        choices=["SB", "QG", "EQG", "SA", "ESA"],
        help='Blocking method to use (default: "QG")',
    )
    parser.add_argument(
        "--max_blocking_size",
        "-maxb",
        type=int,
        default=6,
        help="Maximum blocking size to process (default: 6)",
    )

    args = parser.parse_args()

    main(
        args.dataset,
        args.partition,
        args.model,
        args.blocking_method,
        args.max_blocking_size
    )
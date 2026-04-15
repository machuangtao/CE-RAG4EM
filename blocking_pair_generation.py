import argparse
import json
import os
import pandas as pd
import logging
from pathlib import Path
from typing import Optional, List, Dict, Union, Literal, Tuple
from datetime import datetime
import traceback
import time
from data_utils.data_handler import (
    prepare_structured_data,
    structured_to_text_data,
    prepare_text_data,
    _load_raw_data,
    entity_to_text
)
from data_utils.context_generation import query_generation_from_block
from data_utils.wiki_query import triplet_by_id_async
from tqdm import tqdm

from data_utils import (
    prepare_text_data,
    wiki_query_execution,
    PROMPT_TEMPLATES,
    fetch_and_save_relevant_kg20c_entities,
)
from data_utils.wiki_query import (
    fetch_and_save_relevant_ids,
    fetch_and_save_wikidata,
    fetch_and_save_triplets
)

from model_utils import (EntityMatchPrompt, process_requests)

# Module logger placeholder; configured in setup_logging
logger = logging.getLogger("block_retrieval")
logger.addHandler(logging.NullHandler())

# Setup logging
class LoggerWriter:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level

    def write(self, message):
        if message.strip():
            self.level(message.strip())

    def flush(self):
        pass

def setup_logging(dataset, partition: str) -> Tuple[logging.Logger, str]:
    """Setup logging to both console and file"""
    global logger

    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{dataset}/block_retrieval_{dataset}_{partition}_{timestamp}.log"

    # Ensure the dataset-specific log directory exists
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Configure logger explicitly to avoid relying on basicConfig
    logger = logging.getLogger("block_retrieval")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Remove any existing handlers to avoid duplication
    logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")

    return logger, log_file



def build_entities_text(entity_a: str, entity_b: str) -> str:
    return (
        f"The first entity is {entity_a}.\n"
        f"The second entity is {entity_b}"
    )

def load_subblocks_from_json(json_path: str) -> Dict:
    """Load subblocks data from JSON file"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded subblocks from JSON: {json_path}")
        logger.info(f"Metadata: {data.get('metadata', {})}")
        
        return data['blocks']
        
    except Exception as e:
        logger.error(f"Error loading subblocks from JSON: {e}")
        raise

def save_subblocks_as_json(organized_blocks: Dict, output_path: str):
    """Save the organized blocks/subblocks as JSON"""
    try:
        # Create output directory if it doesn't exist
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data for JSON serialization
        json_data = {
            'metadata': {
                'total_blocks_subblocks': len(organized_blocks),
                'total_pairs': sum(block_data['size'] for block_data in organized_blocks.values()),
                'created_at': datetime.now().isoformat()
            },
            'blocks': organized_blocks
        }
        
        # Save as JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved subblocks data as JSON to: {output_file}")
        return str(output_file)
        
    except Exception as e:
        logger.error(f"Error saving subblocks as JSON: {e}")
        raise

def _kg20c_entities_to_qid_like(entities: List[Dict]) -> List[Dict]:
    """
    Convert KG20C retrieval records to a qid-like shape used by downstream code.
    """
    converted = []
    for item in entities or []:
        if not isinstance(item, dict):
            continue
        entity_id = item.get("entity_id", "")
        name = item.get("name", entity_id)
        entity_type = item.get("type", "")
        description = item.get("description") or (f"type: {entity_type}" if entity_type else "")
        converted.append(
            {
                "QID": entity_id,
                "label": name,
                "description": description,
                "similarity_score": item.get("similarity_score", 0.0),
                "source": item.get("source", "KG20C Astra Vector Search"),
            }
        )
    return converted


def save_retrieval_results_for_subblock(
        block_id: str,
        relevance_data: Dict,
        pair_map: Dict,
        retrieval_results_files_dir: str,
        kg_source: str = "wikidata",
):
    """Save retrieval results for a subblock - consolidated single JSON file with both QIDs and PIDs"""
    try:
        # Create output directory if it doesn't exist
        consolidated_file = Path(retrieval_results_files_dir)
        consolidated_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Use a single consolidated file for both QIDs and PIDs

        # Load existing consolidated data or create new
        if consolidated_file.exists():
            with open(consolidated_file, 'r', encoding='utf-8') as f:
                consolidated_data = json.load(f)
        else:
            consolidated_data = {
                'metadata': {
                    'created_at': datetime.now().isoformat(),
                    'total_blocks': 0
                },
                'blocks': {}
            }

        if kg_source == "kg20c":
            # KG20C backend returns query_id -> [entity results]
            entities = relevance_data.get(block_id, [])
            if isinstance(entities, dict) and block_id in entities:
                entities = entities[block_id]
            qid_array = _kg20c_entities_to_qid_like(entities if isinstance(entities, list) else [])
            pid_array = []
            extra_entities = entities if isinstance(entities, list) else []
        else:
            # Wikidata backend
            qid_relevance = relevance_data.get('relevant_qids', {})
            if isinstance(qid_relevance, dict) and block_id in qid_relevance:
                qid_array = qid_relevance[block_id]
            elif isinstance(qid_relevance, list):
                qid_array = qid_relevance
            else:
                qid_array = qid_relevance

            pid_relevance = relevance_data.get('relevant_pids', {})
            if isinstance(pid_relevance, dict) and block_id in pid_relevance:
                pid_array = pid_relevance[block_id]
            elif isinstance(pid_relevance, list):
                pid_array = pid_relevance
            else:
                pid_array = pid_relevance
            extra_entities = []

        # Add current block data with both QIDs and PIDs
        consolidated_data['blocks'][block_id] = {
            'relevant_qids': qid_array,
            'relevant_pids': pid_array,
            'kg_source': kg_source,
            'relevant_entities': extra_entities,
        }

        # Update metadata
        consolidated_data['metadata']['total_blocks'] = len(consolidated_data['blocks'])
        consolidated_data['metadata']['last_updated'] = datetime.now().isoformat()
        consolidated_data['metadata']['kg_source'] = kg_source
        
        # Save consolidated file
        with open(consolidated_file, 'w', encoding='utf-8') as f:
            json.dump(consolidated_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved retrieval results for {block_id} to consolidated file: {consolidated_file}")
        
    except Exception as e:
        logger.error(f"Error saving retrieval results for {block_id}: {e}")
        raise

def create_subblocks(matching_pairs_df: pd.DataFrame, table_a: pd.DataFrame, table_b: pd.DataFrame, max_block_size: int = 6) -> Tuple[Dict, str]:
    """Organize matching pairs by blocks and create subblocks if needed"""
    logger.info(f"Organizing pairs by blocks with max block size: {max_block_size}")
    
    # Group by block_id
    blocks = {}

    for _, row in matching_pairs_df.iterrows():
        block_id = row['block_id']
        ltable_id = row['ltable_id']
        rtable_id = row['rtable_id']
        label = row['label']
        
        # Get entity data
        entity_a = table_a[table_a['id'] == ltable_id]
        entity_b = table_b[table_b['id'] == rtable_id]
        
        if not entity_a.empty and not entity_b.empty:
            # Convert individual entities to text
            entity_a_text = entity_to_text(entity_a.iloc[0])
            entity_b_text = entity_to_text(entity_b.iloc[0])
            
            if block_id not in blocks:
                blocks[block_id] = []
            
            blocks[block_id].append({
                'ltable_id': int(ltable_id),  
                'rtable_id': int(rtable_id),  
                'label': int(label),          
                'entity_a': entity_a_text,
                'entity_b': entity_b_text,
                'block_id': str(block_id)     
            })
    
    # Create subblocks for large blocks
    sub_blocks = {}
    subblock_stats = {
        'original_blocks': len(blocks),
        'small_blocks_kept': 0,
        'large_blocks_split': 0,
        'total_subblocks_created': 0
    }
    
    for block_id, pairs in blocks.items():
        if len(pairs) <= max_block_size:
            # Small block - keep as is
            block_key = f"block_{block_id}"
            sub_blocks[block_key] = {
                'pairs': pairs,
                'size': len(pairs),
                'is_subblock': False,
                'original_block_id': str(block_id),
                'subblock_index': None
            }
            subblock_stats['small_blocks_kept'] += 1
        else:
            # Large block - split into subblocks
            num_subblocks = (len(pairs) + max_block_size - 1) // max_block_size
            subblock_stats['large_blocks_split'] += 1
            subblock_stats['total_subblocks_created'] += num_subblocks
            
            for i in range(num_subblocks):
                start_idx = i * max_block_size
                end_idx = min((i + 1) * max_block_size, len(pairs))
                subblock_pairs = pairs[start_idx:end_idx]
                
                subblock_key = f"block_{block_id}_sub_{i}"
                sub_blocks[subblock_key] = {
                    'pairs': subblock_pairs,
                    'size': len(subblock_pairs),
                    'is_subblock': True,
                    'original_block_id': str(block_id),
                    'subblock_index': i,
                    'parent_block': str(block_id)
                }
    
    logger.info(f"Block organization statistics:")
    logger.info(f"  Original blocks: {subblock_stats['original_blocks']}")
    logger.info(f"  Small blocks kept as-is: {subblock_stats['small_blocks_kept']}")
    logger.info(f"  Large blocks split: {subblock_stats['large_blocks_split']}")
    logger.info(f"  Total subblocks created: {subblock_stats['total_subblocks_created']}")
    logger.info(f"  Final blocks/subblocks: {len(sub_blocks)}")
    
    return sub_blocks

def retrieve_kg_context_for_block(
        dataset: str,
        partition: str,
        query_type: str,
        block_data: Dict,
        block_id: str,
        blocking_method: str,
        max_block_size: int,
        kg_source: str = "wikidata",
) -> Dict:
    """Retrieve KG context for a specific block or subblock with single concatenated query"""
    logger.info(f"Retrieving KG context for {block_id} with {block_data['size']} pairs")
    retrieval_top_k = 10 if kg_source == "kg20c" else 10

    retrieval_outputs_dir = Path(f"retrieval_outputs/{dataset}")
    retrieval_results_files_dir = (
        f"{retrieval_outputs_dir}/"
        f"{dataset}_{partition}_{blocking_method}_{max_block_size}_{kg_source}_rgroup_retrieval_results.json"
    )

    retrieval_outputs_dir.mkdir(exist_ok=True, parents=True)

    try:
        pairs = block_data['pairs']
        
        # Create pair queries in the form "What are {entityA}, and {entityB}?"
        pair_queries = []
        
        # Map entities to their IDs for later reference
        entity_a_map = {}  # ltable_id -> entity_a
        entity_b_map = {}  # rtable_id -> entity_b
        pair_map = {}      # pair_id -> entity pair info
        
        for pair in pairs:
            ltable_id = str(pair['ltable_id'])
            rtable_id = str(pair['rtable_id'])
            pair_id = f"{ltable_id}_{rtable_id}"
            
            entity_a_text = pair['entity_a']
            entity_b_text = pair['entity_b']
            
            # Create query for this pair in the specified format
            pair_query = query_generation_from_block(dataset, partition, query_type, entity_a_text, entity_b_text)
            pair_queries.append(pair_query)
            
            # Store mappings for later reference
            entity_a_map[ltable_id] = entity_a_text
            entity_b_map[rtable_id] = entity_b_text
            pair_map[pair_id] = {
                'entity_a': entity_a_text,
                'entity_b': entity_b_text,
                'ltable_id': ltable_id,
                'rtable_id': rtable_id,
                'pair_query': pair_query
            }
        
        # Concatenate all pair queries for the subblock
        concatenated_pair_queries = " ; ".join(pair_queries)
        
        logger.info(f"Created {len(pair_queries)} pair queries for subblock {block_id}")
        # logger.info(f"Sample pair queries: {pair_queries[:3]}")
        logger.info(f"Concatenated queries length: {len(concatenated_pair_queries)} characters")
        
        # Create single query dataframe for the entire subblock
        subblock_query_df = pd.DataFrame([{
            'subblock_id': block_id,
            'query': concatenated_pair_queries
        }])
        
        # Use temporary path for API call
        temp_root = f"retrieval_outputs/{dataset}/temp/{kg_source}"
        temp_relevance_path = (
            f"{temp_root}/{blocking_method}/{max_block_size}/"
            f"{block_id}_{partition}_{query_type}_{kg_source}_top{retrieval_top_k}_relevant_ids.json"
        )
        os.makedirs(os.path.dirname(temp_relevance_path), exist_ok=True)

        # Check if relevance data already exists
        should_fetch = True
        if os.path.exists(temp_relevance_path):
            logger.info(f"Loading existing relevance data from {temp_relevance_path}")
            try:
                with open(temp_relevance_path, 'r', encoding='utf-8') as f:
                    cached_relevance_data = json.load(f)
                if kg_source == "kg20c":
                    cached_hits = cached_relevance_data.get(block_id, [])
                    if isinstance(cached_hits, list) and len(cached_hits) >= retrieval_top_k:
                        should_fetch = False
                    else:
                        logger.info(
                            f"Cached KG20C retrieval has fewer than top {retrieval_top_k} hits; refetching {block_id}"
                        )
                else:
                    should_fetch = False
            except Exception as e:
                logger.warning(f"Failed to read cached relevance data for {block_id}: {e}. Refetching.")

        if should_fetch:
            # Execute single KG query for the entire subblock
            try:
                if kg_source == "kg20c":
                    fetch_and_save_relevant_kg20c_entities(
                        subblock_query_df,
                        "subblock_id",
                        "query",
                        temp_relevance_path,
                        top_k=retrieval_top_k,
                    )
                else:
                    fetch_and_save_relevant_ids(subblock_query_df, "subblock_id", "query", temp_relevance_path)
                logger.info(f"Successfully fetched subblock relevance data")
            except Exception as e:
                logger.warning(f"Failed to fetch subblock relevance data: {e}")
                # Create empty relevance data to continue processing
                with open(temp_relevance_path, 'w') as f:
                    if kg_source == "kg20c":
                        json.dump({}, f)
                    else:
                        json.dump({'relevant_qids': {}, 'relevant_pids': {}}, f)
        
        # Load relevance data from temporary file
        relevance_data = {}
        if os.path.exists(temp_relevance_path):
            with open(temp_relevance_path, 'r') as f:
                relevance_data = json.load(f)
            # Clean up temporary file
            # os.remove(temp_relevance_path)
        
        # Save consolidated subblock retrieval results
        save_retrieval_results_for_subblock(
            block_id,
            relevance_data,
            pair_map,
            retrieval_results_files_dir,
            kg_source=kg_source,
        )

        logger.info(f"Saved retrieval results for {block_id}")
        
    except Exception as e:
        logger.error(f"Error retrieving KG context for {block_id}: {e}")
        logger.error(traceback.format_exc())
        return {'error': str(e)}





def main(
        dataset: str,
        partition: str = "test",
        blocking_method: str = "QG",
        max_block_size: int = 6,
        query_type: str = "pair",
        kg_source: str = "wikidata",
):
    # Setup logging
    global logger, log_file_path
    logger, log_file_path = setup_logging(dataset, partition)

    logger.info("=== Starting KG Retrieval Pipeline ===")
    logger.info(f"KG source: {kg_source}")
    
    # Step 1: Create subblocks
    logger.info("=== Step 1: Creating subblocks ===")
    matching_pairs_path = f"blocking_outputs/{dataset}/{dataset}_{partition}_{blocking_method}_matching_pairs.csv"
    matching_pairs_df = pd.read_csv(matching_pairs_path)
    
    # Log the actual columns to debug
    logger.info(f"CSV columns: {matching_pairs_df.columns.tolist()}")
    
    table_a, table_b, loaded_partition = _load_raw_data(dataset, partition)
    sub_blocks = create_subblocks(matching_pairs_df, table_a, table_b, max_block_size)

    # Save subblocks as JSON
    json_file_path = save_subblocks_as_json(sub_blocks, f"blocking_outputs/{dataset}/{dataset}_{partition}_{blocking_method}_{max_block_size}_subblocks_with_pairs.json")

    logger.info(f"Created {len(sub_blocks)} blocks/subblocks")

    # Step 2: Retrieval for all blocks
    logger.info("=== Step 2: KG retrieval for all blocks ===")
    kg_results = {}
    start_time = datetime.now()
    for block_id, block_data in tqdm(sub_blocks.items(), desc="Processing blocks"):
        block_start_time = datetime.now()
        logger.info(f"Retrieving KG context for {block_id}")
        kg_result = retrieve_kg_context_for_block(
            dataset,
            partition,
            query_type,
            block_data,
            block_id,
            blocking_method,
            max_block_size,
            kg_source=kg_source,
        )
        kg_results[block_id] = kg_result
        block_time = (datetime.now() - block_start_time).total_seconds()
        logger.info(f"Completed {block_id} in {block_time:.2f} seconds")

    # Log total retrieval time
    total_elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"Total retrieval time: {total_elapsed / 60:.2f} minutes ({total_elapsed:.2f} seconds)")
    
    
    logger.info("=== Pipeline completed successfully ===")
    
    # Create summary dictionary
    total_pairs = sum(block_data['size'] for block_data in sub_blocks.values())
    summary = {
        'total_blocks_processed': len(sub_blocks),
        'total_pairs_processed': total_pairs,
        'subblocks_json_file': json_file_path
    }
    
    print("\n=== Final Summary ===")
    print(f"Total blocks processed: {summary['total_blocks_processed']}")
    print(f"Total pairs processed: {summary['total_pairs_processed']}")
    print(f"Subblocks JSON saved to: {summary['subblocks_json_file']}")
    print(f"Log file saved to: {log_file_path}")
    
    logger.info(f"Pipeline completed. Log file saved to: {log_file_path}")
        
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="KG Retrieval Pipeline"
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
        "--blocking_method",
        "-b",
        default="QG",
        choices=["SB", "QG", "EQG", "SA", "ESA"],
        help='Blocking method to use (default: "QG")',
    )
    parser.add_argument(
        "--max_block_size",
        "-m",
        type=int,
        default=6,
        help="Maximum block size (default: 6)",
    )
    parser.add_argument(
        "--query_type",
        "-q",
        default="pair",
        choices=["entity", "pair"],
        help='Query type (default: "pair")',
    )
    parser.add_argument(
        "--kg_source",
        default="wikidata",
        choices=["wikidata", "kg20c"],
        help='Knowledge graph source for retrieval (default: "wikidata")',
    )

    args = parser.parse_args()

    main(
        dataset=args.dataset,
        partition=args.partition,
        blocking_method=args.blocking_method,
        max_block_size=args.max_block_size,
        query_type=args.query_type,
        kg_source=args.kg_source,
    )

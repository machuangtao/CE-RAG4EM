import argparse
import os
import sys
import time
import copy
import json
import logging
import numpy as np
import pandas as pd
from data_utils.data_handler import _load_raw_data
from collections import defaultdict
from pyjedai.datamodel import Data
from pyjedai.block_cleaning import BlockPurging, BlockFiltering
from pyjedai.comparison_cleaning import CardinalityEdgePruning, BLAST
from pyjedai.block_building import (
    StandardBlocking,
    ExtendedQGramsBlocking,
    ExtendedSuffixArraysBlocking,
    QGramsBlocking,
    SuffixArraysBlocking
)

# Global logger variable
logger = None
log_filename = None

def setup_logging(dataset, partation):
    """Set up comprehensive logging to capture all output"""
    global logger, log_filename

    log_filename = f"logs/{dataset}/blocking_{dataset}_{partation}_log_{time.strftime('%Y%m%d_%H%M%S')}.log"

    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    # Remove any existing handlers to avoid duplication
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Configure logging with detailed format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Force reconfiguration
    )
    
    logger = logging.getLogger(__name__)
    return logger, log_filename

def log_message(message, level=logging.INFO):
    """Log a message to both file and console"""
    if logger:
        logger.log(level, message)
    else:
        print(message)

def log_print(*args, **kwargs):
    """Enhanced print function that logs everything"""
    # Convert all arguments to strings and join them
    message = ' '.join(str(arg) for arg in args)
    
    # Handle kwargs like sep, end, etc.
    sep = kwargs.get('sep', ' ')
    end = kwargs.get('end', '\n')
    
    if len(args) > 1:
        message = sep.join(str(arg) for arg in args)
    
    # Log the message
    log_message(message)


# Define blocking methods
classic_method_dict = {
    'SB': StandardBlocking(),
    'QG': QGramsBlocking(),
    'EQG': ExtendedQGramsBlocking(),
    'SA': SuffixArraysBlocking(),
    'ESA': ExtendedSuffixArraysBlocking()
}

classic_method_name = {
    'SB': 'StandardBlocking',
    'EQG': 'ExtendedQGramsBlocking',
    'ESA': 'ExtendedSuffixArraysBlocking',
    'QG': 'QGramsBlocking',
    'SA': 'SuffixArraysBlocking'
}

def validate_blocking_attributes(left_df, right_df, blocking_attributes):
    """Validate that blocking attributes exist in both datasets"""
    left_cols = set(left_df.columns)
    right_cols = set(right_df.columns)

    log_print(f"Configured blocking attributes: {blocking_attributes}")
    
    # Check which attributes are available in both datasets
    available_attrs = []
    missing_attrs = []
    
    for attr in blocking_attributes:
        if attr in left_cols and attr in right_cols:
            available_attrs.append(attr)
            log_print(f"  ✓ '{attr}' found in both datasets")
        else:
            missing_attrs.append(attr)
            if attr not in left_cols:
                log_print(f"  ✗ '{attr}' missing from left dataset")
            if attr not in right_cols:
                log_print(f"  ✗ '{attr}' missing from right dataset")
    
    if missing_attrs:
        log_print(f"Warning: Missing attributes {missing_attrs}")
        log_print(f"Available left columns: {list(left_cols)}")
        log_print(f"Available right columns: {list(right_cols)}")
    
    return available_attrs

def extract_blocks_info(blocks):
    """Extract block information from PyJedAI blocks dictionary"""
    log_print("Extracting block information...")
    
    blocks_array = []
    total_entities = 0
    entities_in_blocks = set()
    
    # First, let's safely inspect what kind of object the blocks are
    log_print(f"Blocks object type: {type(blocks)}")
    
    # Check if blocks is a dictionary (most common in PyJedAI)
    if isinstance(blocks, dict):
        log_print(f"Blocks is a dictionary with {len(blocks)} entries")        
        # Process blocks dictionary
        for block_key, block_value in blocks.items():
            
            ltable_entities = []
            rtable_entities = []
            
            try:
                # PyJedAI Block objects have entities_D1 and entities_D2 attributes
                if hasattr(block_value, 'entities_D1') and hasattr(block_value, 'entities_D2'):
                    # Extract entities from OrderedSet or similar collections
                    if block_value.entities_D1 is not None:
                        ltable_entities = list(block_value.entities_D1)
                    if block_value.entities_D2 is not None:
                        rtable_entities = list(block_value.entities_D2)
                    
                    # Convert to integers if they're not already
                    ltable_entities = [int(x) for x in ltable_entities if x is not None]
                    rtable_entities = [int(x) for x in rtable_entities if x is not None]
                
            except Exception as e:
                log_print(f"Error processing block {block_key}: {str(e)}")
                ltable_entities = []
                rtable_entities = []
            
            # Update entity tracking
            entities_in_blocks.update([f"ltable_{id}" for id in ltable_entities])
            entities_in_blocks.update([f"rtable_{id}" for id in rtable_entities])
            
            # Add to blocks array
            blocks_array.append({
                'block_id': block_key,
                'ltable_ids': ltable_entities,
                'rtable_ids': rtable_entities,
                'ltable_count': len(ltable_entities),
                'rtable_count': len(rtable_entities)
            })
            
            total_entities += len(ltable_entities) + len(rtable_entities)
            
    
    # Handle case where blocks is not a dictionary
    else:
        log_print(f"Blocks object is not a dictionary: {type(blocks)}")
        return [], 0
    
    log_print(f"Extracted {len(blocks_array)} blocks")
    log_print(f"Total entity occurrences in blocks: {total_entities}")
    log_print(f"Unique entities in blocks: {len(entities_in_blocks)}")
    
    # Show some statistics
    non_empty_blocks = [b for b in blocks_array if b['ltable_count'] > 0 or b['rtable_count'] > 0]
    log_print(f"Non-empty blocks: {len(non_empty_blocks)}")
    
    if non_empty_blocks:
        avg_left_entities = sum(b['ltable_count'] for b in non_empty_blocks) / len(non_empty_blocks)
        avg_right_entities = sum(b['rtable_count'] for b in non_empty_blocks) / len(non_empty_blocks)
        log_print(f"Average entities per non-empty block: left={avg_left_entities:.2f}, right={avg_right_entities:.2f}")
    
    return blocks_array, len(entities_in_blocks)

def generate_block_pairs(blocks_array, match_df):
    """Generate pairwise matching pairs from blocks and filter it with ground truth labels"""

    # Create ground truth lookup with actual labels
    match_cols = list(match_df.columns)
    ground_truth_dict = {}
    
    # Build lookup from all rows in match_df (both positive and negative)
    for _, row in match_df.iterrows():
        key = (int(row[match_cols[0]]), int(row[match_cols[1]]))
        label = int(row[match_cols[2]])
        ground_truth_dict[key] = label
    
    log_print(f"Ground truth contains {len(ground_truth_dict)} total pairs")

    # Count positive and negative ground truth pairs
    positive_gt = sum(1 for label in ground_truth_dict.values() if label == 1)
    negative_gt = sum(1 for label in ground_truth_dict.values() if label == 0)
    log_print(f"Ground truth breakdown: {positive_gt} positive matches, {negative_gt} negative pairs")

    log_print("Generating pairwise matching pairs from blocks...")
    
    matching_pairs = []
    total_pairs_generated = 0
    all_candidate_pairs = set()
    pairs_with_ground_truth = 0

    # Fix for PyJedAI ID offset issue and determine the minimum right table ID to find offset
    all_ltable_ids_in_blocks = set()
    all_rtable_ids_in_blocks = set()
    
    for block in blocks_array:
        all_ltable_ids_in_blocks.update(block['ltable_ids'])
        all_rtable_ids_in_blocks.update(block['rtable_ids'])
    
    if all_ltable_ids_in_blocks and all_rtable_ids_in_blocks:
        max_ltable_id = max(all_ltable_ids_in_blocks)
        min_rtable_id = min(all_rtable_ids_in_blocks)
        
        # If right table IDs start after left table IDs, we have an offset
        if min_rtable_id > max_ltable_id:
            rtable_offset = min_rtable_id
            log_print(f"Detected PyJedAI ID offset: Right table IDs start at {rtable_offset}")
            log_print(f"Will remap right table IDs by subtracting {rtable_offset}")
        else:
            rtable_offset = 0
            log_print("No ID offset detected")
    else:
        rtable_offset = 0
    
    for block in blocks_array:
        block_id = block['block_id']
        ltable_ids = block['ltable_ids']
        rtable_ids = block['rtable_ids']

        # Generate all possible pairs within the block via Cartesian product
        for ltable_id in ltable_ids:
            for rtable_id in rtable_ids:
                total_pairs_generated += 1

                # Remap right table ID to match ground truth indexing
                remapped_rtable_id = rtable_id - rtable_offset
                pair_key = (ltable_id, remapped_rtable_id)
                
                all_candidate_pairs.add(pair_key)
                
                # Check if this pair has ground truth (either positive or negative)
                if pair_key in ground_truth_dict:
                    matching_pairs.append({
                        'ltable_id': ltable_id,
                        'rtable_id': remapped_rtable_id,
                        'label': ground_truth_dict[pair_key],
                        'block_id': block_id
                    })
                    pairs_with_ground_truth += 1
    
    # Convert matching_pairs to DataFrame first
    pairs_df = pd.DataFrame(matching_pairs)
    
    # Remove duplicate pairs
    initial_pair_count = len(pairs_df)
    pairs_df = pairs_df.drop_duplicates(subset=['ltable_id', 'rtable_id', 'label'], keep='first')
    final_unique_pair_count = len(pairs_df)
    duplicates_removed = initial_pair_count - final_unique_pair_count
    
    log_print(f"Total pairs generated from blocks: {total_pairs_generated}")
    log_print(f"Unique candidate pairs from blocks: {len(all_candidate_pairs)}")
    log_print(f"Pairs with explicit ground truth labels: {pairs_with_ground_truth}")
    log_print(f"Unique Pairs with explicit ground truth: {final_unique_pair_count}")
    
    # Show breakdown of generated pairs
    positive_generated = len(pairs_df[pairs_df['label'] == 1]) if len(pairs_df) > 0 else 0
    negative_generated = len(pairs_df[pairs_df['label'] == 0]) if len(pairs_df) > 0 else 0
    log_print(f"Generated pairs breakdown: {positive_generated} positive matches, {negative_generated} negative pairs")
    
    # Return both the filtered pairs and the total unique candidate count
    pairs_df.attrs['total_candidate_pairs'] = len(all_candidate_pairs)
    
    return pairs_df

def run_blocking_method(method, left_df, right_df, match_df, dataset, blocking_attributes, partition):
    """Run a specific blocking method and extract detailed block information"""
    
    log_print(f"Starting {classic_method_name[method]}...")
    
    # Validate and get available blocking attributes
    attr = validate_blocking_attributes(left_df, right_df, blocking_attributes)
    
    if not attr:
        error_msg = f"None of the specified blocking attributes {blocking_attributes} are available in both datasets"
        log_message(error_msg, logging.ERROR)
        raise ValueError(error_msg)
    
    log_print(f"Using blocking attributes: {attr}")
    
    # Create PyJedAI Data object
    log_print("Creating PyJedAI Data object...")
    data = Data(
        dataset_1=left_df.copy(), id_column_name_1='id',
        dataset_2=right_df.copy(), id_column_name_2='id',
    )
    
    # Clean dataset
    log_print("Cleaning dataset...")
    data.clean_dataset(
        remove_stopwords=False, 
        remove_punctuation=False, 
        remove_numbers=False, 
        remove_unicodes=True
    )
    
    # Build blocks
    log_print(f"Building blocks using {classic_method_name[method]} on attributes: {attr}...")
    start_time = time.time()
    
    bb = classic_method_dict[method]
    blocks = bb.build_blocks(
        copy.deepcopy(data), 
        attributes_1=attr, 
        attributes_2=attr, 
        tqdm_disable=True
    )

    log_print(f"Initial blocks created: {len(blocks)}")


    # Extract block information
    blocks_array, unique_entities = extract_blocks_info(blocks)
    
    # Save block information to JSON
    blocks_info = {
        'method': classic_method_name[method],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_blocks': len(blocks_array),
        'unique_entities_in_blocks': unique_entities,
        'left_table_size': len(left_df),
        'right_table_size': len(right_df),
        'blocking_attributes': attr,
        'blocks': blocks_array
    }

    json_filename = f"blocking_outputs/{dataset}/{dataset}_{partition}_{method}_blocks.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(blocks_info, f, indent=2, ensure_ascii=False)
    log_print(f"Block information saved to: {json_filename}")
    
    # Generate pairwise matching pairs from blocks and filter with ground truth
    log_print("Generating matching pairs from blocks...")
    matching_pairs = generate_block_pairs(blocks_array, match_df)

    
    # Statistics for ALL generated pairs (before filtering)
    log_print("\nSTATISTICS FOR ALL GENERATED PAIRS (BEFORE FILTERING):")
    log_print("-" * 50)
    
    end_time = time.time()
    runtime = end_time - start_time
    
    return blocks_array, blocks_info, matching_pairs, runtime

def calculate_blocking_statistics(blocks_info, pairs_df, left_df, right_df, match_df):
    """Calculate detailed blocking statistics"""
    log_print("Calculating blocking statistics...")
    
    total_possible_pairs = len(left_df) * len(right_df)
    total_blocks = blocks_info['total_blocks']
    unique_entities = blocks_info['unique_entities_in_blocks']
    
    # Use the total candidate pairs count (before ground truth filtering)
    total_candidate_pairs = pairs_df.attrs.get('total_candidate_pairs', len(pairs_df))
    total_blocks = blocks_info['total_blocks']
    unique_entities = blocks_info['unique_entities_in_blocks']


    # Check if pairs_df has the required 'label' column
    if 'label' in pairs_df.columns and len(pairs_df) > 0:
        true_matches_found = len(pairs_df[pairs_df['label'] == 1])
    else:
        true_matches_found = 0
        log_print("Warning: No 'label' column found or pairs_df is empty. Setting true_matches_found to 0.")
    
    # Get actual ground truth count (only positive matches)
    match_cols = list(match_df.columns)
    if len(match_cols) > 2:
        total_true_matches = len(match_df[match_df[match_cols[2]] == 1])
    else:
        total_true_matches = len(match_df)
    
    # Reduction Ratio (RR)
    RR = 1 - total_candidate_pairs / total_possible_pairs
    
    # Pair Completeness (PC) - recall
    PC = true_matches_found / total_true_matches
    
    # Pair Quality (PQ) - precision
    PQ = true_matches_found / total_candidate_pairs
    
    # F-measure
    F = 2 * PC * RR / (PC + RR)
   
    
    stats = {
        'total_blocks': total_blocks,
        'unique_entities_in_blocks': unique_entities,
        'total_possible_pairs': total_possible_pairs,
        'candidate_pairs': total_candidate_pairs,
        'true_matches_found': true_matches_found,
        'total_true_matches': total_true_matches,
        'reduction_ratio': RR,
        'pair_completeness': PC,
        'pair_quality': PQ,
        'f_measure': F
    }
    
    log_print(f"FINAL STATISTICS (AFTER FILTERING):")
    log_print(f"  Total blocks: {total_blocks}")
    log_print(f"  Unique entities in blocks: {unique_entities}")
    log_print(f"  Candidate pairs (filtered): {total_candidate_pairs}")
    log_print(f"  True matches found: {true_matches_found}")
    log_print(f"  Total true matches in dataset: {total_true_matches}")
    log_print(f"  Reduction Ratio: {RR:.4f}")
    log_print(f"  Pair Completeness: {PC:.4f}")
    log_print(f"  Pair Quality: {PQ:.5f}")
    log_print(f"  F-measure: {F:.4f}")
    
    return stats

def main(
        dataset: str,
        partition: str = "test",
):
    if dataset == "abt":
        blocking_attributes = ['name', 'description']
    elif dataset == "amgo":
        blocking_attributes = ['title', 'manufacturer']
    elif dataset == "beer":
        blocking_attributes = ['Beer_Name', 'Brew_Factory_Name', 'Style']
    elif dataset in ["dbac", "dbgo"]:
        blocking_attributes = ['title', 'authors', 'venue']
    elif dataset == "foza":
        blocking_attributes = ['name', 'addr', 'city', 'type']
    elif dataset == "itam":
        blocking_attributes = ['Song_Name', 'Artist_Name', 'Album_Name', 'CopyRight']
    elif dataset == "waam":
        blocking_attributes = ['title', 'category', 'brand', 'modelino']
    elif dataset == "wdc":
        blocking_attributes = ['brand', 'title', 'description']
    else:
        blocking_attributes = ['title', 'name']  # Default attributes

    # Create necessary directories
    os.makedirs(f"logs", exist_ok=True)
    os.makedirs(f"blocking_outputs/{dataset}", exist_ok=True)
    
     # Set up logging first
    logger, log_filename = setup_logging(dataset, partition)
    
    log_print("PYJEDAI BLOCKING METHODS - DETAILED BLOCK ANALYSIS")
    log_print(f"Starting execution at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_print(f"Processing {dataset} dataset with blocking methods...")
    log_print(f"Logging output to: {log_filename}")

    
    try:
        # Load data
        log_print("STEP 1: Loading data...")
        left_df, right_df, match_df = _load_raw_data(dataset, partition)
        
        log_print("\nDATASET STATISTICS:")
        log_print(f"Left table size: {len(left_df)}")
        log_print(f"Right table size: {len(right_df)}")
        log_print(f"Ground truth matches: {len(match_df)}")
        log_print(f"Total possible pairs: {len(left_df) * len(right_df)}")
        log_print("-" * 60)
        
        results = []
        all_methods_results = {}
        
        # Test each blocking method
        log_print("\nSTEP 2: Testing blocking methods...")
        for i, method in enumerate(classic_method_dict.keys(), 1):
            log_print(f"\n[{i}/{len(classic_method_dict)}] Testing {classic_method_name[method]}:")
            log_print("-" * 40)
            
            try:
                blocks_array, blocks_info, pairs_df, runtime = run_blocking_method(
                    method, left_df, right_df, match_df, dataset, blocking_attributes, partition
                )
                
                # Calculate statistics
                stats = calculate_blocking_statistics(blocks_info, pairs_df, left_df, right_df, match_df)
                
                # Store results
                result = {
                    'Method': classic_method_name[method],
                    'Runtime (s)': round(runtime, 4),
                    'Total_Blocks': stats['total_blocks'],
                    'Unique_Entities': stats['unique_entities_in_blocks'],
                    'Candidate_Pairs': stats['candidate_pairs'],
                    'True_Matches_Found': stats['true_matches_found'],
                    'RR (%)': round(100 * stats['reduction_ratio'], 2),
                    'PC (%)': round(100 * stats['pair_completeness'], 2),
                    'PQ (%)': round(100 * stats['pair_quality'], 2),
                    'F-measure (%)': round(100 * stats['f_measure'], 2)
                }
                results.append(result)
                
                # Store detailed results
                all_methods_results[method] = {
                    'blocks_array': blocks_array,
                    'blocks_info': blocks_info,
                    'pairs_df': pairs_df,
                    'stats': stats
                }
                
                log_print(f"RESULTS FOR {classic_method_name[method]}:")
                log_print(f"  Runtime: {runtime:.4f} seconds")
                log_print(f"  Total blocks: {stats['total_blocks']}")
                log_print(f"  Unique entities in blocks: {stats['unique_entities_in_blocks']}")
                log_print(f"  Candidate pairs: {stats['candidate_pairs']}")
                log_print(f"  True matches found: {stats['true_matches_found']}")
                log_print(f"  Reduction Ratio: {100 * stats['reduction_ratio']:.2f}%")
                log_print(f"  Pair Completeness: {100 * stats['pair_completeness']:.2f}%")
                log_print(f"  Pair Quality: {100 * stats['pair_quality']:.2f}%")
                log_print(f"  F-measure: {100 * stats['f_measure']:.2f}%")
                
                # Save matching pairs
                if len(pairs_df) > 0:
                    pairs_file = f"blocking_outputs/{dataset}/{dataset}_{partition}_{method}_matching_pairs.csv"
                    pairs_df.to_csv(pairs_file, index=False)
                    log_print(f"  Matching pairs saved to: {pairs_file}")
                else:
                    log_print(f"  No matching pairs to save for {method}")
                
            except Exception as e:
                log_message(f"Error with {method}: {str(e)}", logging.ERROR)
                import traceback
                error_trace = traceback.format_exc()
                log_message(f"Traceback: {error_trace}", logging.ERROR)
                continue
        
        # Create summary table
        if results:
            log_print("\n" + "=" * 80)
            log_print("FINAL SUMMARY RESULTS")
            log_print("=" * 80)
            results_df = pd.DataFrame(results)
            
            # Log the summary table line by line for better formatting
            summary_lines = results_df.to_string(index=False).split('\n')
            for line in summary_lines:
                log_print(line)
            
            # Save summary
            summary_file = f"blocking_outputs/{dataset}/{dataset}_{partition}_blocking_detailed_summary.csv"
            results_df.to_csv(summary_file, index=False)
            log_print(f"\nDetailed summary saved to: {summary_file}")
            
            # Save complete results as JSON
            complete_results_file = f"blocking_outputs/{dataset}/{dataset}_{partition}_complete_results.json"
            complete_results = {
                'dataset_info': {
                    'task': dataset,
                    'left_table_size': len(left_df),
                    'right_table_size': len(right_df),
                    'ground_truth_matches': len(match_df),
                    'total_possible_pairs': len(left_df) * len(right_df)
                },
                'methods_results': {}
            }
            
            for method, method_results in all_methods_results.items():
                complete_results['methods_results'][method] = {
                    'blocks_info': method_results['blocks_info'],
                    'statistics': method_results['stats'],
                    'matching_pairs_count': len(method_results['pairs_df'])
                }
            
            with open(complete_results_file, 'w', encoding='utf-8') as f:
                json.dump(complete_results, f, indent=2, ensure_ascii=False)
            log_print(f"Complete results saved to: {complete_results_file}")
            
            # Find best performing method
            if not results_df.empty:
                best_f_measure = results_df['F-measure (%)'].max()
                best_method = results_df[results_df['F-measure (%)'] == best_f_measure]['Method'].iloc[0]
                log_print(f"\nBest performing method: {best_method} (F-measure: {best_f_measure}%)")
        
        else:
            log_print("\nNo successful results to summarize.")
        
        log_print(f"\nExecution completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log_print(f"Complete execution log saved to: {log_filename}")
        log_print("=" * 80)
        
    except Exception as e:
        log_message(f"Fatal error in main execution: {str(e)}", logging.ERROR)
        import traceback
        error_trace = traceback.format_exc()
        log_message(f"Full traceback: {error_trace}", logging.ERROR)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Blocking Pipeline"
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

    args = parser.parse_args()

    main(
        dataset=args.dataset,
        partition=args.partition,
    )
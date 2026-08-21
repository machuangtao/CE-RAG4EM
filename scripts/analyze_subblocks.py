import json
from collections import defaultdict

MAX_SIZE = 6
INPUT_FILE = "blocking_outputs/dbgo/dbgo_test_QG_6_subblocks_with_pairs.json"

with open(INPUT_FILE) as f:
    data = json.load(f)

blocks = data["blocks"]

# Classify blocks
subblocks = []          # entries where is_subblock == True
non_subblocks = []      # entries where is_subblock == False (kept as-is, size <= MAX_SIZE)

for bid, binfo in blocks.items():
    if binfo.get("is_subblock", False):
        subblocks.append((bid, binfo))
    else:
        non_subblocks.append((bid, binfo))

# Group subblocks by parent_block to find original blocks that were split
parent_to_subblocks = defaultdict(list)
for bid, binfo in subblocks:
    parent_to_subblocks[binfo["parent_block"]].append(binfo)

num_split_blocks = len(parent_to_subblocks)          # blocks that were split
total_blocks = num_split_blocks + len(non_subblocks)  # all original blocks

# Fraction of blocks that exceeded max size and had to be split
frac_split = num_split_blocks / total_blocks if total_blocks else 0

# Average batch size:
#   - non-subblocks contribute their own size (they fit in one batch)
#   - split blocks contribute each subblock's size (each subblock is one batch)
batch_sizes = [binfo["size"] for _, binfo in non_subblocks]
for parent, subs in parent_to_subblocks.items():
    for s in subs:
        batch_sizes.append(s["size"])

avg_batch_size = sum(batch_sizes) / len(batch_sizes) if batch_sizes else 0

print(f"Total original blocks:             {total_blocks}")
print(f"Blocks split into subblocks:       {num_split_blocks}")
print(f"Blocks NOT split (size <= {MAX_SIZE}): {len(non_subblocks)}")
print(f"Fraction split:                    {frac_split:.4f} ({frac_split*100:.2f}%)")
print(f"Total batches (subblocks + kept):  {len(batch_sizes)}")
print(f"Average batch size:                {avg_batch_size:.4f}")

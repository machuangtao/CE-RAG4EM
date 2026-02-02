from pathlib import Path
import textwrap

# Base data directory
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / ".." / "data" / "raw"

# Dataset aliases
DATASETS = {
    "abt": "abt_buy",
    "amgo": "amazon_google",
    "beer": "beer",
    "dbac": "dblp_acm",
    "dbgo": "dblp_scholar",
    "foza": "fodors_zagat",
    "itam": "itunes_amazon",
    "waam": "walmart_amazon",
    "wdc": "wdc"
}

# Dataset Paths
DATASET_PATHS = {
    key: {
        "tableA": DATA_DIR / name / "tableA.csv",
        "tableB": DATA_DIR / name / "tableB.csv",
        "train": DATA_DIR / name / "train.csv",
        "valid": DATA_DIR / name / "valid.csv",
        "test": DATA_DIR / name / "test.csv",
        "dev": DATA_DIR / name / "dev.csv",
    }
    for key, name in DATASETS.items()
}

# Optional renaming for datasets (applies to both tableA and tableB)
COLUMN_RENAMES = {
    "beer": {
        "Beer_Name": "beer name",
        "Brew_Factory_Name": "brew factory",
        "Style": "style",
        "ABV": "ABV"
    },
    "foza": {
        "addr": "address"
    },
    "itam": {
        "Sone_Name": "song",
        "Artist_Name": "artist",
        "Album_Name": "album",
        "Genre": "genre",
        "Price": "price",
        "CopyRight": "copyright",
        "Time": "duration",
        "Released": "released date",
    },
    "waam": {
        "modelno": "model number",
    },
}

# Optional columns to be dropped from datasets (applies to both tableA and tableB)
COLUMN_DROP = {}

prompt_llm4em = textwrap.dedent("""\
You are an expert in entity matching, who is to determine whether these two given entity representations refer to the same entity.

## Input
Entity 1: {}
Entity 2: {}

## Instructions
1. Analyse each entity's semantics independently: consider key terms, roles, and context.
2. Perform a step-by-step logical comparison of the two entities.

## Output Format                                 
Match Decision: Yes / No
""")

prompt_rag4em = textwrap.dedent("""\
You are an expert in entity matching, who is to determine whether these two given entity representations refer to the same entity.
You are also provided additional information retrieved from WikiData, which might be helpful for your reasoning.


## Input
Entity 1: {}
Entity 2: {}
Additional Information (You can use this in your reasoning if available):\n{}

## Instructions
1. Analyse each entity's semantics independently: consider key terms, roles, and context.
2. Rank the relevance of each entry in the additional information, and only use it if it is helpful to making the decision.
3. Perform a step-by-step logical comparison of the two entities.

## Output Format
Match Decision: Yes / No
""")



PROMPT_TEMPLATES = {
    'llm4em': {
        'user': prompt_llm4em,
    },
    'rag4em': {
        'user': prompt_rag4em,
    },
}